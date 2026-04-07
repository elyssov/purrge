// ═══════════════════════════════════════════════════════════════
// PROMETHEUS ENGINE — Sparse Voxel Octree (SVO)
//
// Stores only non-empty voxels in a tree structure.
// 1024³ grid with 0.03% fill = megabytes, not gigabytes.
//
// Each node either:
//   - Is a leaf with a voxel value
//   - Has 8 children (octants)
//   - Is empty (null)
//
// Octant numbering (x,y,z bits):
//   0 = (0,0,0)  4 = (1,0,0)
//   1 = (0,0,1)  5 = (1,0,1)
//   2 = (0,1,0)  6 = (1,1,0)
//   3 = (0,1,1)  7 = (1,1,1)
// ═══════════════════════════════════════════════════════════════

/// Packed voxel data (same as flat grid)
#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct Voxel {
    pub packed: u32,  // material(8) | r(8) | g(8) | b(8)
    pub flags: u32,
}

impl Voxel {
    pub fn solid(mat: u8, r: u8, g: u8, b: u8) -> Self {
        Self {
            packed: (mat as u32) | ((r as u32) << 8) | ((g as u32) << 16) | ((b as u32) << 24),
            flags: 0,
        }
    }
    pub fn empty() -> Self { Self { packed: 0, flags: 0 } }
    pub fn is_empty(&self) -> bool { self.packed == 0 }
    pub fn is_solid(&self) -> bool { self.packed != 0 }
}

/// SVO node index (0 = null/empty)
type NodeIdx = u32;
const NULL_NODE: NodeIdx = 0;

/// An SVO node — either a branch (8 children) or a leaf (single voxel)
#[derive(Clone)]
enum Node {
    /// Empty space — no voxels in this octant
    Empty,
    /// Leaf — single voxel value (represents a 1×1×1 region)
    Leaf(Voxel),
    /// Uniform — entire octant filled with same voxel (LOD optimization)
    Uniform(Voxel),
    /// Branch — 8 child indices
    Branch([NodeIdx; 8]),
}

/// Sparse Voxel Octree
pub struct SVO {
    nodes: Vec<Node>,
    /// Grid dimension (must be power of 2)
    pub size: usize,
    /// Depth of tree (log2(size))
    depth: u32,
    /// Statistics
    pub voxel_count: usize,
    pub node_count: usize,
}

impl SVO {
    /// Create empty SVO for given grid size (must be power of 2)
    pub fn new(size: usize) -> Self {
        assert!(size.is_power_of_two(), "SVO size must be power of 2");
        let depth = (size as f64).log2() as u32;
        Self {
            nodes: vec![Node::Empty], // node 0 = root (empty initially)
            size,
            depth,
            voxel_count: 0,
            node_count: 1,
        }
    }

    /// Set a voxel at (x, y, z)
    pub fn set(&mut self, x: usize, y: usize, z: usize, voxel: Voxel) {
        if x >= self.size || y >= self.size || z >= self.size { return; }
        if voxel.is_empty() {
            self.remove(x, y, z);
            return;
        }
        self.set_recursive(0, x, y, z, self.size, voxel);
    }

    /// Get voxel at (x, y, z) — returns empty if not set
    pub fn get(&self, x: usize, y: usize, z: usize) -> Voxel {
        if x >= self.size || y >= self.size || z >= self.size { return Voxel::empty(); }
        self.get_recursive(0, x, y, z, self.size)
    }

    /// Remove voxel at (x, y, z)
    pub fn remove(&mut self, x: usize, y: usize, z: usize) {
        if x >= self.size || y >= self.size || z >= self.size { return; }
        self.remove_recursive(0, x, y, z, self.size);
    }

    /// Clear all voxels
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.nodes.push(Node::Empty);
        self.voxel_count = 0;
        self.node_count = 1;
    }

    /// Memory usage in bytes (approximate)
    pub fn memory_bytes(&self) -> usize {
        self.nodes.len() * std::mem::size_of::<Node>()
    }

    /// Export to flat array for GPU upload (for given region)
    /// Returns a flat array of size³ voxels
    pub fn export_flat(&self, size: usize) -> Vec<Voxel> {
        let mut flat = vec![Voxel::empty(); size * size * size];
        self.export_recursive(0, 0, 0, 0, self.size, size, &mut flat);
        flat
    }

    /// Export a sub-region to flat array
    pub fn export_region(&self, ox: usize, oy: usize, oz: usize, size: usize) -> Vec<Voxel> {
        let mut flat = vec![Voxel::empty(); size * size * size];
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let v = self.get(ox + x, oy + y, oz + z);
                    if v.is_solid() {
                        flat[z * size * size + y * size + x] = v;
                    }
                }
            }
        }
        flat
    }

    // ─── Internal recursive operations ───────────────────────

    fn octant_index(x: usize, y: usize, z: usize, half: usize) -> usize {
        let ix = if x >= half { 1 } else { 0 };
        let iy = if y >= half { 1 } else { 0 };
        let iz = if z >= half { 1 } else { 0 };
        ix * 4 + iy * 2 + iz
    }

    fn alloc_node(&mut self, node: Node) -> NodeIdx {
        let idx = self.nodes.len() as NodeIdx;
        self.nodes.push(node);
        self.node_count += 1;
        idx
    }

    fn set_recursive(&mut self, node_idx: NodeIdx, x: usize, y: usize, z: usize, size: usize, voxel: Voxel) {
        if size == 1 {
            // Leaf level
            match &self.nodes[node_idx as usize] {
                Node::Empty | Node::Uniform(_) => { self.voxel_count += 1; }
                Node::Leaf(_) => {} // replacing existing
                Node::Branch(_) => {}
            }
            self.nodes[node_idx as usize] = Node::Leaf(voxel);
            return;
        }

        let half = size / 2;
        let oct = Self::octant_index(x, y, z, half);
        let local_x = x % half;
        let local_y = y % half;
        let local_z = z % half;

        // Ensure this node is a branch
        match &self.nodes[node_idx as usize] {
            Node::Empty => {
                let children = [NULL_NODE; 8];
                self.nodes[node_idx as usize] = Node::Branch(children);
            }
            Node::Leaf(v) | Node::Uniform(v) => {
                // Expand leaf into branch with uniform children
                let v = *v;
                let mut children = [NULL_NODE; 8];
                for i in 0..8 {
                    children[i] = self.alloc_node(Node::Uniform(v));
                }
                self.nodes[node_idx as usize] = Node::Branch(children);
            }
            Node::Branch(_) => {}
        }

        // Get or create child
        if let Node::Branch(ref mut children) = self.nodes[node_idx as usize].clone() {
            if children[oct] == NULL_NODE {
                children[oct] = self.alloc_node(Node::Empty);
                self.nodes[node_idx as usize] = Node::Branch(*children);
            }
            let child = children[oct];
            self.set_recursive(child, local_x, local_y, local_z, half, voxel);
        }
    }

    fn get_recursive(&self, node_idx: NodeIdx, x: usize, y: usize, z: usize, size: usize) -> Voxel {
        match &self.nodes[node_idx as usize] {
            Node::Empty => Voxel::empty(),
            Node::Leaf(v) => *v,
            Node::Uniform(v) => *v,
            Node::Branch(children) => {
                let half = size / 2;
                let oct = Self::octant_index(x, y, z, half);
                if children[oct] == NULL_NODE {
                    Voxel::empty()
                } else {
                    self.get_recursive(children[oct], x % half, y % half, z % half, half)
                }
            }
        }
    }

    fn remove_recursive(&mut self, node_idx: NodeIdx, x: usize, y: usize, z: usize, size: usize) {
        if size == 1 {
            if self.nodes[node_idx as usize].is_solid_node() { self.voxel_count -= 1; }
            self.nodes[node_idx as usize] = Node::Empty;
            return;
        }

        if let Node::Branch(children) = self.nodes[node_idx as usize].clone() {
            let half = size / 2;
            let oct = Self::octant_index(x, y, z, half);
            if children[oct] != NULL_NODE {
                self.remove_recursive(children[oct], x % half, y % half, z % half, half);
            }
        }
    }

    fn export_recursive(&self, node_idx: NodeIdx, ox: usize, oy: usize, oz: usize,
                        size: usize, grid_size: usize, flat: &mut Vec<Voxel>) {
        match &self.nodes[node_idx as usize] {
            Node::Empty => {}
            Node::Leaf(v) => {
                if ox < grid_size && oy < grid_size && oz < grid_size {
                    flat[oz * grid_size * grid_size + oy * grid_size + ox] = *v;
                }
            }
            Node::Uniform(v) => {
                // Fill entire region
                for z in oz..(oz + size).min(grid_size) {
                    for y in oy..(oy + size).min(grid_size) {
                        for x in ox..(ox + size).min(grid_size) {
                            flat[z * grid_size * grid_size + y * grid_size + x] = *v;
                        }
                    }
                }
            }
            Node::Branch(children) => {
                let half = size / 2;
                for oct in 0..8 {
                    if children[oct] != NULL_NODE {
                        let cx = ox + if oct & 4 != 0 { half } else { 0 };
                        let cy = oy + if oct & 2 != 0 { half } else { 0 };
                        let cz = oz + if oct & 1 != 0 { half } else { 0 };
                        self.export_recursive(children[oct], cx, cy, cz, half, grid_size, flat);
                    }
                }
            }
        }
    }
}

impl Node {
    fn is_solid_node(&self) -> bool {
        matches!(self, Node::Leaf(v) | Node::Uniform(v) if v.is_solid())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_set_get() {
        let mut svo = SVO::new(64);
        svo.set(10, 20, 30, Voxel::solid(1, 255, 0, 0));
        let v = svo.get(10, 20, 30);
        assert!(v.is_solid());
        assert_eq!(svo.voxel_count, 1);

        // Empty position
        let e = svo.get(5, 5, 5);
        assert!(e.is_empty());
    }

    #[test]
    fn test_many_voxels() {
        let mut svo = SVO::new(256);
        // Fill a 50³ cube
        for z in 100..150 {
            for y in 100..150 {
                for x in 100..150 {
                    svo.set(x, y, z, Voxel::solid(1, 200, 100, 50));
                }
            }
        }
        assert_eq!(svo.voxel_count, 125_000);
        println!("50³ cube in 256³ SVO: {} nodes, {} bytes",
            svo.node_count, svo.memory_bytes());

        // Verify
        assert!(svo.get(125, 125, 125).is_solid());
        assert!(svo.get(0, 0, 0).is_empty());
    }

    #[test]
    fn test_export_flat() {
        let mut svo = SVO::new(64);
        svo.set(10, 20, 30, Voxel::solid(1, 255, 0, 0));
        svo.set(11, 20, 30, Voxel::solid(1, 0, 255, 0));

        let flat = svo.export_flat(64);
        assert!(flat[30*64*64 + 20*64 + 10].is_solid());
        assert!(flat[30*64*64 + 20*64 + 11].is_solid());
        assert!(flat[0].is_empty());
    }

    #[test]
    fn test_1024_sparse() {
        let mut svo = SVO::new(1024);
        // Place a few voxels far apart
        svo.set(0, 0, 0, Voxel::solid(1, 255, 0, 0));
        svo.set(1023, 1023, 1023, Voxel::solid(2, 0, 255, 0));
        svo.set(512, 512, 512, Voxel::solid(3, 0, 0, 255));

        assert_eq!(svo.voxel_count, 3);
        println!("3 voxels in 1024³ SVO: {} nodes, {} bytes ({:.2} KB)",
            svo.node_count, svo.memory_bytes(), svo.memory_bytes() as f64 / 1024.0);

        // Should be tiny!
        assert!(svo.memory_bytes() < 10_000); // less than 10 KB for 3 voxels in 1024³
    }

    #[test]
    fn test_hollow_sphere_in_1024() {
        let mut svo = SVO::new(1024);
        let cx = 512.0_f32;
        let cy = 512.0_f32;
        let cz = 512.0_f32;
        let r = 100.0_f32;
        let r2 = r * r;
        let shell = 3.0; // 3-voxel thick shell

        let mut count = 0;
        let ri = r as i32;
        for dz in -ri..=ri { for dy in -ri..=ri { for dx in -ri..=ri {
            let d2 = (dx*dx + dy*dy + dz*dz) as f32;
            if d2 <= r2 && d2 >= (r - shell) * (r - shell) {
                svo.set(
                    (cx as i32 + dx) as usize,
                    (cy as i32 + dy) as usize,
                    (cz as i32 + dz) as usize,
                    Voxel::solid(1, 200, 50, 50),
                );
                count += 1;
            }
        }}}

        println!("Hollow sphere r=100, shell=3 in 1024³:");
        println!("  Voxels: {}", count);
        println!("  SVO nodes: {}", svo.node_count);
        println!("  Memory: {:.1} KB", svo.memory_bytes() as f64 / 1024.0);
        println!("  Full sphere would be: {} voxels", (4.0/3.0 * std::f32::consts::PI * r * r * r) as usize);

        assert!(svo.voxel_count > 100_000); // shell has many voxels
        assert!(svo.memory_bytes() < 50_000_000); // but memory is reasonable
    }
}
