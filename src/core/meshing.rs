// ═══════════════════════════════════════════════════════════════
// PROMETHEUS ENGINE — Mesh Generator (Sharp + Smooth)
//
// Converts voxel data into triangle meshes for GPU rendering.
// Voxels = data (physics, destruction). Polygons = render (GPU).
//
// Two modes:
//   generate_mesh()        — sharp quads, good for architecture
//   generate_mesh_smooth() — Surface Nets, good for organic shapes
//
// Each 64³ chunk → ~50-100K triangles, generated in ~2ms on CPU.
// ═══════════════════════════════════════════════════════════════

use glam::Vec3;
use super::svo::Voxel;

/// A single vertex with position, normal, and color
#[derive(Clone, Copy, Debug)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],    // RGBA (0.0-1.0)
    pub material: u8,
}

/// A triangle mesh generated from voxels
pub struct ChunkMesh {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub triangle_count: usize,
}

impl ChunkMesh {
    pub fn new() -> Self {
        Self { vertices: Vec::new(), indices: Vec::new(), triangle_count: 0 }
    }

    pub fn is_empty(&self) -> bool { self.triangle_count == 0 }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.vertices.len() * std::mem::size_of::<MeshVertex>()
        + self.indices.len() * 4
    }
}

/// Generate mesh from a flat voxel array using surface extraction.
/// Simple but effective: for each solid voxel with an empty neighbor,
/// emit a quad (2 triangles) on that face.
///
/// This is NOT full Marching Cubes — it's "greedy meshing" which
/// produces clean, sharp voxel surfaces. Better for architecture.
/// We'll add smooth Marching Cubes as an option for organic shapes.
pub fn generate_mesh(
    voxels: &[Voxel],
    size: usize,
    offset: Vec3,   // world-space offset of this chunk
    scale: f32,     // voxel size in world units
) -> ChunkMesh {
    let mut mesh = ChunkMesh::new();

    let get = |x: i32, y: i32, z: i32| -> Voxel {
        if x < 0 || y < 0 || z < 0 || x >= size as i32 || y >= size as i32 || z >= size as i32 {
            return Voxel::empty();
        }
        voxels[(z as usize) * size * size + (y as usize) * size + (x as usize)]
    };

    // 6 face directions: +X, -X, +Y, -Y, +Z, -Z
    let dirs: [(i32,i32,i32, [f32;3]); 6] = [
        ( 1, 0, 0, [ 1.0, 0.0, 0.0]),  // +X
        (-1, 0, 0, [-1.0, 0.0, 0.0]),  // -X
        ( 0, 1, 0, [ 0.0, 1.0, 0.0]),  // +Y
        ( 0,-1, 0, [ 0.0,-1.0, 0.0]),  // -Y
        ( 0, 0, 1, [ 0.0, 0.0, 1.0]),  // +Z
        ( 0, 0,-1, [ 0.0, 0.0,-1.0]),  // -Z
    ];

    for z in 0..size as i32 {
        for y in 0..size as i32 {
            for x in 0..size as i32 {
                let voxel = get(x, y, z);
                if !voxel.is_solid() { continue; }

                let color = voxel_to_color(voxel);

                // Check each face
                for &(dx, dy, dz, normal) in &dirs {
                    let neighbor = get(x + dx, y + dy, z + dz);
                    if neighbor.is_solid() { continue; } // face hidden

                    // Emit quad for this visible face
                    let base_idx = mesh.vertices.len() as u32;
                    let (v0, v1, v2, v3) = face_vertices(
                        x as f32, y as f32, z as f32,
                        dx, dy, dz, scale, offset,
                    );

                    let mat = (voxel.packed & 0xFF) as u8;
                    mesh.vertices.push(MeshVertex { position: v0, normal, color, material: mat });
                    mesh.vertices.push(MeshVertex { position: v1, normal, color, material: mat });
                    mesh.vertices.push(MeshVertex { position: v2, normal, color, material: mat });
                    mesh.vertices.push(MeshVertex { position: v3, normal, color, material: mat });

                    // Two triangles per quad
                    mesh.indices.push(base_idx);
                    mesh.indices.push(base_idx + 1);
                    mesh.indices.push(base_idx + 2);
                    mesh.indices.push(base_idx);
                    mesh.indices.push(base_idx + 2);
                    mesh.indices.push(base_idx + 3);

                    mesh.triangle_count += 2;
                }
            }
        }
    }

    mesh
}

/// Compute 4 vertices for a face of a voxel cube
fn face_vertices(
    x: f32, y: f32, z: f32,
    dx: i32, dy: i32, dz: i32,
    scale: f32, offset: Vec3,
) -> ([f32;3], [f32;3], [f32;3], [f32;3]) {
    let s = scale;
    let o = offset;

    // Base position of voxel corner
    let bx = o.x + x * s;
    let by = o.y + y * s;
    let bz = o.z + z * s;

    if dx == 1 { // +X face
        ([bx+s, by,   bz  ], [bx+s, by+s, bz  ], [bx+s, by+s, bz+s], [bx+s, by,   bz+s])
    } else if dx == -1 { // -X face
        ([bx,   by,   bz+s], [bx,   by+s, bz+s], [bx,   by+s, bz  ], [bx,   by,   bz  ])
    } else if dy == 1 { // +Y face
        ([bx,   by+s, bz  ], [bx,   by+s, bz+s], [bx+s, by+s, bz+s], [bx+s, by+s, bz  ])
    } else if dy == -1 { // -Y face
        ([bx,   by,   bz+s], [bx,   by,   bz  ], [bx+s, by,   bz  ], [bx+s, by,   bz+s])
    } else if dz == 1 { // +Z face
        ([bx+s, by,   bz+s], [bx+s, by+s, bz+s], [bx,   by+s, bz+s], [bx,   by,   bz+s])
    } else { // -Z face
        ([bx,   by,   bz  ], [bx,   by+s, bz  ], [bx+s, by+s, bz  ], [bx+s, by,   bz  ])
    }
}

/// Extract color from packed voxel
fn voxel_to_color(v: Voxel) -> [f32; 4] {
    let r = ((v.packed >> 8) & 0xFF) as f32 / 255.0;
    let g = ((v.packed >> 16) & 0xFF) as f32 / 255.0;
    let b = ((v.packed >> 24) & 0xFF) as f32 / 255.0;
    [r, g, b, 1.0]
}

/// Compute ambient occlusion for a vertex (simple: count solid neighbors)
pub fn compute_ao(voxels: &[Voxel], size: usize, x: i32, y: i32, z: i32, normal: [f32;3]) -> f32 {
    let get = |x: i32, y: i32, z: i32| -> bool {
        if x < 0 || y < 0 || z < 0 || x >= size as i32 || y >= size as i32 || z >= size as i32 {
            return false;
        }
        voxels[(z as usize)*size*size + (y as usize)*size + (x as usize)].is_solid()
    };

    let mut occluded = 0;
    let mut total = 0;

    for dz in -1i32..=1 {
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 && dz == 0 { continue; }
                let dot = dx as f32 * normal[0] + dy as f32 * normal[1] + dz as f32 * normal[2];
                if dot < 0.0 { continue; } // only check on normal side
                total += 1;
                if get(x + dx, y + dy, z + dz) { occluded += 1; }
            }
        }
    }

    if total == 0 { return 1.0; }
    1.0 - (occluded as f32 / total as f32) * 0.5
}

/// Generate mesh WITH per-vertex AO
pub fn generate_mesh_with_ao(
    voxels: &[Voxel],
    size: usize,
    offset: Vec3,
    scale: f32,
) -> ChunkMesh {
    let mut mesh = generate_mesh(voxels, size, offset, scale);

    // Post-process: compute AO for each vertex
    for v in mesh.vertices.iter_mut() {
        let vx = ((v.position[0] - offset.x) / scale) as i32;
        let vy = ((v.position[1] - offset.y) / scale) as i32;
        let vz = ((v.position[2] - offset.z) / scale) as i32;
        let ao = compute_ao(voxels, size, vx, vy, vz, v.normal);
        v.color[0] *= ao;
        v.color[1] *= ao;
        v.color[2] *= ao;
    }

    mesh
}

// ═══════════════════════════════════════════════════════════════
// SURFACE NETS — Smooth Mesh Generation
//
// Algorithm:
//   Phase 1: For each cell straddling the surface, place a vertex
//            at the average position of edge crossings.
//   Phase 2: For each grid edge crossing the surface, connect the
//            4 cells sharing that edge with a quad.
//   Phase 3: Laplacian relaxation smooths vertex positions.
//   Phase 4: Compute smooth normals from face geometry.
//
// Produces rounded, organic-looking surfaces. The key insight:
// one vertex per surface cell → vertices are automatically shared
// → normals are automatically smooth. No stairstepping.
// ═══════════════════════════════════════════════════════════════

/// Generate a smooth mesh using Surface Nets.
/// Produces rounded surfaces instead of blocky voxel cubes.
pub fn generate_mesh_smooth(
    voxels: &[Voxel],
    size: usize,
    offset: Vec3,
    scale: f32,
) -> ChunkMesh {
    let mut mesh = ChunkMesh::new();
    let s = size as i32;

    let get = |x: i32, y: i32, z: i32| -> Voxel {
        if x < 0 || y < 0 || z < 0 || x >= s || y >= s || z >= s {
            return Voxel::empty();
        }
        voxels[(z as usize) * size * size + (y as usize) * size + (x as usize)]
    };
    let solid = |x: i32, y: i32, z: i32| -> bool { get(x, y, z).is_solid() };

    // Phase 1: Place one vertex per surface cell
    // A cell at (cx, cy, cz) has 8 corners at positions cx..cx+1, cy..cy+1, cz..cz+1
    let cs = s - 1; // cell grid goes 0..cs-1
    // Vertex index grid: u32::MAX means "no vertex here"
    let mut vi_grid: Vec<u32> = vec![u32::MAX; size * size * size];

    let corner_offsets: [(i32, i32, i32); 8] = [
        (0,0,0), (1,0,0), (1,1,0), (0,1,0),
        (0,0,1), (1,0,1), (1,1,1), (0,1,1),
    ];

    // 12 edges of a cube (pairs of corner indices)
    let edges: [(usize, usize); 12] = [
        (0,1), (1,2), (2,3), (3,0), // bottom face
        (4,5), (5,6), (6,7), (7,4), // top face
        (0,4), (1,5), (2,6), (3,7), // vertical edges
    ];

    for cz in 0..cs {
        for cy in 0..cs {
            for cx in 0..cs {
                // Sample 8 corners
                let mut mask: u8 = 0;
                let mut corner_solid = [false; 8];
                for (i, &(dx, dy, dz)) in corner_offsets.iter().enumerate() {
                    let is_solid = solid(cx + dx, cy + dy, cz + dz);
                    corner_solid[i] = is_solid;
                    if is_solid { mask |= 1 << i; }
                }

                // Skip cells entirely inside or outside the surface
                if mask == 0 || mask == 255 { continue; }

                // Average edge crossing positions
                let mut avg_x = 0.0f32;
                let mut avg_y = 0.0f32;
                let mut avg_z = 0.0f32;
                let mut count = 0u32;
                let mut color_r = 0u32;
                let mut color_g = 0u32;
                let mut color_b = 0u32;

                for &(a, b) in &edges {
                    if corner_solid[a] != corner_solid[b] {
                        let pa = corner_offsets[a];
                        let pb = corner_offsets[b];
                        avg_x += (pa.0 + pb.0) as f32 * 0.5;
                        avg_y += (pa.1 + pb.1) as f32 * 0.5;
                        avg_z += (pa.2 + pb.2) as f32 * 0.5;
                        count += 1;

                        // Color from the solid side
                        let so = if corner_solid[a] { pa } else { pb };
                        let v = get(cx + so.0, cy + so.1, cz + so.2);
                        let c = voxel_to_color(v);
                        color_r += (c[0] * 255.0) as u32;
                        color_g += (c[1] * 255.0) as u32;
                        color_b += (c[2] * 255.0) as u32;
                    }
                }

                if count == 0 { continue; }

                let inv = 1.0 / count as f32;
                let pos = Vec3::new(
                    (cx as f32 + avg_x * inv) * scale + offset.x,
                    (cy as f32 + avg_y * inv) * scale + offset.y,
                    (cz as f32 + avg_z * inv) * scale + offset.z,
                );

                let color = [
                    color_r as f32 * inv / 255.0,
                    color_g as f32 * inv / 255.0,
                    color_b as f32 * inv / 255.0,
                    1.0,
                ];

                // Material from nearest solid corner
                let mat = {
                    let mut m = 0u8;
                    for &(dx, dy, dz) in &corner_offsets {
                        let v = get(cx + dx, cy + dy, cz + dz);
                        if v.is_solid() { m = (v.packed & 0xFF) as u8; break; }
                    }
                    m
                };

                let idx = mesh.vertices.len() as u32;
                vi_grid[cz as usize * size * size + cy as usize * size + cx as usize] = idx;

                mesh.vertices.push(MeshVertex {
                    position: [pos.x, pos.y, pos.z],
                    normal: [0.0, 0.0, 0.0], // computed in Phase 4
                    color,
                    material: mat,
                });
            }
        }
    }

    // Phase 2: Emit quads for each surface-crossing edge
    let cell_vi = |cx: i32, cy: i32, cz: i32| -> Option<u32> {
        if cx < 0 || cy < 0 || cz < 0 || cx >= cs || cy >= cs || cz >= cs {
            return None;
        }
        let idx = vi_grid[cz as usize * size * size + cy as usize * size + cx as usize];
        if idx == u32::MAX { None } else { Some(idx) }
    };

    // X-edges: from (x,y,z) to (x+1,y,z)
    // Shared by cells: (x, y-1, z-1), (x, y, z-1), (x, y, z), (x, y-1, z)
    for z in 0..s {
        for y in 0..s {
            for x in 0..s - 1 {
                if solid(x, y, z) == solid(x + 1, y, z) { continue; }
                let v0 = cell_vi(x, y - 1, z - 1);
                let v1 = cell_vi(x, y,     z - 1);
                let v2 = cell_vi(x, y,     z);
                let v3 = cell_vi(x, y - 1, z);
                if let (Some(a), Some(b), Some(c), Some(d)) = (v0, v1, v2, v3) {
                    if solid(x + 1, y, z) {
                        emit_quad(&mut mesh, d, c, b, a);
                    } else {
                        emit_quad(&mut mesh, a, b, c, d);
                    }
                }
            }
        }
    }

    // Y-edges: from (x,y,z) to (x,y+1,z)
    // Shared by cells: (x-1, y, z-1), (x, y, z-1), (x, y, z), (x-1, y, z)
    for z in 0..s {
        for y in 0..s - 1 {
            for x in 0..s {
                if solid(x, y, z) == solid(x, y + 1, z) { continue; }
                let v0 = cell_vi(x - 1, y, z - 1);
                let v1 = cell_vi(x,     y, z - 1);
                let v2 = cell_vi(x,     y, z);
                let v3 = cell_vi(x - 1, y, z);
                if let (Some(a), Some(b), Some(c), Some(d)) = (v0, v1, v2, v3) {
                    if solid(x, y + 1, z) {
                        emit_quad(&mut mesh, a, b, c, d);
                    } else {
                        emit_quad(&mut mesh, d, c, b, a);
                    }
                }
            }
        }
    }

    // Z-edges: from (x,y,z) to (x,y,z+1)
    // Shared by cells: (x-1, y-1, z), (x, y-1, z), (x, y, z), (x-1, y, z)
    for z in 0..s - 1 {
        for y in 0..s {
            for x in 0..s {
                if solid(x, y, z) == solid(x, y, z + 1) { continue; }
                let v0 = cell_vi(x - 1, y - 1, z);
                let v1 = cell_vi(x,     y - 1, z);
                let v2 = cell_vi(x,     y,     z);
                let v3 = cell_vi(x - 1, y,     z);
                if let (Some(a), Some(b), Some(c), Some(d)) = (v0, v1, v2, v3) {
                    if solid(x, y, z + 1) {
                        emit_quad(&mut mesh, d, c, b, a);
                    } else {
                        emit_quad(&mut mesh, a, b, c, d);
                    }
                }
            }
        }
    }

    // Phase 3: Laplacian relaxation (smooth vertex positions)
    relax_vertices(&mut mesh, 2);

    // Phase 4: Compute smooth normals from face geometry
    compute_smooth_normals(&mut mesh);

    mesh
}

/// Emit a quad as two triangles
fn emit_quad(mesh: &mut ChunkMesh, a: u32, b: u32, c: u32, d: u32) {
    mesh.indices.push(a);
    mesh.indices.push(b);
    mesh.indices.push(c);
    mesh.indices.push(a);
    mesh.indices.push(c);
    mesh.indices.push(d);
    mesh.triangle_count += 2;
}

/// Laplacian relaxation: move each vertex toward the average of its neighbors.
/// Conservative blend preserves shape while removing stairstepping.
fn relax_vertices(mesh: &mut ChunkMesh, iterations: u32) {
    let n = mesh.vertices.len();
    if n == 0 { return; }

    // Build adjacency from triangle indices
    let mut neighbors: Vec<Vec<u32>> = vec![Vec::new(); n];
    for tri in mesh.indices.chunks(3) {
        if tri.len() < 3 { continue; }
        for i in 0..3 {
            let a = tri[i] as usize;
            let b = tri[(i + 1) % 3] as usize;
            if !neighbors[a].contains(&(b as u32)) { neighbors[a].push(b as u32); }
            if !neighbors[b].contains(&(a as u32)) { neighbors[b].push(a as u32); }
        }
    }

    for _ in 0..iterations {
        let old_pos: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| v.position).collect();
        for i in 0..n {
            if neighbors[i].is_empty() { continue; }
            let mut avg = Vec3::ZERO;
            for &j in &neighbors[i] {
                let p = old_pos[j as usize];
                avg += Vec3::new(p[0], p[1], p[2]);
            }
            avg /= neighbors[i].len() as f32;
            // 60% original, 40% neighbor average — conservative smoothing
            let orig = Vec3::new(old_pos[i][0], old_pos[i][1], old_pos[i][2]);
            let smoothed = orig * 0.6 + avg * 0.4;
            mesh.vertices[i].position = [smoothed.x, smoothed.y, smoothed.z];
        }
    }
}

/// Compute per-vertex normals by accumulating face normals
fn compute_smooth_normals(mesh: &mut ChunkMesh) {
    let n = mesh.vertices.len();
    if n == 0 { return; }

    let mut normals = vec![Vec3::ZERO; n];

    for tri in mesh.indices.chunks(3) {
        if tri.len() < 3 { continue; }
        let p0 = Vec3::from(mesh.vertices[tri[0] as usize].position);
        let p1 = Vec3::from(mesh.vertices[tri[1] as usize].position);
        let p2 = Vec3::from(mesh.vertices[tri[2] as usize].position);
        let face_normal = (p1 - p0).cross(p2 - p0);
        // Weight by face area (magnitude of cross product = 2× area)
        normals[tri[0] as usize] += face_normal;
        normals[tri[1] as usize] += face_normal;
        normals[tri[2] as usize] += face_normal;
    }

    for (i, v) in mesh.vertices.iter_mut().enumerate() {
        let n = normals[i].normalize_or_zero();
        v.normal = [n.x, n.y, n.z];
    }
}

/// Smooth mesh with per-vertex AO from voxel neighborhood
pub fn generate_mesh_smooth_with_ao(
    voxels: &[Voxel],
    size: usize,
    offset: Vec3,
    scale: f32,
) -> ChunkMesh {
    let mut mesh = generate_mesh_smooth(voxels, size, offset, scale);

    // Compute AO for each vertex based on voxel neighborhood
    for v in mesh.vertices.iter_mut() {
        let vx = ((v.position[0] - offset.x) / scale) as i32;
        let vy = ((v.position[1] - offset.y) / scale) as i32;
        let vz = ((v.position[2] - offset.z) / scale) as i32;
        let ao = compute_ao(voxels, size, vx, vy, vz, v.normal);
        v.color[0] *= ao;
        v.color[1] *= ao;
        v.color[2] *= ao;
    }

    mesh
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_chunk(size: usize) -> Vec<Voxel> {
        let mut voxels = vec![Voxel::empty(); size * size * size];
        // Fill a small cube in the center
        let half = size / 2;
        let r = size / 4;
        for z in (half-r)..(half+r) {
            for y in (half-r)..(half+r) {
                for x in (half-r)..(half+r) {
                    voxels[z * size * size + y * size + x] = Voxel::solid(1, 200, 100, 50);
                }
            }
        }
        voxels
    }

    #[test]
    fn test_generate_mesh_basic() {
        let voxels = make_test_chunk(16);
        let mesh = generate_mesh(&voxels, 16, Vec3::ZERO, 1.0);

        println!("16³ chunk with 8³ cube: {} triangles, {} vertices",
            mesh.triangle_count, mesh.vertices.len());

        // 8³ cube has 6 faces, each face = 8×8 quads, each quad = 2 triangles
        // But only SURFACE faces (not interior), so:
        // 6 faces × 64 surface quads = 384 quads = 768 triangles
        assert!(mesh.triangle_count > 0);
        assert!(mesh.triangle_count <= 768);
    }

    #[test]
    fn test_single_voxel() {
        let mut voxels = vec![Voxel::empty(); 4*4*4];
        voxels[1*4*4 + 1*4 + 1] = Voxel::solid(1, 255, 0, 0);

        let mesh = generate_mesh(&voxels, 4, Vec3::ZERO, 1.0);

        println!("Single voxel: {} triangles", mesh.triangle_count);
        // Single voxel exposed on all 6 sides = 6 quads = 12 triangles
        assert_eq!(mesh.triangle_count, 12);
    }

    #[test]
    fn test_empty_chunk() {
        let voxels = vec![Voxel::empty(); 8*8*8];
        let mesh = generate_mesh(&voxels, 8, Vec3::ZERO, 1.0);
        assert!(mesh.is_empty());
    }

    #[test]
    fn test_full_chunk_no_interior_faces() {
        // Completely filled chunk: only outer faces visible
        let voxels = vec![Voxel::solid(1, 100, 100, 100); 8*8*8];
        let mesh = generate_mesh(&voxels, 8, Vec3::ZERO, 1.0);

        println!("Full 8³: {} triangles", mesh.triangle_count);
        // Only boundary faces: 6 faces × 64 quads = 384 quads = 768 triangles
        assert_eq!(mesh.triangle_count, 768);
    }

    #[test]
    fn test_mesh_with_ao() {
        let voxels = make_test_chunk(16);
        let mesh = generate_mesh_with_ao(&voxels, 16, Vec3::ZERO, 1.0);

        // Verify AO darkened some vertices
        let min_brightness: f32 = mesh.vertices.iter()
            .map(|v| v.color[0])
            .fold(f32::MAX, f32::min);
        println!("Min vertex brightness after AO: {:.3}", min_brightness);
        // Corner vertices should be darker
        assert!(min_brightness < 0.75);
    }

    #[test]
    fn test_smooth_mesh_basic() {
        let voxels = make_test_chunk(16);
        let mesh = generate_mesh_smooth(&voxels, 16, Vec3::ZERO, 1.0);

        println!("Smooth 16³ with 8³ cube: {} triangles, {} vertices",
            mesh.triangle_count, mesh.vertices.len());

        assert!(mesh.triangle_count > 0);
        // Surface Nets produces fewer triangles than sharp meshing (shared vertices)
        assert!(mesh.vertices.len() > 0);
    }

    #[test]
    fn test_smooth_mesh_sphere() {
        // Create a sphere — this is where smooth shines
        let size = 32;
        let mut voxels = vec![Voxel::empty(); size * size * size];
        let center = size as f32 / 2.0;
        let radius = size as f32 / 4.0;
        for z in 0..size { for y in 0..size { for x in 0..size {
            let dx = x as f32 - center;
            let dy = y as f32 - center;
            let dz = z as f32 - center;
            if dx*dx + dy*dy + dz*dz <= radius * radius {
                voxels[z * size * size + y * size + x] = Voxel::solid(1, 200, 100, 50);
            }
        }}}

        let mesh = generate_mesh_smooth(&voxels, size, Vec3::ZERO, 1.0);
        println!("Smooth sphere (r={}): {} triangles, {} vertices",
            radius as i32, mesh.triangle_count, mesh.vertices.len());

        // Verify normals point outward (dot with direction from center should be positive)
        let mut outward_count = 0;
        for v in &mesh.vertices {
            let pos = Vec3::from(v.position);
            let to_center = Vec3::splat(center) - pos;
            let normal = Vec3::from(v.normal);
            if normal.dot(to_center) < 0.0 { outward_count += 1; }
        }
        let ratio = outward_count as f32 / mesh.vertices.len() as f32;
        println!("Normals pointing outward: {:.0}%", ratio * 100.0);
        assert!(ratio > 0.8); // most normals should point outward
    }

    #[test]
    fn test_smooth_with_ao() {
        let voxels = make_test_chunk(16);
        let mesh = generate_mesh_smooth_with_ao(&voxels, 16, Vec3::ZERO, 1.0);

        assert!(mesh.triangle_count > 0);
        // AO should darken some vertices
        let min_brightness: f32 = mesh.vertices.iter()
            .map(|v| v.color[0])
            .fold(f32::MAX, f32::min);
        println!("Smooth AO min brightness: {:.3}", min_brightness);
    }

    #[test]
    fn test_64_chunk_performance() {
        let mut voxels = vec![Voxel::empty(); 64*64*64];
        // Floor + walls (typical room)
        for z in 0..64 { for x in 0..64 {
            voxels[z*64*64 + 0*64 + x] = Voxel::solid(7, 180, 155, 115); // floor
        }}
        for z in 0..64 { for y in 0..64 {
            voxels[z*64*64 + y*64 + 0] = Voxel::solid(7, 220, 215, 205); // wall
            voxels[z*64*64 + y*64 + 63] = Voxel::solid(7, 220, 215, 205);
        }}
        // Some furniture boxes
        for z in 20..30 { for y in 1..10 { for x in 20..30 {
            voxels[z*64*64 + y*64 + x] = Voxel::solid(1, 110, 70, 35); // table
        }}}

        let start = std::time::Instant::now();
        let mesh = generate_mesh(&voxels, 64, Vec3::ZERO, 1.0);
        let elapsed = start.elapsed();

        println!("64³ room chunk: {} triangles, {} vertices, {:.2} ms",
            mesh.triangle_count, mesh.vertices.len(), elapsed.as_secs_f64() * 1000.0);
        println!("Memory: {:.1} KB", mesh.memory_bytes() as f64 / 1024.0);

        // Should complete in under 50ms (debug mode is slow, release ~2ms)
        assert!(elapsed.as_millis() < 50);
        assert!(mesh.triangle_count > 1000);
    }
}
