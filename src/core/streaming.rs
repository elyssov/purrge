// ═══════════════════════════════════════════════════════════════
// PROMETHEUS ENGINE — Streaming Voxel World
//
// The world is infinite and procedural. We only materialize the
// voxels the camera can see. Everything else is formulas.
//
// Architecture:
//   World = procedural formulas (rooms, bodies, terrain)
//   Chunks = 64³ voxel blocks, generated on demand
//   View = set of chunks around camera (frustum)
//   Diff = destroyed/modified voxels (sparse, persistent)
//
// Target: Intel i5-11 + Iris Xe + 16GB RAM
// Budget: 128 MB VRAM, 64 chunks × 2MB = 128 MB
// ═══════════════════════════════════════════════════════════════

use std::collections::HashMap;
use super::svo::Voxel;

/// Size of one chunk in voxels
pub const CHUNK_SIZE: usize = 64;
/// Chunk memory: 64³ × 8 bytes = 2 MB
pub const CHUNK_BYTES: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * 8;

/// Chunk coordinates (world position / CHUNK_SIZE)
#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
pub struct ChunkCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ChunkCoord {
    pub fn new(x: i32, y: i32, z: i32) -> Self { Self { x, y, z } }

    /// World-space origin of this chunk (minimum corner)
    pub fn world_origin(&self) -> (i32, i32, i32) {
        (self.x * CHUNK_SIZE as i32, self.y * CHUNK_SIZE as i32, self.z * CHUNK_SIZE as i32)
    }
}

/// A single chunk — 64³ voxels, generated on demand
pub struct Chunk {
    pub coord: ChunkCoord,
    pub data: Vec<Voxel>,
    /// True if this chunk has been generated
    pub generated: bool,
    /// Frame number when last accessed (for LRU eviction)
    pub last_used: u64,
    /// Number of non-empty voxels (for culling)
    pub solid_count: usize,
}

impl Chunk {
    pub fn new(coord: ChunkCoord) -> Self {
        Self {
            coord,
            data: vec![Voxel::empty(); CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE],
            generated: false,
            last_used: 0,
            solid_count: 0,
        }
    }

    pub fn idx(x: usize, y: usize, z: usize) -> usize {
        z * CHUNK_SIZE * CHUNK_SIZE + y * CHUNK_SIZE + x
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> Voxel {
        self.data[Self::idx(x, y, z)]
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, v: Voxel) {
        let idx = Self::idx(x, y, z);
        let was_solid = self.data[idx].is_solid();
        let is_solid = v.is_solid();
        self.data[idx] = v;
        match (was_solid, is_solid) {
            (false, true) => self.solid_count += 1,
            (true, false) => self.solid_count -= 1,
            _ => {}
        }
    }

    pub fn is_empty(&self) -> bool { self.solid_count == 0 }
}

/// Modification to the procedural world (destruction, placed objects)
#[derive(Clone, Debug)]
pub struct WorldDiff {
    /// Removed voxels: world coordinate → true
    pub removed: HashMap<(i32, i32, i32), bool>,
    /// Added/changed voxels: world coordinate → voxel
    pub added: HashMap<(i32, i32, i32), Voxel>,
}

impl WorldDiff {
    pub fn new() -> Self {
        Self { removed: HashMap::new(), added: HashMap::new() }
    }

    /// Record a destroyed voxel
    pub fn destroy(&mut self, x: i32, y: i32, z: i32) {
        self.removed.insert((x, y, z), true);
        self.added.remove(&(x, y, z));
    }

    /// Record a placed voxel
    pub fn place(&mut self, x: i32, y: i32, z: i32, v: Voxel) {
        self.added.insert((x, y, z), v);
        self.removed.remove(&(x, y, z));
    }

    /// Check if a voxel was destroyed
    pub fn is_destroyed(&self, x: i32, y: i32, z: i32) -> bool {
        self.removed.contains_key(&(x, y, z))
    }

    /// Check if a voxel was placed
    pub fn get_placed(&self, x: i32, y: i32, z: i32) -> Option<&Voxel> {
        self.added.get(&(x, y, z))
    }

    /// Memory usage
    pub fn memory_bytes(&self) -> usize {
        self.removed.len() * 20 + self.added.len() * 28 // approximate
    }
}

/// Generator function type: given world coords, returns voxel
pub type GeneratorFn = Box<dyn Fn(i32, i32, i32) -> Voxel + Send + Sync>;

/// Streaming voxel world — infinite, procedural, chunked
pub struct StreamingWorld {
    /// Active chunks (loaded in memory)
    chunks: HashMap<ChunkCoord, Chunk>,
    /// Procedural generator
    generator: GeneratorFn,
    /// Modifications (destruction, placement)
    pub diff: WorldDiff,
    /// Current frame number
    frame: u64,
    /// Maximum chunks in memory
    pub max_chunks: usize,
    /// View distance in chunks
    pub view_distance: i32,
}

impl StreamingWorld {
    /// Create a new streaming world with a generator function
    pub fn new(generator: GeneratorFn) -> Self {
        Self {
            chunks: HashMap::new(),
            generator,
            diff: WorldDiff::new(),
            frame: 0,
            max_chunks: 64,     // 64 × 2MB = 128 MB
            view_distance: 2,   // 2 chunks = 128 voxels in each direction
        }
    }

    /// Update: generate/evict chunks based on camera position
    pub fn update(&mut self, camera_x: f32, camera_y: f32, camera_z: f32) {
        self.frame += 1;

        // Camera chunk
        let ccx = (camera_x / CHUNK_SIZE as f32).floor() as i32;
        let ccy = (camera_y / CHUNK_SIZE as f32).floor() as i32;
        let ccz = (camera_z / CHUNK_SIZE as f32).floor() as i32;

        let vd = self.view_distance;

        // Generate needed chunks
        for cz in (ccz - vd)..=(ccz + vd) {
            for cy in (ccy - vd)..=(ccy + vd) {
                for cx in (ccx - vd)..=(ccx + vd) {
                    let coord = ChunkCoord::new(cx, cy, cz);
                    if !self.chunks.contains_key(&coord) {
                        let mut chunk = Chunk::new(coord);
                        self.generate_chunk(&mut chunk);
                        self.chunks.insert(coord, chunk);
                    }
                    if let Some(chunk) = self.chunks.get_mut(&coord) {
                        chunk.last_used = self.frame;
                    }
                }
            }
        }

        // Evict distant chunks (LRU)
        if self.chunks.len() > self.max_chunks {
            let mut to_remove: Vec<ChunkCoord> = Vec::new();
            for (coord, chunk) in &self.chunks {
                if self.frame - chunk.last_used > 10 { // not used for 10 frames
                    to_remove.push(*coord);
                }
            }
            // Sort by last_used (oldest first)
            to_remove.sort_by_key(|c| self.chunks[c].last_used);
            for coord in to_remove {
                if self.chunks.len() <= self.max_chunks { break; }
                self.chunks.remove(&coord);
            }
        }
    }

    /// Get voxel at world coordinates (generates chunk if needed)
    pub fn get(&self, x: i32, y: i32, z: i32) -> Voxel {
        // Check diff first
        if self.diff.is_destroyed(x, y, z) { return Voxel::empty(); }
        if let Some(v) = self.diff.get_placed(x, y, z) { return *v; }

        // Check loaded chunks
        let coord = ChunkCoord::new(
            x.div_euclid(CHUNK_SIZE as i32),
            y.div_euclid(CHUNK_SIZE as i32),
            z.div_euclid(CHUNK_SIZE as i32),
        );
        if let Some(chunk) = self.chunks.get(&coord) {
            let lx = x.rem_euclid(CHUNK_SIZE as i32) as usize;
            let ly = y.rem_euclid(CHUNK_SIZE as i32) as usize;
            let lz = z.rem_euclid(CHUNK_SIZE as i32) as usize;
            return chunk.get(lx, ly, lz);
        }

        // Not loaded — generate on the fly (slower, but correct)
        (self.generator)(x, y, z)
    }

    /// Destroy a voxel at world coordinates
    pub fn destroy_at(&mut self, x: i32, y: i32, z: i32) {
        self.diff.destroy(x, y, z);
        // Also update loaded chunk if present
        let coord = ChunkCoord::new(
            x.div_euclid(CHUNK_SIZE as i32),
            y.div_euclid(CHUNK_SIZE as i32),
            z.div_euclid(CHUNK_SIZE as i32),
        );
        if let Some(chunk) = self.chunks.get_mut(&coord) {
            let lx = x.rem_euclid(CHUNK_SIZE as i32) as usize;
            let ly = y.rem_euclid(CHUNK_SIZE as i32) as usize;
            let lz = z.rem_euclid(CHUNK_SIZE as i32) as usize;
            chunk.set(lx, ly, lz, Voxel::empty());
        }
    }

    /// Destroy sphere of voxels at world coordinates
    pub fn destroy_sphere(&mut self, cx: f32, cy: f32, cz: f32, radius: f32) {
        let r = radius.ceil() as i32;
        let r2 = radius * radius;
        for dz in -r..=r { for dy in -r..=r { for dx in -r..=r {
            if (dx*dx + dy*dy + dz*dz) as f32 <= r2 {
                self.destroy_at(cx as i32 + dx, cy as i32 + dy, cz as i32 + dz);
            }
        }}}
    }

    /// Export visible region as flat array for GPU (centered on camera)
    pub fn export_view(&self, center_x: i32, center_y: i32, center_z: i32, size: usize) -> Vec<Voxel> {
        let half = size as i32 / 2;
        let mut flat = vec![Voxel::empty(); size * size * size];

        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let wx = center_x - half + x as i32;
                    let wy = center_y - half + y as i32;
                    let wz = center_z - half + z as i32;
                    let v = self.get(wx, wy, wz);
                    if v.is_solid() {
                        flat[z * size * size + y * size + x] = v;
                    }
                }
            }
        }
        flat
    }

    /// Number of loaded chunks
    pub fn loaded_chunks(&self) -> usize { self.chunks.len() }

    /// Total memory usage
    pub fn memory_bytes(&self) -> usize {
        self.chunks.len() * CHUNK_BYTES + self.diff.memory_bytes()
    }

    // ─── Internal ────────────────────────────────────────────

    fn generate_chunk(&self, chunk: &mut Chunk) {
        let (ox, oy, oz) = chunk.coord.world_origin();
        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let wx = ox + x as i32;
                    let wy = oy + y as i32;
                    let wz = oz + z as i32;

                    // Check diff
                    if self.diff.is_destroyed(wx, wy, wz) { continue; }
                    if let Some(v) = self.diff.get_placed(wx, wy, wz) {
                        chunk.set(x, y, z, *v);
                        continue;
                    }

                    // Generate procedurally
                    let v = (self.generator)(wx, wy, wz);
                    if v.is_solid() {
                        chunk.set(x, y, z, v);
                    }
                }
            }
        }
        chunk.generated = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_generator(x: i32, y: i32, z: i32) -> Voxel {
        // Simple room: floor at y=0, walls at edges of 256 cube
        if y == 0 && x >= 0 && x < 256 && z >= 0 && z < 256 {
            return Voxel::solid(7, 180, 155, 115); // floor
        }
        Voxel::empty()
    }

    #[test]
    fn test_streaming_basic() {
        let mut world = StreamingWorld::new(Box::new(test_generator));
        world.update(128.0, 10.0, 128.0);

        // Floor should exist
        let v = world.get(128, 0, 128);
        assert!(v.is_solid());

        // Air should be empty
        let v = world.get(128, 50, 128);
        assert!(v.is_empty());

        println!("Chunks loaded: {}", world.loaded_chunks());
        println!("Memory: {:.1} MB", world.memory_bytes() as f64 / 1_048_576.0);
    }

    #[test]
    fn test_streaming_destroy() {
        let mut world = StreamingWorld::new(Box::new(test_generator));
        world.update(128.0, 10.0, 128.0);

        // Floor exists
        assert!(world.get(128, 0, 128).is_solid());

        // Destroy it
        world.destroy_at(128, 0, 128);
        assert!(world.get(128, 0, 128).is_empty());

        // Diff recorded
        assert!(world.diff.is_destroyed(128, 0, 128));
    }

    #[test]
    fn test_streaming_export_view() {
        let mut world = StreamingWorld::new(Box::new(test_generator));
        world.update(128.0, 10.0, 128.0);

        let flat = world.export_view(128, 10, 128, 64);
        assert_eq!(flat.len(), 64 * 64 * 64);

        // Count solid voxels in view
        let solid = flat.iter().filter(|v| v.is_solid()).count();
        println!("Solid voxels in 64³ view: {}", solid);
        assert!(solid > 0);
    }

    #[test]
    fn test_streaming_memory() {
        let mut world = StreamingWorld::new(Box::new(test_generator));
        world.view_distance = 2;
        world.update(128.0, 10.0, 128.0);

        let mem = world.memory_bytes();
        println!("Memory with view_distance=2: {:.1} MB ({} chunks)",
            mem as f64 / 1_048_576.0, world.loaded_chunks());

        // Should be under 512 MB for view_distance=2 (125 chunks × 2MB = 250MB)
        assert!(mem < 512 * 1024 * 1024);
    }
}
