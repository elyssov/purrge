// ═══════════════════════════════════════════════════════════════
// PURRGE — Furniture System
//
// Each piece of furniture is a SEPARATE object with its own
// voxel grid, mass, bounding box, and support check.
// Objects are NOT baked into the room grid.
// They sit ON surfaces. Remove the surface → they FALL.
// ═══════════════════════════════════════════════════════════════

use glam::Vec3;
use crate::core::svo::Voxel;
use crate::core::procgen::Rng;
use crate::core::meshing::{generate_mesh, ChunkMesh};
use crate::apartment::{VoxelGrid, GRID, FLOOR_Y};

/// A single piece of furniture — separate from room grid
pub struct FurnitureObj {
    pub name: String,
    /// World position (bottom-left corner of local grid)
    pub pos: Vec3,
    /// Local voxel grid (small, just this object)
    pub voxels: Vec<Voxel>,
    pub grid_size: usize,
    /// Mass in kg (from material density × volume)
    pub mass: f32,
    /// Dollar value for repair bill
    pub value: f32,
    /// Is it currently falling?
    pub falling: bool,
    /// Velocity when falling
    pub vel: Vec3,
    /// Has it shattered on floor?
    pub shattered: bool,
    /// Material brittleness (for shatter on impact)
    pub brittleness: f32,
    /// Mesh needs rebuild
    pub mesh_dirty: bool,
}

impl FurnitureObj {
    pub fn new(name: &str, pos: Vec3, grid_size: usize, mass: f32, value: f32) -> Self {
        Self {
            name: name.to_string(),
            pos,
            voxels: vec![Voxel::empty(); grid_size * grid_size * grid_size],
            grid_size,
            mass, value,
            falling: false, vel: Vec3::ZERO,
            shattered: false, brittleness: 0.5,
            mesh_dirty: true,
        }
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, v: Voxel) {
        if x < self.grid_size && y < self.grid_size && z < self.grid_size {
            self.voxels[z * self.grid_size * self.grid_size + y * self.grid_size + x] = v;
            self.mesh_dirty = true;
        }
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> Voxel {
        if x < self.grid_size && y < self.grid_size && z < self.grid_size {
            self.voxels[z * self.grid_size * self.grid_size + y * self.grid_size + x]
        } else {
            Voxel::empty()
        }
    }

    pub fn fill_box(&mut self, x0: usize, y0: usize, z0: usize, x1: usize, y1: usize, z1: usize, v: Voxel) {
        for z in z0..=z1.min(self.grid_size - 1) {
            for y in y0..=y1.min(self.grid_size - 1) {
                for x in x0..=x1.min(self.grid_size - 1) {
                    self.set(x, y, z, v);
                }
            }
        }
    }

    /// World-space AABB
    pub fn world_min(&self) -> Vec3 { self.pos }
    pub fn world_max(&self) -> Vec3 { self.pos + Vec3::splat(self.grid_size as f32) }
    pub fn world_center(&self) -> Vec3 { self.pos + Vec3::splat(self.grid_size as f32 * 0.5) }

    /// Check if a world point is inside this object
    pub fn contains_world(&self, wx: f32, wy: f32, wz: f32) -> bool {
        let lx = (wx - self.pos.x) as i32;
        let ly = (wy - self.pos.y) as i32;
        let lz = (wz - self.pos.z) as i32;
        if lx < 0 || ly < 0 || lz < 0 { return false; }
        let (lx, ly, lz) = (lx as usize, ly as usize, lz as usize);
        self.get(lx, ly, lz).is_solid()
    }

    /// Check if object has support below it (from room grid or other objects)
    pub fn has_support(&self, room: &VoxelGrid, others: &[FurnitureObj]) -> bool {
        let gs = self.grid_size;
        // Check bottom layer of local grid — any solid voxel must have something below it
        for z in 0..gs {
            for x in 0..gs {
                // Find lowest solid voxel in this column
                for y in 0..gs {
                    if self.get(x, y, z).is_solid() {
                        let wy = self.pos.y + y as f32 - 1.0; // one below
                        let wx = self.pos.x + x as f32;
                        let wz = self.pos.z + z as f32;

                        // Check room grid
                        let rwx = wx as usize;
                        let rwy = wy as usize;
                        let rwz = wz as usize;
                        if rwx < GRID && rwy < GRID && rwz < GRID && room.get(rwx, rwy, rwz).is_solid() {
                            return true;
                        }

                        // Check other furniture
                        for other in others {
                            if std::ptr::eq(other, self) { continue; }
                            if other.shattered || other.falling { continue; }
                            if other.contains_world(wx, wy, wz) {
                                return true;
                            }
                        }

                        break; // only check lowest solid per column
                    }
                }
            }
        }
        false
    }

    /// Remove voxels in a sphere around world point. Returns removed count.
    pub fn scratch_world(&mut self, center: Vec3, radius: f32) -> usize {
        let r2 = radius * radius;
        let ri = radius.ceil() as i32;
        let mut removed = 0;
        for dz in -ri..=ri { for dy in -ri..=ri { for dx in -ri..=ri {
            if (dx*dx + dy*dy + dz*dz) as f32 > r2 { continue; }
            let wx = center.x as i32 + dx;
            let wy = center.y as i32 + dy;
            let wz = center.z as i32 + dz;
            let lx = wx - self.pos.x as i32;
            let ly = wy - self.pos.y as i32;
            let lz = wz - self.pos.z as i32;
            if lx >= 0 && ly >= 0 && lz >= 0 {
                let (lx, ly, lz) = (lx as usize, ly as usize, lz as usize);
                if lx < self.grid_size && ly < self.grid_size && lz < self.grid_size {
                    if self.get(lx, ly, lz).is_solid() {
                        self.set(lx, ly, lz, Voxel::empty());
                        removed += 1;
                    }
                }
            }
        }}}
        if removed > 0 { self.mesh_dirty = true; }
        removed
    }

    /// Build mesh for rendering (offset by world position)
    pub fn build_mesh(&self) -> ChunkMesh {
        generate_mesh(&self.voxels, self.grid_size, self.pos, 1.0)
    }

    /// Update physics (falling)
    pub fn update(&mut self, dt: f32, floor_y: f32) -> bool {
        if !self.falling || self.shattered { return false; }

        self.vel.y -= 120.0 * dt; // gravity
        self.pos += self.vel * dt;

        // Floor collision
        if self.pos.y <= floor_y {
            self.pos.y = floor_y;
            let impact = (-self.vel.y).max(0.0);
            if impact > 30.0 && self.brittleness > 0.3 {
                self.shattered = true;
                return true; // signal: shattered!
            }
            self.vel.y = self.vel.y.abs() * 0.2; // bounce
            if self.vel.y.abs() < 5.0 {
                self.vel = Vec3::ZERO;
                self.falling = false;
            }
        }
        self.mesh_dirty = true;
        false
    }

    /// Count solid voxels
    pub fn voxel_count(&self) -> usize {
        self.voxels.iter().filter(|v| v.is_solid()).count()
    }
}

// ═══════════════════════════════════════════════════════════════
// FURNITURE CATALOGUE — builders for each type
// ═══════════════════════════════════════════════════════════════

fn vary(rng: &mut Rng, r: u8, g: u8, b: u8, j: u8) -> Voxel {
    let ji = j as i32;
    let vr = (r as i32 + rng.range(-ji, ji)).clamp(0, 255) as u8;
    let vg = (g as i32 + rng.range(-ji, ji)).clamp(0, 255) as u8;
    let vb = (b as i32 + rng.range(-ji, ji)).clamp(0, 255) as u8;
    Voxel::solid(1, vr, vg, vb)
}

pub fn make_sofa(rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    let mut f = FurnitureObj::new("Sofa", pos, 32, 45.0, 800.0);
    let palettes: [([u8;3],[u8;3]); 4] = [
        ([75,115,165],[90,135,185]), ([165,75,80],[185,95,100]),
        ([85,140,85],[105,165,105]), ([155,130,85],[175,155,110]),
    ];
    let (base, cush) = palettes[rng.range(0, 3) as usize];
    let sb = Voxel::solid(5, base[0], base[1], base[2]);
    let sc = Voxel::solid(5, cush[0], cush[1], cush[2]);
    f.fill_box(0, 0, 0, 26, 20, 31, sb);    // frame
    f.fill_box(2, 20, 2, 24, 24, 29, sc);    // cushions
    f.fill_box(0, 20, 0, 5, 30, 31, sb);     // back
    f.fill_box(0, 20, 0, 26, 26, 3, sb);     // arm
    f.fill_box(0, 20, 28, 26, 26, 31, sb);   // arm
    f.brittleness = 0.1;
    f
}

pub fn make_table(rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    let mut f = FurnitureObj::new("Table", pos, 32, 20.0, 400.0);
    let tw = vary(rng, 115, 72, 38, 12);
    // 4 legs
    f.fill_box(1, 0, 1, 3, 28, 3, tw);
    f.fill_box(27, 0, 1, 29, 28, 3, tw);
    f.fill_box(1, 0, 27, 3, 28, 29, tw);
    f.fill_box(27, 0, 27, 29, 28, 29, tw);
    // Top
    let tt = vary(rng, 138, 90, 48, 10);
    f.fill_box(0, 28, 0, 31, 30, 31, tt);
    f.brittleness = 0.2;
    f
}

pub fn make_tv_stand(rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    let mut f = FurnitureObj::new("TV Stand", pos, 16, 15.0, 200.0);
    let tw = vary(rng, 115, 72, 38, 10);
    f.fill_box(0, 0, 0, 15, 14, 8, tw);
    f.brittleness = 0.3;
    f
}

pub fn make_tv(_rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    let mut f = FurnitureObj::new("TV", pos, 32, 8.0, 1200.0);
    let screen = Voxel::solid(4, 28, 28, 32);
    let bezel = Voxel::solid(4, 55, 55, 60);
    // Thin flat screen
    f.fill_box(0, 0, 2, 2, 24, 29, screen);
    f.fill_box(0, 0, 0, 2, 26, 31, bezel);
    f.brittleness = 0.9; // glass!
    f
}

pub fn make_vase(rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    let mut f = FurnitureObj::new("Vase", pos, 8, 1.5, 200.0);
    let colors: [[u8;3]; 5] = [[195,65,55],[55,130,195],[195,180,55],[165,55,165],[55,175,120]];
    let c = colors[rng.range(0, 4) as usize];
    let v = Voxel::solid(3, c[0], c[1], c[2]);
    // Simple sphere-ish
    for z in 0..8 { for y in 0..8 { for x in 0..8 {
        let dx = x as f32 - 3.5; let dy = y as f32 - 3.5; let dz = z as f32 - 3.5;
        if dx*dx + dy*dy + dz*dz <= 12.0 { f.set(x, y, z, v); }
    }}}
    f.brittleness = 1.0; // ceramic
    f
}

pub fn make_chair(rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    let mut f = FurnitureObj::new("Chair", pos, 16, 5.0, 150.0);
    let tw = vary(rng, 98, 62, 28, 12);
    // Legs
    f.fill_box(1, 0, 1, 3, 14, 3, tw);
    f.fill_box(11, 0, 1, 13, 14, 3, tw);
    f.fill_box(1, 0, 11, 3, 14, 13, tw);
    f.fill_box(11, 0, 11, 13, 14, 13, tw);
    // Seat
    f.fill_box(0, 14, 0, 14, 16, 14, tw);
    // Back
    f.fill_box(0, 14, 12, 14, 28, 14, tw);
    f.brittleness = 0.2;
    f
}

pub fn make_bookshelf(rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    let mut f = FurnitureObj::new("Bookshelf", pos, 24, 30.0, 350.0);
    let sw = vary(rng, 98, 62, 28, 10);
    f.fill_box(0, 0, 0, 23, 20, 12, sw);
    // Shelves
    f.fill_box(1, 6, 1, 22, 6, 11, sw);
    f.fill_box(1, 13, 1, 22, 13, 11, sw);
    // Books
    let bc: [[u8;3]; 4] = [[55,75,135],[155,48,48],[48,125,55],[130,65,130]];
    let mut bx = 2;
    for i in 0..3 {
        let c = bc[rng.range(0, 3) as usize];
        let bw = rng.range(3, 6) as usize;
        if bx + bw < 22 {
            f.fill_box(bx, 7, 2, bx+bw, 12, 10, Voxel::solid(9, c[0], c[1], c[2]));
            bx += bw + 1;
        }
    }
    f.brittleness = 0.3;
    f
}

pub fn make_lamp(_rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    let mut f = FurnitureObj::new("Lamp", pos, 10, 2.0, 100.0);
    let mt = Voxel::solid(4, 148, 148, 152);
    let shade = Voxel::solid(6, 225, 215, 180);
    f.fill_box(4, 0, 4, 5, 8, 5, mt); // pole
    // Shade (sphere-ish)
    for z in 0..10 { for y in 6..10 { for x in 0..10 {
        let dx = x as f32 - 4.5; let dy = y as f32 - 8.0; let dz = z as f32 - 4.5;
        if dx*dx + dy*dy*4.0 + dz*dz <= 16.0 { f.set(x, y, z, shade); }
    }}}
    f.brittleness = 0.7;
    f
}

pub fn make_fridge(_rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    let mut f = FurnitureObj::new("Fridge", pos, 16, 80.0, 1500.0);
    let fr = Voxel::solid(4, 235, 235, 240);
    f.fill_box(0, 0, 0, 14, 15, 14, fr);
    // Handle
    f.fill_box(0, 5, 6, 0, 12, 7, Voxel::solid(4, 160, 160, 165));
    f.brittleness = 0.05; // steel
    f
}
