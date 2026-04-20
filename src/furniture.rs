// ═══════════════════════════════════════════════════════════════
// PURRGE — Furniture System
//
// Each piece of furniture is a SEPARATE object with its own
// voxel grid, mass, bounding box, and support check.
// Objects are NOT baked into the room grid.
// They sit ON surfaces. Remove the surface → they FALL.
// ═══════════════════════════════════════════════════════════════

use glam::Vec3;
use std::collections::HashMap;
use crate::core::svo::Voxel;
use crate::core::procgen::Rng;
use crate::core::meshing::{generate_mesh, ChunkMesh};
use crate::apartment::{VoxelGrid, GRID, FLOOR_Y};

/// A single piece of furniture — sparse voxels, real scale
pub struct FurnitureObj {
    pub name: String,
    /// World position (bottom-left corner)
    pub pos: Vec3,
    /// Sparse local voxels
    pub voxels: HashMap<(u16, u16, u16), Voxel>,
    /// Bounding box size (for meshing export)
    pub size_x: usize,
    pub size_y: usize,
    pub size_z: usize,
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
    pub fn new(name: &str, pos: Vec3, sx: usize, sy: usize, sz: usize, mass: f32, value: f32) -> Self {
        Self {
            name: name.to_string(),
            pos,
            voxels: HashMap::new(),
            size_x: sx, size_y: sy, size_z: sz,
            mass, value,
            falling: false, vel: Vec3::ZERO,
            shattered: false, brittleness: 0.5,
            mesh_dirty: true,
        }
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, v: Voxel) {
        if v.is_solid() {
            self.voxels.insert((x as u16, y as u16, z as u16), v);
        } else {
            self.voxels.remove(&(x as u16, y as u16, z as u16));
        }
        self.mesh_dirty = true;
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> Voxel {
        *self.voxels.get(&(x as u16, y as u16, z as u16)).unwrap_or(&Voxel::empty())
    }

    pub fn fill_box(&mut self, x0: usize, y0: usize, z0: usize, x1: usize, y1: usize, z1: usize, v: Voxel) {
        for z in z0..=z1 {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    self.set(x, y, z, v);
                }
            }
        }
    }

    pub fn world_min(&self) -> Vec3 { self.pos }
    pub fn world_max(&self) -> Vec3 { self.pos + Vec3::new(self.size_x as f32, self.size_y as f32, self.size_z as f32) }
    pub fn world_center(&self) -> Vec3 { (self.world_min() + self.world_max()) * 0.5 }

    /// Max dimension for meshing grid
    fn mesh_grid_size(&self) -> usize { self.size_x.max(self.size_y).max(self.size_z) + 2 }

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
        let gs = self.size_x.max(self.size_y).max(self.size_z);
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
                if lx < self.size_x && ly < self.size_y && lz < self.size_z {
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

    /// Build mesh for rendering — export sparse to flat, then mesh
    pub fn build_mesh(&self) -> ChunkMesh {
        let gs = self.mesh_grid_size();
        let mut flat = vec![Voxel::empty(); gs * gs * gs];
        for (&(x, y, z), &v) in &self.voxels {
            let lx = x as usize + 1; // +1 border
            let ly = y as usize + 1;
            let lz = z as usize + 1;
            if lx < gs && ly < gs && lz < gs {
                flat[lz * gs * gs + ly * gs + lx] = v;
            }
        }
        let offset = self.pos - Vec3::new(1.0, 1.0, 1.0);
        generate_mesh(&flat, gs, offset, 1.0)
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
        self.voxels.len()
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

// ═══ All sizes in cm (1 voxel = 1 cm) ═══

/// Sofa: 200×45×80 cm, seat height 45, back 80
pub fn make_sofa(rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    let mut f = FurnitureObj::new("Sofa", pos, 200, 80, 85, 45.0, 800.0);
    let palettes: [([u8;3],[u8;3]); 4] = [
        ([75,115,165],[90,135,185]), ([165,75,80],[185,95,100]),
        ([85,140,85],[105,165,105]), ([155,130,85],[175,155,110]),
    ];
    let (base, cush) = palettes[rng.range(0, 3) as usize];
    let sb = Voxel::solid(5, base[0], base[1], base[2]);
    let sc = Voxel::solid(5, cush[0], cush[1], cush[2]);
    // Frame (base)
    f.fill_box(0, 0, 0, 199, 35, 84, sb);
    // Cushions
    f.fill_box(8, 35, 8, 191, 45, 76, sc);
    // Back
    f.fill_box(0, 35, 70, 199, 79, 84, sb);
    // Arms
    f.fill_box(0, 35, 0, 15, 60, 84, sb);
    f.fill_box(184, 35, 0, 199, 60, 84, sb);
    f.brittleness = 0.1;
    f
}

/// Coffee table: 120×45×60 cm
pub fn make_table(rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    let mut f = FurnitureObj::new("Coffee Table", pos, 120, 50, 60, 15.0, 400.0);
    let tw = vary(rng, 115, 72, 38, 12);
    let tt = vary(rng, 138, 90, 48, 10);
    // 4 legs (4×4 cm, height 43)
    f.fill_box(3, 0, 3, 6, 42, 6, tw);
    f.fill_box(113, 0, 3, 116, 42, 6, tw);
    f.fill_box(3, 0, 53, 6, 42, 56, tw);
    f.fill_box(113, 0, 53, 116, 42, 56, tw);
    // Tabletop (120×3×60)
    f.fill_box(0, 43, 0, 119, 45, 59, tt);
    f.brittleness = 0.3;
    f
}

/// TV stand: 120×50×40 cm (solid cabinet)
pub fn make_tv_stand(rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    let mut f = FurnitureObj::new("TV Stand", pos, 120, 50, 40, 25.0, 200.0);
    let tw = vary(rng, 115, 72, 38, 10);
    f.fill_box(0, 0, 0, 119, 49, 39, tw);
    // Shelf cutout
    f.fill_box(3, 3, 0, 116, 25, 36, Voxel::empty());
    f.brittleness = 0.3;
    f
}

/// TV: 100×60×5 cm (thin flat screen)
pub fn make_tv(_rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    let mut f = FurnitureObj::new("TV", pos, 100, 60, 8, 8.0, 1200.0);
    let screen = Voxel::solid(4, 20, 20, 28);
    let bezel = Voxel::solid(4, 50, 50, 55);
    f.fill_box(0, 0, 0, 99, 59, 7, bezel);
    f.fill_box(3, 3, 0, 96, 56, 3, screen);
    f.brittleness = 0.9;
    f
}

/// Vase: 15×25×15 cm (spheroid)
pub fn make_vase(rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    let mut f = FurnitureObj::new("Vase", pos, 16, 28, 16, 1.5, 200.0);
    let colors: [[u8;3]; 5] = [[195,65,55],[55,130,195],[195,180,55],[165,55,165],[55,175,120]];
    let c = colors[rng.range(0, 4) as usize];
    let v = Voxel::solid(3, c[0], c[1], c[2]);
    for z in 0..16 { for y in 0..28 { for x in 0..16 {
        let dx = x as f32 - 7.5; let dy = (y as f32 - 14.0) * 0.6; let dz = z as f32 - 7.5;
        if dx*dx + dy*dy + dz*dz <= 50.0 { f.set(x, y, z, v); }
    }}}
    f.brittleness = 1.0;
    f
}

/// Chair: 45×90×45 cm (seat at 45cm)
pub fn make_chair(rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    let mut f = FurnitureObj::new("Chair", pos, 45, 90, 45, 5.0, 150.0);
    let tw = vary(rng, 98, 62, 28, 12);
    // 4 legs (3×3, height 43)
    f.fill_box(2, 0, 2, 4, 42, 4, tw);
    f.fill_box(40, 0, 2, 42, 42, 4, tw);
    f.fill_box(2, 0, 40, 4, 42, 42, tw);
    f.fill_box(40, 0, 40, 42, 42, 42, tw);
    // Seat
    f.fill_box(0, 43, 0, 44, 47, 44, tw);
    // Back
    f.fill_box(0, 43, 40, 44, 89, 44, tw);
    f.brittleness = 0.2;
    f
}

/// Bookshelf: 80×180×30 cm (tall)
pub fn make_bookshelf(rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    let mut f = FurnitureObj::new("Bookshelf", pos, 80, 180, 32, 35.0, 350.0);
    let sw = vary(rng, 98, 62, 28, 10);
    // Frame
    f.fill_box(0, 0, 0, 79, 179, 31, sw);
    // 4 shelf cutouts
    for sy in [3, 45, 90, 135] {
        f.fill_box(3, sy, 0, 76, sy + 38, 28, Voxel::empty());
    }
    // Books on shelves
    let bc: [[u8;3]; 6] = [[55,75,135],[155,48,48],[48,125,55],[130,65,130],[180,160,50],[50,140,160]];
    for sy in [3, 45, 90, 135] {
        let mut bx = 5;
        for _ in 0..rng.range(3, 7) {
            let c = bc[rng.range(0, 5) as usize];
            let bw = rng.range(5, 15) as usize;
            let bh = rng.range(20, 35) as usize;
            if bx + bw < 75 {
                f.fill_box(bx, sy, 3, bx+bw, sy+bh, 26, Voxel::solid(9, c[0], c[1], c[2]));
                bx += bw + 2;
            }
        }
    }
    f.brittleness = 0.25;
    f
}

/// Floor lamp: 25×150×25 cm
pub fn make_lamp(_rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    let mut f = FurnitureObj::new("Lamp", pos, 30, 155, 30, 3.0, 100.0);
    let mt = Voxel::solid(4, 148, 148, 152);
    let shade = Voxel::solid(6, 225, 215, 180);
    // Base
    f.fill_box(10, 0, 10, 19, 3, 19, mt);
    // Pole
    f.fill_box(14, 3, 14, 15, 125, 15, mt);
    // Shade
    for z in 0..30 { for y in 125..155 { for x in 0..30 {
        let dx = x as f32 - 14.5; let dy = (y as f32 - 140.0) * 0.5; let dz = z as f32 - 14.5;
        if dx*dx + dy*dy + dz*dz <= 200.0 { f.set(x, y, z, shade); }
    }}}
    f.brittleness = 0.7;
    f
}

/// Fridge: 60×180×60 cm
pub fn make_fridge(_rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    let mut f = FurnitureObj::new("Fridge", pos, 60, 180, 65, 80.0, 1500.0);
    let fr = Voxel::solid(4, 235, 235, 240);
    f.fill_box(0, 0, 0, 59, 179, 64, fr);
    // Handle
    f.fill_box(0, 60, 25, 0, 120, 28, Voxel::solid(4, 160, 160, 165));
    // Divider line (freezer top)
    f.fill_box(0, 120, 0, 59, 121, 64, Voxel::solid(4, 210, 210, 215));
    f.brittleness = 0.05;
    f
}

/// Bed: 160×50×200 cm (double, with pillows + headboard)
pub fn make_bed(rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    let mut f = FurnitureObj::new("Bed", pos, 160, 90, 200, 80.0, 1200.0);
    let frame = vary(rng, 120, 80, 45, 10);
    let bed_colors: [[u8;3]; 4] = [[205,205,215],[185,215,205],[215,195,195],[195,205,225]];
    let bc = bed_colors[rng.range(0, 3) as usize];
    let mattress = Voxel::solid(5, bc[0], bc[1], bc[2]);
    let pillow = Voxel::solid(5, 240, 235, 225);
    // Frame box (low)
    f.fill_box(0, 0, 0, 159, 20, 199, frame);
    // Mattress
    f.fill_box(5, 20, 5, 154, 50, 194, mattress);
    // 2 pillows at head (z near 0)
    f.fill_box(10, 50, 5, 75, 65, 40, pillow);
    f.fill_box(85, 50, 5, 150, 65, 40, pillow);
    // Headboard (tall, back at z=0)
    f.fill_box(0, 20, 0, 159, 89, 8, frame);
    f.brittleness = 0.1;
    f
}

/// Nightstand: 45×55×45 cm (little cabinet)
pub fn make_nightstand(rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    let mut f = FurnitureObj::new("Nightstand", pos, 45, 55, 45, 8.0, 150.0);
    let wood = vary(rng, 110, 75, 40, 10);
    f.fill_box(0, 0, 0, 44, 54, 44, wood);
    // Drawer slit
    f.fill_box(5, 20, 0, 39, 22, 3, Voxel::solid(4, 140, 140, 145));
    // Handle
    f.fill_box(18, 15, 0, 26, 18, 2, Voxel::solid(4, 180, 170, 90));
    f.brittleness = 0.2;
    f
}

/// Wardrobe: 120×200×60 cm (tall closet)
pub fn make_wardrobe(rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    let mut f = FurnitureObj::new("Wardrobe", pos, 120, 200, 60, 60.0, 800.0);
    let wood = vary(rng, 130, 88, 50, 12);
    f.fill_box(0, 0, 0, 119, 199, 59, wood);
    // Door split down the middle
    f.fill_box(59, 5, 0, 61, 195, 2, Voxel::solid(4, 80, 60, 40));
    // Two handles
    f.fill_box(50, 95, 0, 56, 105, 3, Voxel::solid(4, 180, 170, 90));
    f.fill_box(64, 95, 0, 70, 105, 3, Voxel::solid(4, 180, 170, 90));
    f.brittleness = 0.15;
    f
}

/// Desk: 140×75×70 cm (writing desk with drawer)
pub fn make_desk(rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    let mut f = FurnitureObj::new("Desk", pos, 140, 75, 70, 30.0, 500.0);
    let wood = vary(rng, 115, 72, 38, 10);
    let top = vary(rng, 138, 90, 48, 8);
    // Solid side panels (acts as legs)
    f.fill_box(0, 0, 0, 4, 70, 69, wood);
    f.fill_box(135, 0, 0, 139, 70, 69, wood);
    // Back panel
    f.fill_box(5, 0, 0, 134, 70, 4, wood);
    // Drawer unit on right
    f.fill_box(95, 30, 5, 134, 70, 65, wood);
    // Drawer slit
    f.fill_box(95, 48, 5, 134, 50, 7, Voxel::solid(4, 140, 140, 145));
    // Tabletop
    f.fill_box(0, 70, 0, 139, 74, 69, top);
    f.brittleness = 0.25;
    f
}

/// Toilet: 40×75×60 cm (tank + bowl)
pub fn make_toilet(_rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    let mut f = FurnitureObj::new("Toilet", pos, 40, 75, 60, 35.0, 400.0);
    let porcelain = Voxel::solid(3, 245, 245, 248);
    // Bowl (front)
    for z in 0..40 { for y in 0..40 { for x in 0..40 {
        let dx = x as f32 - 19.5; let dz = z as f32 - 19.5;
        if dx*dx + dz*dz <= 380.0 && y > 10 { f.set(x, y, z, porcelain); }
    }}}
    // Base under bowl
    f.fill_box(10, 0, 10, 30, 10, 30, porcelain);
    // Tank (back)
    f.fill_box(5, 35, 40, 35, 70, 58, porcelain);
    // Seat line
    f.fill_box(0, 38, 0, 39, 40, 40, Voxel::solid(4, 180, 180, 185));
    f.brittleness = 0.95;
    f
}

/// Sink: 60×90×45 cm (pedestal sink)
pub fn make_sink(_rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    let mut f = FurnitureObj::new("Sink", pos, 60, 90, 45, 20.0, 300.0);
    let porcelain = Voxel::solid(3, 245, 245, 248);
    let metal = Voxel::solid(4, 200, 200, 210);
    // Pedestal
    f.fill_box(22, 0, 12, 37, 70, 33, porcelain);
    // Basin (hollow box on top)
    f.fill_box(0, 70, 0, 59, 88, 44, porcelain);
    f.fill_box(5, 75, 5, 54, 88, 39, Voxel::empty());
    // Faucet
    f.fill_box(27, 88, 2, 32, 95, 8, metal);
    f.brittleness = 0.85;
    f
}

/// Bathtub: 180×55×80 cm
pub fn make_bathtub(_rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    let mut f = FurnitureObj::new("Bathtub", pos, 180, 55, 80, 90.0, 1800.0);
    let porcelain = Voxel::solid(3, 245, 245, 248);
    // Outer hull
    f.fill_box(0, 0, 0, 179, 54, 79, porcelain);
    // Inner cavity
    f.fill_box(6, 10, 6, 173, 54, 73, Voxel::empty());
    // Faucet wall end
    f.fill_box(170, 55, 30, 178, 70, 50, Voxel::solid(4, 200, 200, 210));
    f.brittleness = 0.6;
    f
}
