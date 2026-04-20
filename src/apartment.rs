// ═══════════════════════════════════════════════════════════════
// PURRGE — Apartment Generator
// 4-quadrant layout with shuffled room roles + owner-influenced pools.
// 1 voxel = 1 cm. Grid 1024³ sparse (HashMap).
// ═══════════════════════════════════════════════════════════════

use glam::Vec3;
use std::collections::HashMap;
use crate::core::svo::Voxel;
use crate::core::procgen::Rng;
use crate::game::scoring::OwnerType;

pub const GRID: usize = 1024;
pub const FLOOR_Y: usize = 2;

/// Sparse voxel grid — HashMap, NOT flat array.
pub struct VoxelGrid {
    pub data: HashMap<(u32, u32, u32), Voxel>,
}

impl VoxelGrid {
    pub fn new() -> Self { Self { data: HashMap::with_capacity(500_000) } }

    pub fn set(&mut self, x: usize, y: usize, z: usize, v: Voxel) {
        if v.is_solid() {
            self.data.insert((x as u32, y as u32, z as u32), v);
        } else {
            self.data.remove(&(x as u32, y as u32, z as u32));
        }
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> Voxel {
        *self.data.get(&(x as u32, y as u32, z as u32)).unwrap_or(&Voxel::empty())
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

    pub fn voxel_count(&self) -> usize { self.data.len() }

    pub fn fill_sphere(&mut self, cx:f32, cy:f32, cz:f32, r:f32, v:Voxel) {
        let r2 = r * r;
        let ri = r.ceil() as i32;
        for dz in -ri..=ri { for dy in -ri..=ri { for dx in -ri..=ri {
            if (dx*dx + dy*dy + dz*dz) as f32 <= r2 {
                let x = (cx as i32 + dx) as usize;
                let y = (cy as i32 + dy) as usize;
                let z = (cz as i32 + dz) as usize;
                if x < GRID && y < GRID && z < GRID { self.set(x, y, z, v); }
            }
        }}}
    }

    pub fn fill_cyl(&mut self, cx:f32, cz:f32, r:f32, y0:usize, y1:usize, v:Voxel) {
        let r2 = r * r;
        let ri = r.ceil() as i32;
        for y in y0..=y1.min(GRID-1) { for dz in -ri..=ri { for dx in -ri..=ri {
            if (dx*dx + dz*dz) as f32 <= r2 {
                let x = (cx as i32 + dx) as usize;
                let z = (cz as i32 + dz) as usize;
                if x < GRID && z < GRID { self.set(x, y, z, v); }
            }
        }}}
    }

    /// Remove interior voxels (all 6 neighbors solid = invisible)
    pub fn hollow(&mut self) {
        let keys: Vec<(u32,u32,u32)> = self.data.keys().copied().collect();
        let mut remove = Vec::new();
        for (x, y, z) in keys {
            if x == 0 || y == 0 || z == 0 { continue; }
            let xu = x as usize; let yu = y as usize; let zu = z as usize;
            if self.get(xu-1, yu, zu).is_solid()
            && self.get(xu+1, yu, zu).is_solid()
            && self.get(xu, yu-1, zu).is_solid()
            && self.get(xu, yu+1, zu).is_solid()
            && self.get(xu, yu, zu-1).is_solid()
            && self.get(xu, yu, zu+1).is_solid() {
                remove.push((x, y, z));
            }
        }
        for k in remove { self.data.remove(&k); }
    }

    /// Raycast for camera collision / scratch targeting
    pub fn raycast(&self, origin: glam::Vec3, dir: glam::Vec3, max_dist: f32) -> Option<glam::Vec3> {
        let steps = (max_dist * 2.0) as usize;
        for i in 1..=steps {
            let t = i as f32 * 0.5;
            let p = origin + dir * t;
            let x = p.x as usize; let y = p.y as usize; let z = p.z as usize;
            if x < GRID && y < GRID && z < GRID && self.get(x, y, z).is_solid() {
                return Some(p - dir * 0.5);
            }
        }
        None
    }

    pub fn floor_at(&self, x: f32, z: f32, from_y: f32) -> f32 {
        let ix = x as usize; let iz = z as usize;
        if ix >= GRID || iz >= GRID { return FLOOR_Y as f32 + 1.0; }
        let top = (from_y as usize).min(GRID - 1);
        let bottom = top.saturating_sub(300);
        for y in (bottom..=top).rev() {
            if self.get(ix, y, iz).is_solid() { return y as f32 + 1.0; }
        }
        FLOOR_Y as f32 + 1.0
    }

    pub fn collides(&self, x: f32, z: f32, y: f32, r: f32) -> bool {
        let offsets: [(f32,f32); 8] = [
            (r,0.0),(-r,0.0),(0.0,r),(0.0,-r),
            (r*0.7,r*0.7),(-r*0.7,r*0.7),(r*0.7,-r*0.7),(-r*0.7,-r*0.7),
        ];
        for dy in [0.0, -4.0, -8.0, 4.0] {
            let cy = (y + dy) as usize;
            if cy >= GRID { continue; }
            for &(dx, dz) in &offsets {
                let cx = (x + dx) as usize;
                let cz = (z + dz) as usize;
                if cx < GRID && cz < GRID && self.get(cx, cy, cz).is_solid() {
                    return true;
                }
            }
        }
        false
    }

    pub fn scratch_at(&mut self, origin: glam::Vec3, forward: glam::Vec3, right: glam::Vec3) -> Vec<(glam::Vec3, Voxel)> {
        let mut debris = Vec::new();
        for claw in [-1.0_f32, 0.0, 1.0] {
            let base = origin + right * (claw * 10.0);
            let from = base + glam::Vec3::Y * 20.0 + forward * 8.0;
            let to = base - glam::Vec3::Y * 25.0 - forward * 5.0;
            let dir = to - from;
            let len = dir.length();
            if len < 0.1 { continue; }
            let steps = (len * 1.0) as usize;
            let r2 = 8.0_f32 * 8.0;
            let ri = 9_i32;
            for i in 0..=steps {
                let t = i as f32 / steps as f32;
                let p = from + dir * t;
                for dz in -ri..=ri { for dy in -ri..=ri { for dx in -ri..=ri {
                    if (dx*dx+dy*dy+dz*dz) as f32 <= r2 {
                        let x = (p.x as i32+dx) as usize;
                        let y = (p.y as i32+dy) as usize;
                        let z = (p.z as i32+dz) as usize;
                        if x < GRID && y < GRID && z < GRID && y > FLOOR_Y+1 {
                            let old = self.get(x, y, z);
                            if old.is_solid() {
                                debris.push((glam::Vec3::new(x as f32, y as f32, z as f32), old));
                                self.set(x, y, z, Voxel::empty());
                            }
                        }
                    }
                }}}
            }
        }
        debris
    }
}

// ═══════════════════════════════════════════════════════════════
// Color palette helpers
// ═══════════════════════════════════════════════════════════════

pub(crate) fn vary(rng: &mut Rng, r: u8, g: u8, b: u8, jitter: u8) -> Voxel {
    let j = jitter as i32;
    let vr = (r as i32 + rng.range(-j, j)).clamp(0, 255) as u8;
    let vg = (g as i32 + rng.range(-j, j)).clamp(0, 255) as u8;
    let vb = (b as i32 + rng.range(-j, j)).clamp(0, 255) as u8;
    Voxel::solid(1, vr, vg, vb)
}

// ═══════════════════════════════════════════════════════════════
// ROOM ROLES — what a quadrant can become
// ═══════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RoomRole {
    Living,
    Kitchen,
    Bedroom,
    Bathroom,
    Study,
}

impl RoomRole {
    fn name(self) -> &'static str {
        match self {
            RoomRole::Living => "Living", RoomRole::Kitchen => "Kitchen",
            RoomRole::Bedroom => "Bedroom", RoomRole::Bathroom => "Bathroom",
            RoomRole::Study => "Study",
        }
    }
}

// Fisher-Yates shuffle using our Rng
fn shuffle<T: Copy>(rng: &mut Rng, items: &mut [T]) {
    let n = items.len();
    for i in (1..n).rev() {
        let j = rng.range(0, i as i32) as usize;
        items.swap(i, j);
    }
}

// ═══════════════════════════════════════════════════════════════
// FURNITURE POOLS — per role + owner modifiers
// ═══════════════════════════════════════════════════════════════

use crate::furniture::{FurnitureObj, make_sofa, make_table, make_tv_stand, make_tv,
    make_vase, make_lamp, make_chair, make_bookshelf, make_fridge,
    make_bed, make_nightstand, make_wardrobe, make_desk, make_toilet,
    make_sink, make_bathtub};

#[derive(Clone, Copy)]
enum FurnKind {
    Sofa, Table, TvStand, Tv, Vase, Lamp, Chair, Bookshelf, Fridge,
    Bed, Nightstand, Wardrobe, Desk, Toilet, Sink, Bathtub,
}

fn furn_size(k: FurnKind) -> (usize, usize) {
    match k {
        FurnKind::Sofa => (200, 85),       FurnKind::Table => (120, 60),
        FurnKind::TvStand => (120, 40),    FurnKind::Tv => (100, 8),
        FurnKind::Vase => (16, 16),        FurnKind::Lamp => (30, 30),
        FurnKind::Chair => (45, 45),       FurnKind::Bookshelf => (80, 32),
        FurnKind::Fridge => (60, 65),
        FurnKind::Bed => (160, 200),       FurnKind::Nightstand => (45, 45),
        FurnKind::Wardrobe => (120, 60),   FurnKind::Desk => (140, 70),
        FurnKind::Toilet => (40, 60),      FurnKind::Sink => (60, 45),
        FurnKind::Bathtub => (180, 80),
    }
}

fn make_item(k: FurnKind, rng: &mut Rng, pos: Vec3) -> FurnitureObj {
    match k {
        FurnKind::Sofa => make_sofa(rng, pos),
        FurnKind::Table => make_table(rng, pos),
        FurnKind::TvStand => make_tv_stand(rng, pos),
        FurnKind::Tv => make_tv(rng, pos),
        FurnKind::Vase => make_vase(rng, pos),
        FurnKind::Lamp => make_lamp(rng, pos),
        FurnKind::Chair => make_chair(rng, pos),
        FurnKind::Bookshelf => make_bookshelf(rng, pos),
        FurnKind::Fridge => make_fridge(rng, pos),
        FurnKind::Bed => make_bed(rng, pos),
        FurnKind::Nightstand => make_nightstand(rng, pos),
        FurnKind::Wardrobe => make_wardrobe(rng, pos),
        FurnKind::Desk => make_desk(rng, pos),
        FurnKind::Toilet => make_toilet(rng, pos),
        FurnKind::Sink => make_sink(rng, pos),
        FurnKind::Bathtub => make_bathtub(rng, pos),
    }
}

/// Base pool for a room role — (kind, min, max)
fn base_pool(role: RoomRole) -> Vec<(FurnKind, i32, i32)> {
    match role {
        RoomRole::Living => vec![
            (FurnKind::Sofa, 1, 1), (FurnKind::TvStand, 1, 1), (FurnKind::Tv, 1, 1),
            (FurnKind::Table, 1, 2), (FurnKind::Vase, 1, 3), (FurnKind::Lamp, 1, 2),
            (FurnKind::Chair, 0, 2), (FurnKind::Bookshelf, 0, 1),
        ],
        RoomRole::Kitchen => vec![
            (FurnKind::Table, 1, 1), (FurnKind::Chair, 2, 4), (FurnKind::Fridge, 1, 1),
            (FurnKind::Lamp, 0, 1), (FurnKind::Vase, 0, 2),
        ],
        RoomRole::Bedroom => vec![
            (FurnKind::Bed, 1, 1), (FurnKind::Nightstand, 1, 2), (FurnKind::Wardrobe, 1, 1),
            (FurnKind::Lamp, 1, 2), (FurnKind::Vase, 0, 1), (FurnKind::Bookshelf, 0, 1),
        ],
        RoomRole::Bathroom => vec![
            (FurnKind::Toilet, 1, 1), (FurnKind::Sink, 1, 1), (FurnKind::Bathtub, 1, 1),
            (FurnKind::Vase, 0, 1),
        ],
        RoomRole::Study => vec![
            (FurnKind::Desk, 1, 1), (FurnKind::Chair, 1, 1), (FurnKind::Bookshelf, 2, 4),
            (FurnKind::Lamp, 1, 2), (FurnKind::Vase, 0, 2),
        ],
    }
}

/// Owner personality shifts furniture pool (design doc §12)
fn adjust_pool_for_owner(pool: &mut Vec<(FurnKind, i32, i32)>, owner: &OwnerType) {
    match owner {
        OwnerType::Gamer => {
            // More TVs, more tables (for consoles), less vases
            for p in pool.iter_mut() {
                match p.0 {
                    FurnKind::Tv | FurnKind::TvStand => { p.1 = (p.1 + 1).min(3); p.2 = (p.2 + 1).min(4); }
                    FurnKind::Table => { p.2 = (p.2 + 1).min(4); }
                    FurnKind::Vase => { p.2 = (p.2 - 1).max(0); }
                    _ => {}
                }
            }
        }
        OwnerType::Bookworm => {
            for p in pool.iter_mut() {
                if matches!(p.0, FurnKind::Bookshelf) { p.1 = (p.1 + 2).min(5); p.2 = (p.2 + 3).min(8); }
            }
        }
        OwnerType::Minimalist => {
            // Halve counts (rounded down), min 0
            for p in pool.iter_mut() {
                p.2 = (p.2 / 2).max(p.1);
            }
        }
        OwnerType::Hoarder => {
            // Double counts + extra clutter
            for p in pool.iter_mut() {
                p.2 = (p.2 * 2).min(10);
                if matches!(p.0, FurnKind::Vase | FurnKind::Lamp) { p.1 = (p.1 + 1).min(4); }
            }
        }
        OwnerType::Nostalgic => {
            // Lots of stuff on tables/shelves (vases stand in for photos for now)
            for p in pool.iter_mut() {
                if matches!(p.0, FurnKind::Vase) { p.1 = (p.1 + 2).min(5); p.2 = (p.2 + 3).min(8); }
            }
        }
        OwnerType::PlantLover => {
            // Extra vases (potted plants)
            for p in pool.iter_mut() {
                if matches!(p.0, FurnKind::Vase) { p.1 = (p.1 + 2).min(5); p.2 = (p.2 + 3).min(8); }
            }
        }
        OwnerType::Artist | OwnerType::Fitness | OwnerType::CatLover | OwnerType::Normal => {}
    }
}

// ═══════════════════════════════════════════════════════════════
// MAIN GENERATOR
// ═══════════════════════════════════════════════════════════════

/// Generate a full apartment with shuffled room roles.
/// Owner personality influences furniture counts.
pub fn generate_apartment_v2(seed: u64, owner: &OwnerType) -> (VoxelGrid, Vec<FurnitureObj>) {
    let mut g = VoxelGrid::new();
    let mut rng = Rng::new(seed);
    let mut furn: Vec<FurnitureObj> = Vec::new();

    let m = 20_usize;
    let w = 3_usize;
    let h = 280_usize;
    let fy = FLOOR_Y as f32 + 1.0;

    let left = m + w;
    let right = GRID - m - w;
    let front = m + w;
    let back = GRID - m - w;
    let width = right - left;
    let depth = back - front;

    // Internal walls — two splits give 4 quadrants
    let wall_z = front + (depth * 50 / 100) + rng.range(-25, 25) as usize;
    let wall_x = left + (width * 50 / 100) + rng.range(-25, 25) as usize;

    // Colors
    let fl_wood = vary(&mut rng, 175, 148, 108, 15);
    let fl_tile = vary(&mut rng, 198, 193, 186, 8);
    let fl_bath = vary(&mut rng, 210, 220, 225, 6);
    let wl = vary(&mut rng, 232, 226, 218, 6);
    let cl = vary(&mut rng, 245, 245, 252, 4);

    // ── SHELL ──
    g.fill_box(left, FLOOR_Y, front, right, FLOOR_Y, back, fl_wood);
    g.fill_box(left, h, front, right, h, back, cl);
    g.fill_box(left-w, FLOOR_Y, front-w, left, h, back+w, wl);
    g.fill_box(right, FLOOR_Y, front-w, right+w, h, back+w, wl);
    g.fill_box(left, FLOOR_Y, front-w, right, h, front, wl);
    g.fill_box(left, FLOOR_Y, back, right, h, back+w, wl);

    // ── INTERNAL WALLS (full cross = 4 quadrants) ──
    g.fill_box(left, FLOOR_Y, wall_z, right, h, wall_z+w, wl);
    g.fill_box(wall_x, FLOOR_Y, front, wall_x+w, h, back, wl);

    // ── SHUFFLE ROOM ROLES INTO 4 QUADRANTS ──
    // Quadrants: 0=FL (front-left), 1=FR (front-right), 2=BL (back-left), 3=BR (back-right)
    // We always include Living + Kitchen + Bedroom + one of [Bathroom, Study]
    let fourth = if rng.chance(0.7) { RoomRole::Bathroom } else { RoomRole::Study };
    let mut roles = [RoomRole::Living, RoomRole::Kitchen, RoomRole::Bedroom, fourth];
    shuffle(&mut rng, &mut roles);

    // Quadrant bounds: (x_min, z_min, x_max, z_max)
    let quads: [(usize, usize, usize, usize); 4] = [
        (left,     front,      wall_x,    wall_z),    // FL
        (wall_x+w, front,      right,     wall_z),    // FR
        (left,     wall_z+w,   wall_x,    back),      // BL
        (wall_x+w, wall_z+w,   right,     back),      // BR
    ];

    // Floor material per role
    for qi in 0..4 {
        let (x0, z0, x1, z1) = quads[qi];
        let mat = match roles[qi] {
            RoomRole::Kitchen => fl_tile,
            RoomRole::Bathroom => fl_bath,
            _ => fl_wood,
        };
        g.fill_box(x0, FLOOR_Y, z0, x1.min(GRID-1), FLOOR_Y, z1.min(GRID-1), mat);
    }

    // ── DOORS between adjacent quadrants ──
    let dw = 90_usize; let dh = 220_usize;
    let frame = Voxel::solid(1, 155, 125, 85);

    let mut cut_door = |g: &mut VoxelGrid, wall_axis: char, wall_coord: usize, span_min: usize, span_max: usize, rng: &mut Rng| {
        let span = span_max.saturating_sub(span_min);
        if span < dw + 10 { return; }
        let pos = span_min + 5 + rng.range(0, (span as i32 - dw as i32 - 10).max(1)) as usize;
        if wall_axis == 'x' {
            // wall is along X (i.e. wall_coord is Z)
            g.fill_box(pos, FLOOR_Y+1, wall_coord, pos+dw, dh, wall_coord+w, Voxel::empty());
            g.fill_box(pos-2, FLOOR_Y+1, wall_coord, pos, dh+2, wall_coord+w, frame);
            g.fill_box(pos+dw, FLOOR_Y+1, wall_coord, pos+dw+2, dh+2, wall_coord+w, frame);
        } else {
            // wall is along Z (i.e. wall_coord is X)
            g.fill_box(wall_coord, FLOOR_Y+1, pos, wall_coord+w, dh, pos+dw, Voxel::empty());
            g.fill_box(wall_coord, FLOOR_Y+1, pos-2, wall_coord+w, dh+2, pos, frame);
            g.fill_box(wall_coord, FLOOR_Y+1, pos+dw, wall_coord+w, dh+2, pos+dw+2, frame);
        }
    };

    // FL↔FR (door in vertical wall, front half: wall_x between front..wall_z)
    cut_door(&mut g, 'z', wall_x, front + 10, wall_z - 10, &mut rng);
    // BL↔BR (door in vertical wall, back half)
    cut_door(&mut g, 'z', wall_x, wall_z + w + 10, back - 10, &mut rng);
    // FL↔BL (door in horizontal wall, left half)
    cut_door(&mut g, 'x', wall_z, left + 10, wall_x - 10, &mut rng);
    // FR↔BR (door in horizontal wall, right half)
    cut_door(&mut g, 'x', wall_z, wall_x + w + 10, right - 10, &mut rng);

    // ── PROCEDURAL FURNITURE PLACEMENT PER QUADRANT ──
    let mut placed: Vec<(f32, f32, f32, f32)> = Vec::new();

    let collides_any = |px: f32, pz: f32, sx: f32, sz: f32, placed: &[(f32,f32,f32,f32)]| -> bool {
        let gap = 20.0;
        for &(ax, az, bx, bz) in placed {
            if px - gap < bx && px + sx + gap > ax && pz - gap < bz && pz + sz + gap > az {
                return true;
            }
        }
        false
    };

    for qi in 0..4 {
        let role = roles[qi];
        let (rx_min, rz_min, rx_max, rz_max) = quads[qi];
        // Inner padding so furniture doesn't touch walls
        let pad = 15_usize;
        let in_min_x = rx_min + pad;
        let in_min_z = rz_min + pad;
        let in_max_x = rx_max.saturating_sub(pad);
        let in_max_z = rz_max.saturating_sub(pad);

        let mut pool = base_pool(role);
        adjust_pool_for_owner(&mut pool, owner);

        for (kind, count_min, count_max) in pool {
            let count = rng.range(count_min, count_max);
            let (sx, sz) = furn_size(kind);
            for _ in 0..count {
                let mut done = false;
                for _ in 0..24 {
                    let max_x = (in_max_x as i32 - sx as i32).max(in_min_x as i32 + 1);
                    let max_z = (in_max_z as i32 - sz as i32).max(in_min_z as i32 + 1);
                    let px = rng.range(in_min_x as i32, max_x) as f32;
                    let pz = rng.range(in_min_z as i32, max_z) as f32;
                    if !collides_any(px, pz, sx as f32, sz as f32, &placed) {
                        // TV gets stacked on nearest TvStand
                        let py = if matches!(kind, FurnKind::Tv) {
                            furn.iter().rev().find(|f| f.name == "TV Stand"
                                && (f.pos.x - px).abs() < 150.0 && (f.pos.z - pz).abs() < 150.0)
                                .map(|f| f.pos.y + f.size_y as f32 + 1.0).unwrap_or(fy)
                        } else { fy };
                        let item = make_item(kind, &mut rng, Vec3::new(px, py, pz));
                        placed.push((px, pz, px + sx as f32, pz + sz as f32));
                        furn.push(item);
                        done = true;
                        break;
                    }
                }
                if !done { /* silently drop — room too crowded */ }
            }
        }
    }

    println!("  Apartment: {} voxels, {} furniture, seed={}, owner={:?}",
             g.voxel_count(), furn.len(), seed, owner);
    print!("  Layout: ");
    for (qi, lbl) in [(0,"FL"), (1,"FR"), (2,"BL"), (3,"BR")].iter() {
        print!("{}={} ", lbl, roles[*qi].name());
    }
    println!();

    (g, furn)
}
