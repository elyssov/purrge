// ═══════════════════════════════════════════════════════════════
// PURRGE — Apartment Generator
// Procedural apartment with 3-4 rooms. Uses engine Voxel type.
// ═══════════════════════════════════════════════════════════════

use crate::core::svo::Voxel;
use crate::core::procgen::Rng;

pub const GRID: usize = 192; // was 256 — smaller = faster room mesh rebuild on scratch
pub const TOTAL: usize = GRID * GRID * GRID;
pub const FLOOR_Y: usize = 2;

/// Flat voxel grid for apartment
pub struct VoxelGrid {
    pub data: Vec<Voxel>,
}

impl VoxelGrid {
    pub fn new() -> Self { Self { data: vec![Voxel::empty(); TOTAL] } }

    pub fn set(&mut self, x: usize, y: usize, z: usize, v: Voxel) {
        if x < GRID && y < GRID && z < GRID {
            self.data[z * GRID * GRID + y * GRID + x] = v;
        }
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> Voxel {
        if x < GRID && y < GRID && z < GRID {
            self.data[z * GRID * GRID + y * GRID + x]
        } else {
            Voxel::empty()
        }
    }

    pub fn fill_box(&mut self, x0:usize,y0:usize,z0:usize, x1:usize,y1:usize,z1:usize, v:Voxel) {
        for z in z0..=z1.min(GRID-1) {
            for y in y0..=y1.min(GRID-1) {
                for x in x0..=x1.min(GRID-1) {
                    self.set(x, y, z, v);
                }
            }
        }
    }

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

    /// Remove interior voxels (optimization for meshing)
    pub fn hollow(&mut self) {
        let mut remove = Vec::new();
        for z in 1..GRID-1 { for y in 1..GRID-1 { for x in 1..GRID-1 {
            let i = z * GRID * GRID + y * GRID + x;
            if self.data[i].is_empty() { continue; }
            if self.data[i-1].is_solid() && self.data[i+1].is_solid()
            && self.data[i-GRID].is_solid() && self.data[i+GRID].is_solid()
            && self.data[i-GRID*GRID].is_solid() && self.data[i+GRID*GRID].is_solid() {
                remove.push(i);
            }
        }}}
        for i in remove { self.data[i] = Voxel::empty(); }
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

    /// Floor height at position
    pub fn floor_at(&self, x: f32, z: f32, from_y: f32) -> f32 {
        let ix = x as usize; let iz = z as usize;
        if ix >= GRID || iz >= GRID { return FLOOR_Y as f32 + 1.0; }
        for y in (0..=(from_y as usize).min(GRID-1)).rev() {
            if self.get(ix, y, iz).is_solid() { return y as f32 + 1.0; }
        }
        FLOOR_Y as f32 + 1.0
    }

    /// Collision check for cat-sized body
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

    /// Scratch: 3 claw lines, returns destroyed voxels
    pub fn scratch_at(&mut self, origin: glam::Vec3, forward: glam::Vec3, right: glam::Vec3) -> Vec<(glam::Vec3, Voxel)> {
        let mut debris = Vec::new();
        // 3 claw swipes, wider spread, bigger radius — VKUSNO!
        for claw in [-1.5_f32, 0.0, 1.5] {
            let base = origin + right * (claw * 3.5);
            let from = base + glam::Vec3::Y * 8.0 + forward * 3.0;
            let to = base - glam::Vec3::Y * 10.0 - forward * 2.0;
            let dir = to - from;
            let len = dir.length();
            if len < 0.1 { continue; }
            let steps = (len * 2.0) as usize;
            let r2 = 2.5_f32 * 2.5; // bigger claw radius
            let ri = 3_i32;
            for i in 0..=steps {
                let t = i as f32 / steps as f32;
                let p = from + dir * t;
                for dz in -ri..=ri { for dy in -ri..=ri { for dx in -ri..=ri {
                    if (dx*dx+dy*dy+dz*dz) as f32 <= r2 {
                        let x = (p.x as i32+dx) as usize;
                        let y = (p.y as i32+dy) as usize;
                        let z = (p.z as i32+dz) as usize;
                        if x < GRID && y < GRID && z < GRID && y > FLOOR_Y+1 {
                            let idx = z*GRID*GRID + y*GRID + x;
                            let old = self.data[idx];
                            if old.is_solid() {
                                debris.push((glam::Vec3::new(x as f32, y as f32, z as f32), old));
                                self.data[idx] = Voxel::empty();
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
// Color palette helpers — seed-based variation
// ═══════════════════════════════════════════════════════════════

/// Pick a color variant: base ± jitter per channel
fn vary(rng: &mut Rng, r: u8, g: u8, b: u8, jitter: u8) -> Voxel {
    let j = jitter as i32;
    let vr = (r as i32 + rng.range(-j, j)).clamp(0, 255) as u8;
    let vg = (g as i32 + rng.range(-j, j)).clamp(0, 255) as u8;
    let vb = (b as i32 + rng.range(-j, j)).clamp(0, 255) as u8;
    Voxel::solid(1, vr, vg, vb)
}

// ═══════════════════════════════════════════════════════════════
// Furniture catalogue — each item is a function
// ═══════════════════════════════════════════════════════════════

/// Sofa at (x, z) along X axis. Random color from palette.
fn place_sofa(g: &mut VoxelGrid, rng: &mut Rng, x: usize, z: usize) {
    let palettes: [([u8;3],[u8;3]); 5] = [
        ([75,115,165], [90,135,185]),   // blue
        ([165,75,80],  [185,95,100]),   // burgundy
        ([85,140,85],  [105,165,105]),  // green
        ([155,130,85], [175,155,110]),   // tan
        ([120,95,145], [145,120,170]),   // purple
    ];
    let (base, cush) = palettes[rng.range(0, palettes.len() as i32 - 1) as usize];
    let sb = Voxel::solid(5, base[0], base[1], base[2]);
    let sc = Voxel::solid(5, cush[0], cush[1], cush[2]);
    // Seat
    g.fill_box(x, FLOOR_Y+1, z-35, x+26, 22, z+35, sb);
    g.fill_box(x+2, 22, z-33, x+24, 26, z+33, sc);
    // Back + arms
    g.fill_box(x, 22, z-35, x+5, 44, z+35, sb);
    g.fill_box(x, 22, z-35, x+26, 32, z-30, sb);
    g.fill_box(x, 22, z+30, x+26, 32, z+35, sb);
}

/// Coffee table at (x, z)
fn place_coffee_table(g: &mut VoxelGrid, rng: &mut Rng, x: usize, z: usize) {
    let tw = vary(rng, 115, 72, 38, 15);
    let tt = vary(rng, 138, 90, 48, 10);
    g.fill_box(x, FLOOR_Y+1, z-14, x+30, 14, z+14, tw);
    g.fill_box(x-2, 14, z-16, x+32, 16, z+16, tt);
}

/// TV stand + TV at (x, z), facing -X
fn place_tv(g: &mut VoxelGrid, _rng: &mut Rng, x: usize, z: usize) {
    let tw = Voxel::solid(1, 115, 72, 38);
    let tvf = Voxel::solid(4, 28, 28, 32);
    let tvb = Voxel::solid(4, 55, 55, 60);
    // Stand
    g.fill_box(x, FLOOR_Y+1, z-8, x+8, 40, z+8, tw);
    // Screen
    g.fill_box(x+8, 40, z-28, x+11, 78, z+28, tvf);
    // Bezel
    g.fill_box(x+7, 39, z-30, x+12, 80, z-28, tvb);
    g.fill_box(x+7, 39, z+28, x+12, 80, z+30, tvb);
    g.fill_box(x+7, 78, z-30, x+12, 80, z+30, tvb);
}

/// Floor lamp at (x, z)
fn place_floor_lamp(g: &mut VoxelGrid, rng: &mut Rng, x: usize, z: usize) {
    let mt = Voxel::solid(4, 148, 148, 152);
    let shades: [[u8;3]; 4] = [[225,215,180],[220,180,160],[180,210,200],[210,200,220]];
    let shade = shades[rng.range(0, 3) as usize];
    let lm = Voxel::solid(6, shade[0], shade[1], shade[2]);
    g.fill_box(x, FLOOR_Y+1, z, x+2, 60, z+2, mt);
    g.fill_sphere(x as f32 + 1.0, 62.0, z as f32 + 1.0, 4.0, lm);
}

/// Cardboard box (hollow) at (x, z)
fn place_box(g: &mut VoxelGrid, _rng: &mut Rng, x: usize, z: usize) {
    let bx = Voxel::solid(9, 175, 145, 90);
    g.fill_box(x, FLOOR_Y+1, z, x+14, FLOOR_Y+14, z+14, bx);
    g.fill_box(x+2, FLOOR_Y+3, z+2, x+12, FLOOR_Y+14, z+12, Voxel::empty());
}

/// Parrot cage with bird at (x, z)
fn place_parrot_cage(g: &mut VoxelGrid, rng: &mut Rng, x: usize, z: usize) {
    let mt = Voxel::solid(4, 148, 148, 152);
    let cage = Voxel::solid(4, 180, 170, 100);
    let bird_colors: [[u8;3]; 3] = [[50,180,60],[180,60,50],[50,120,200]];
    let bc = bird_colors[rng.range(0, 2) as usize];
    let bird = Voxel::solid(9, bc[0], bc[1], bc[2]);
    // Stand
    g.fill_box(x-2, FLOOR_Y+1, z-2, x+2, 35, z+2, mt);
    // Cage
    g.fill_box(x-6, 35, z-6, x+6, 55, z+6, cage);
    g.fill_box(x-5, 36, z-5, x+5, 54, z+5, Voxel::empty());
    // Bird
    g.fill_sphere(x as f32, 42.0, z as f32, 2.5, bird);
    g.fill_sphere(x as f32, 45.0, z as f32, 2.0, Voxel::solid(9, bc[0].saturating_add(20), bc[1].saturating_add(20), bc[2]));
}

/// Kitchen table with 4 legs at (cx, cz)
fn place_kitchen_table(g: &mut VoxelGrid, rng: &mut Rng, cx: usize, cz: usize) {
    let tw = vary(rng, 115, 72, 38, 12);
    let tt = vary(rng, 138, 90, 48, 10);
    // 4 legs
    for &(dx,dz) in &[(-18i32,-14),( 16,-14),(-18, 12),( 16, 12)] {
        let lx = (cx as i32 + dx) as usize;
        let lz = (cz as i32 + dz) as usize;
        g.fill_box(lx, FLOOR_Y+1, lz, lx+2, 52, lz+2, tw);
    }
    // Tabletop
    g.fill_box(cx-20, 52, cz-16, cx+20, 54, cz+16, tt);
}

/// Chair at (x, z) with random seat color
fn place_chair(g: &mut VoxelGrid, rng: &mut Rng, x: usize, z: usize) {
    let tw = vary(rng, 98, 62, 28, 12);
    // 4 legs
    for &(dx,dz) in &[(-7i32,-7),(5,-7),(-7,5),(5,5)] {
        g.fill_box((x as i32+dx) as usize, FLOOR_Y+1, (z as i32+dz) as usize,
                   (x as i32+dx+2) as usize, 28, (z as i32+dz+2) as usize, tw);
    }
    // Seat
    g.fill_box(x-8, 28, z-8, x+8, 30, z+8, tw);
    // Back
    g.fill_box(x-8, 28, z+6, x+8, 50, z+8, tw);
}

/// Vase (sphere on surface) at (x, y, z)
fn place_vase(g: &mut VoxelGrid, rng: &mut Rng, x: f32, y: f32, z: f32) {
    let colors: [[u8;3]; 5] = [[195,65,55],[55,130,195],[195,180,55],[165,55,165],[55,175,120]];
    let c = colors[rng.range(0, 4) as usize];
    g.fill_sphere(x, y, z, 3.5, Voxel::solid(3, c[0], c[1], c[2]));
}

/// Rug (flat carpet) at area
fn place_rug(g: &mut VoxelGrid, rng: &mut Rng, x0: usize, z0: usize, x1: usize, z1: usize) {
    let rugs: [[u8;3]; 5] = [[135,55,65],[55,85,135],[85,130,80],[145,120,80],[110,70,110]];
    let c = rugs[rng.range(0, 4) as usize];
    g.fill_box(x0, FLOOR_Y+1, z0, x1, FLOOR_Y+1, z1, Voxel::solid(5, c[0], c[1], c[2]));
}

/// Bed with headboard and pillows
fn place_bed(g: &mut VoxelGrid, rng: &mut Rng, x0: usize, z0: usize, x1: usize, z1: usize) {
    let tw = vary(rng, 120, 80, 45, 10);
    let bed_colors: [[u8;3]; 4] = [[200,200,210],[180,210,200],[210,190,190],[190,200,220]];
    let bc = bed_colors[rng.range(0, 3) as usize];
    let bed = Voxel::solid(5, bc[0], bc[1], bc[2]);
    let pil = Voxel::solid(5, 240, 235, 225);
    // Frame
    g.fill_box(x0, FLOOR_Y+1, z0, x1, 24, z1, tw);
    // Mattress
    g.fill_box(x0+2, 24, z0+2, x1-2, 30, z1-2, bed);
    // Pillows (near z1)
    g.fill_box(x0+5, 30, z1-10, x1-5, 34, z1-2, pil);
    // Headboard
    g.fill_box(x0, 24, z1-4, x1, 42, z1, tw);
}

/// Nightstand at (x, z)
fn place_nightstand(g: &mut VoxelGrid, rng: &mut Rng, x: usize, z: usize) {
    let tw = vary(rng, 110, 75, 40, 10);
    g.fill_box(x, FLOOR_Y+1, z, x+20, 28, z+20, tw);
    // Lamp on top
    let shades: [[u8;3]; 3] = [[225,215,180],[200,220,210],[220,200,210]];
    let shade = shades[rng.range(0, 2) as usize];
    g.fill_sphere(x as f32 + 10.0, 32.0, z as f32 + 10.0, 3.0, Voxel::solid(6, shade[0], shade[1], shade[2]));
}

/// Wardrobe/closet at (x0, z0) to (x1, z1), full height
fn place_wardrobe(g: &mut VoxelGrid, rng: &mut Rng, x0: usize, z0: usize, x1: usize, z1: usize) {
    let wd = vary(rng, 130, 88, 50, 12);
    g.fill_box(x0, FLOOR_Y+1, z0, x1, 85, z1, wd);
}

/// Bookshelf with colorful books
fn place_bookshelf(g: &mut VoxelGrid, rng: &mut Rng, x0: usize, z0: usize, x1: usize, z1: usize) {
    let sw = vary(rng, 98, 62, 28, 10);
    g.fill_box(x0, FLOOR_Y+1, z0, x1, 80, z1, sw);
    // Shelves (hollow compartments)
    for sy in [16_usize, 38, 58] {
        g.fill_box(x0+2, sy, z0+2, x1-2, sy+14, z1-2, Voxel::empty());
    }
    // Random books on each shelf
    let book_colors: [[u8;3]; 6] = [
        [55,75,135],[155,48,48],[48,125,55],[180,160,50],[130,65,130],[50,140,160],
    ];
    for sy in [16_usize, 38, 58] {
        let n_books = rng.range(1, 3) as usize;
        let mut bx = x0 + 3;
        for _ in 0..n_books {
            let bc = book_colors[rng.range(0, 5) as usize];
            let bw = rng.range(4, 10) as usize;
            let bh = rng.range(8, 12) as usize;
            if bx + bw < x1 - 2 {
                g.fill_box(bx, sy, z0+2, bx+bw, sy+bh, z1-2, Voxel::solid(9, bc[0], bc[1], bc[2]));
                bx += bw + 1;
            }
        }
    }
}

/// Desk at (x0, z0)
fn place_desk(g: &mut VoxelGrid, rng: &mut Rng, x0: usize, z0: usize, x1: usize, z1: usize) {
    let tw = vary(rng, 115, 72, 38, 12);
    let tt = vary(rng, 138, 90, 48, 10);
    g.fill_box(x0, FLOOR_Y+1, z0, x1, 38, z1, tw);
    g.fill_box(x0-2, 38, z0-2, x1+2, 40, z1+1, tt);
    // Desk lamp
    let mt = Voxel::solid(4, 148, 148, 152);
    let lx = x0 + 10; let lz = z0 + 4;
    g.fill_box(lx, 40, lz, lx+2, 52, lz+2, mt);
    g.fill_sphere(lx as f32 + 1.0, 54.0, lz as f32 + 1.0, 3.5, Voxel::solid(6, 225, 215, 180));
}

/// Kitchen counter along wall
fn place_counter(g: &mut VoxelGrid, rng: &mut Rng, x0: usize, z0: usize, x1: usize, z1: usize) {
    let ct = vary(rng, 198, 192, 188, 8);
    let tt = vary(rng, 138, 90, 48, 8);
    g.fill_box(x0, FLOOR_Y+1, z0, x1, 44, z1, ct);
    g.fill_box(x0, 44, z0, x1, 46, z1, tt);
}

/// Microwave on counter
fn place_microwave(g: &mut VoxelGrid, _rng: &mut Rng, x0: usize, y: usize, z0: usize) {
    let mb = Voxel::solid(4, 178, 178, 182);
    g.fill_box(x0, y, z0, x0+24, y+14, z0+8, mb);
}

/// Fridge
fn place_fridge(g: &mut VoxelGrid, rng: &mut Rng, x: usize, z: usize) {
    let colors: [[u8;3]; 3] = [[235,235,240],[200,200,205],[220,215,210]];
    let c = colors[rng.range(0, 2) as usize];
    let fr = Voxel::solid(4, c[0], c[1], c[2]);
    g.fill_box(x, FLOOR_Y+1, z, x+14, 80, z+14, fr);
    // Handle
    g.fill_box(x-1, 30, z+5, x, 60, z+7, Voxel::solid(4, 160, 160, 165));
}

/// Dog bed at (x, z)
fn place_dog_bed(g: &mut VoxelGrid, rng: &mut Rng, x: usize, z: usize) {
    let beds: [[u8;3]; 3] = [[120,90,70],[90,110,130],[130,100,90]];
    let c = beds[rng.range(0, 2) as usize];
    g.fill_box(x, FLOOR_Y+1, z, x+18, FLOOR_Y+5, z+15, Voxel::solid(5, c[0], c[1], c[2]));
}

/// Plant in pot
fn place_plant(g: &mut VoxelGrid, rng: &mut Rng, x: f32, y: f32, z: f32) {
    // Pot
    let pot_colors: [[u8;3]; 3] = [[165,80,55],[100,90,80],[75,100,85]];
    let pc = pot_colors[rng.range(0, 2) as usize];
    g.fill_cyl(x, z, 4.0, y as usize, y as usize + 8, Voxel::solid(3, pc[0], pc[1], pc[2]));
    // Foliage
    g.fill_sphere(x, y + 12.0, z, 5.0, Voxel::solid(9, 60, 140, 55));
    g.fill_sphere(x + 2.0, y + 14.0, z - 1.0, 3.5, Voxel::solid(9, 50, 155, 50));
}

// ═══════════════════════════════════════════════════════════════
// MAIN GENERATOR — procedural layout + furniture
// ═══════════════════════════════════════════════════════════════

/// Generate a full apartment. Layout & furniture vary with seed.
pub fn generate_apartment(seed: u64) -> VoxelGrid {
    let mut g = VoxelGrid::new();
    let mut rng = Rng::new(seed);

    let m = 6_usize; let w = 3_usize; let h = 195_usize;

    // Procedural wall positions (within range)
    let wall_z = 140 + rng.range(0, 16) as usize; // horizontal divider (140-156)
    let wall_x = 120 + rng.range(0, 12) as usize;  // vertical divider in back half

    // Floor/ceiling/wall colors — slight variation per seed
    let fl1 = vary(&mut rng, 175, 150, 110, 10); // wood floor
    let fl2 = vary(&mut rng, 195, 190, 185, 8);  // tile floor (kitchen)
    let fl3 = vary(&mut rng, 200, 205, 210, 8);  // tile floor (study)
    let wl  = vary(&mut rng, 228, 222, 212, 6);  // walls
    let cl  = vary(&mut rng, 242, 242, 248, 4);  // ceiling

    // ── STRUCTURE ──
    g.fill_box(m, 0, m, GRID-m, FLOOR_Y, GRID-m, fl1);     // base floor
    g.fill_box(m, h-w, m, GRID-m, h, GRID-m, cl);           // ceiling
    // Outer walls
    g.fill_box(m, 0, GRID-m-w, GRID-m, h, GRID-m, wl);
    g.fill_box(m, 0, m, GRID-m, h, m+w, wl);
    g.fill_box(m, 0, m, m+w, h, GRID-m, wl);
    g.fill_box(GRID-m-w, 0, m, GRID-m, h, GRID-m, wl);
    // Internal walls
    g.fill_box(m, 0, wall_z, GRID-m, h, wall_z+w, wl);
    g.fill_box(wall_x, 0, wall_z+w, wall_x+w, h, GRID-m, wl);

    // ── DOORS with frames ──
    let door_w = 22_usize; let door_h = 56_usize;
    let frame = Voxel::solid(1, 160, 130, 90); // wooden door frame

    // Door 1: living room ↔ bedroom
    let d1 = 40 + rng.range(0, 20) as usize;
    g.fill_box(d1, FLOOR_Y+1, wall_z, d1+door_w, door_h, wall_z+w, Voxel::empty());
    g.fill_box(d1-2, FLOOR_Y+1, wall_z, d1, door_h+2, wall_z+w, frame);
    g.fill_box(d1+door_w, FLOOR_Y+1, wall_z, d1+door_w+2, door_h+2, wall_z+w, frame);
    g.fill_box(d1-2, door_h, wall_z, d1+door_w+2, door_h+2, wall_z+w, frame);

    // Door 2: kitchen ↔ study
    let d2 = 155 + rng.range(0, 15) as usize;
    g.fill_box(d2, FLOOR_Y+1, wall_z, d2+door_w, door_h, wall_z+w, Voxel::empty());
    g.fill_box(d2-2, FLOOR_Y+1, wall_z, d2, door_h+2, wall_z+w, frame);
    g.fill_box(d2+door_w, FLOOR_Y+1, wall_z, d2+door_w+2, door_h+2, wall_z+w, frame);
    g.fill_box(d2-2, door_h, wall_z, d2+door_w+2, door_h+2, wall_z+w, frame);

    // Door 3: study ↔ bedroom
    let d3 = wall_z + w + 35 + rng.range(0, 15) as usize;
    g.fill_box(wall_x, FLOOR_Y+1, d3, wall_x+w, door_h, d3+door_w, Voxel::empty());
    g.fill_box(wall_x, FLOOR_Y+1, d3-2, wall_x+w, door_h+2, d3, frame);
    g.fill_box(wall_x, FLOOR_Y+1, d3+door_w, wall_x+w, door_h+2, d3+door_w+2, frame);
    g.fill_box(wall_x, door_h, d3-2, wall_x+w, door_h+2, d3+door_w+2, frame);

    // Light switches next to doors
    let switch_c = Voxel::solid(6, 240, 235, 225);
    g.fill_box(d1+door_w+4, 38, wall_z-1, d1+door_w+7, 44, wall_z, switch_c);
    g.fill_box(d2-7, 38, wall_z-1, d2-4, 44, wall_z, switch_c);

    // ── ROOM FLOORS ──
    g.fill_box(wall_x+w, FLOOR_Y+1, m+w, GRID-m-w, FLOOR_Y+1, wall_z-1, fl2);       // kitchen
    g.fill_box(wall_x+w, FLOOR_Y+1, wall_z+w, GRID-m-w, FLOOR_Y+1, GRID-m-w, fl3);  // study
    // Baseboard
    g.fill_box(m, FLOOR_Y+1, m+w, GRID-m, FLOOR_Y+4, m+w+1, vary(&mut rng, 198, 190, 180, 8));

    // ═══════════════════════════════════════════════════════════
    // LIVING ROOM — front-left quadrant (m..wall_x, m..wall_z)
    // ═══════════════════════════════════════════════════════════
    let lr_cx = (m + wall_x) / 2;       // center X of living room
    let lr_cz = (m + wall_z) / 2;       // center Z

    // Rug
    place_rug(&mut g, &mut rng, lr_cx - 40, lr_cz - 35, lr_cx + 30, lr_cz + 35);

    // Sofa (against left wall)
    let sofa_x = m + w + 4;
    place_sofa(&mut g, &mut rng, sofa_x, lr_cz);

    // Coffee table (further from sofa — room to walk)
    place_coffee_table(&mut g, &mut rng, sofa_x + 50, lr_cz);

    // TV (against right wall of living room, or center-right)
    let tv_x = wall_x - 16;
    place_tv(&mut g, &mut rng, tv_x, lr_cz);

    // Floor lamp (corner)
    place_floor_lamp(&mut g, &mut rng, m + w + 2, m + w + 8);

    // Cardboard box (random corner)
    if rng.range(0, 1) == 0 {
        place_box(&mut g, &mut rng, lr_cx + 20, m + w + 8);
    }

    // Parrot cage (random presence, 70% chance)
    if rng.range(0, 9) < 7 {
        let pcx = lr_cx + rng.range(-10, 10) as usize;
        let pcz = lr_cz + 35 + rng.range(0, 10) as usize;
        place_parrot_cage(&mut g, &mut rng, pcx, pcz);
    }

    // Plant in living room
    if rng.range(0, 1) == 0 {
        place_plant(&mut g, &mut rng, (tv_x - 10) as f32, FLOOR_Y as f32 + 1.0, (lr_cz + 30) as f32);
    }

    // ═══════════════════════════════════════════════════════════
    // KITCHEN — front-right quadrant (wall_x..GRID-m, m..wall_z)
    // ═══════════════════════════════════════════════════════════
    let kx_cx = (wall_x + w + GRID - m) / 2;
    let kx_cz = (m + w + wall_z) / 2;

    // Kitchen table (center)
    place_kitchen_table(&mut g, &mut rng, kx_cx, kx_cz);

    // 2 chairs (further apart — room for cat)
    place_chair(&mut g, &mut rng, kx_cx, kx_cz - 35);
    place_chair(&mut g, &mut rng, kx_cx, kx_cz + 35);

    // Vase on table
    place_vase(&mut g, &mut rng, kx_cx as f32, 59.0, kx_cz as f32);

    // Counter along back wall
    place_counter(&mut g, &mut rng, wall_x + w + 5, wall_z - 12, GRID - m - w - 2, wall_z - 1);

    // Microwave on counter
    place_microwave(&mut g, &mut rng, wall_x + w + 40, 46, wall_z - 11);

    // Fridge (corner)
    place_fridge(&mut g, &mut rng, GRID - m - w - 18, m + w + 4);

    // Dog bed (in kitchen)
    place_dog_bed(&mut g, &mut rng, wall_x + w + 5, m + w + 4);

    // Plant in kitchen
    if rng.range(0, 1) == 0 {
        place_plant(&mut g, &mut rng, (wall_x + w + 15) as f32, 46.0, (wall_z - 8) as f32);
    }

    // ═══════════════════════════════════════════════════════════
    // BEDROOM — back-left quadrant (m..wall_x, wall_z..GRID-m)
    // ═══════════════════════════════════════════════════════════
    let br_cz = (wall_z + w + GRID - m) / 2;

    // Rug
    place_rug(&mut g, &mut rng, m + w + 15, wall_z + w + 10, wall_x - 15, GRID - m - w - 15);

    // Bed (against back wall)
    place_bed(&mut g, &mut rng, m + w + 10, GRID - m - w - 55, wall_x - 15, GRID - m - w - 2);

    // Nightstand
    place_nightstand(&mut g, &mut rng, wall_x - 30, GRID - m - w - 25);

    // Wardrobe
    place_wardrobe(&mut g, &mut rng, wall_x - 40, wall_z + w + 5, wall_x - 10, wall_z + w + 20);

    // Plant in bedroom
    if rng.range(0, 1) == 0 {
        place_plant(&mut g, &mut rng, (m + w + 8) as f32, FLOOR_Y as f32 + 1.0, br_cz as f32);
    }

    // ═══════════════════════════════════════════════════════════
    // STUDY — back-right quadrant (wall_x..GRID-m, wall_z..GRID-m)
    // ═══════════════════════════════════════════════════════════

    // Desk (against back wall)
    place_desk(&mut g, &mut rng, wall_x + w + 20, GRID - m - w - 16, wall_x + w + 90, GRID - m - w - 2);

    // Bookshelf
    place_bookshelf(&mut g, &mut rng, wall_x + w + 4, wall_z + w + 5, wall_x + w + 24, wall_z + w + 18);

    // Extra bookshelf (50% chance)
    if rng.range(0, 1) == 0 {
        place_bookshelf(&mut g, &mut rng, wall_x + w + 28, wall_z + w + 5, wall_x + w + 48, wall_z + w + 18);
    }

    // Plant in study
    place_plant(&mut g, &mut rng, (GRID - m - w - 12) as f32, FLOOR_Y as f32 + 1.0, (GRID - m - w - 12) as f32);

    g.hollow();
    g
}
