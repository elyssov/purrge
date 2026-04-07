// ═══════════════════════════════════════════════════════════════
// PROMETHEUS ENGINE — Procedural Generation
//
// Parameterized world generation: rooms, buildings, cities, nature.
// Describe WHAT you want, engine decides HOW.
// Seed-based: same seed = same result. Reproducible.
// ═══════════════════════════════════════════════════════════════

use glam::Vec3;

/// Seed for reproducible generation
pub type Seed = u64;

/// Simple deterministic RNG from seed
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: Seed) -> Self { Self { state: seed.wrapping_add(1) } }

    pub fn next(&mut self) -> u64 {
        // xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Random float 0.0 - 1.0
    pub fn f32(&mut self) -> f32 { (self.next() % 10000) as f32 / 10000.0 }

    /// Random int in range [min, max]
    pub fn range(&mut self, min: i32, max: i32) -> i32 {
        min + (self.next() % (max - min + 1) as u64) as i32
    }

    /// Random float in range
    pub fn frange(&mut self, min: f32, max: f32) -> f32 { min + self.f32() * (max - min) }

    /// Random bool with probability
    pub fn chance(&mut self, probability: f32) -> bool { self.f32() < probability }

    /// Pick random element from slice
    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[(self.next() % items.len() as u64) as usize]
    }
}

// ─── Room Types ──────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum RoomType {
    LivingRoom,
    Kitchen,
    Bedroom,
    Bathroom,
    Hallway,
    Office,
}

#[derive(Clone, Debug)]
pub struct FurnitureItem {
    pub name: String,
    pub position: Vec3,       // in room-local coords
    pub size: Vec3,           // width, height, depth
    pub material: u8,
    pub color: [u8; 3],
    pub value: f32,           // monetary value (for PURRGE scoring)
    pub destructible: bool,
}

#[derive(Clone, Debug)]
pub struct RoomSpec {
    pub room_type: RoomType,
    pub width: f32,
    pub depth: f32,
    pub height: f32,
    pub wall_thickness: f32,
    pub has_window: bool,
    pub has_door: bool,
    pub door_wall: u8,       // 0=north, 1=east, 2=south, 3=west
    pub furniture: Vec<FurnitureItem>,
}

/// Generate a room specification from parameters
pub fn generate_room(room_type: RoomType, size: f32, clutter: f32, seed: Seed) -> RoomSpec {
    let mut rng = Rng::new(seed);
    let s = size;

    let width = s * rng.frange(0.8, 1.2);
    let depth = s * rng.frange(0.8, 1.2);
    let height = s * 0.5 * rng.frange(0.9, 1.1);

    let mut furniture = Vec::new();

    match room_type {
        RoomType::LivingRoom => {
            // Sofa (always)
            furniture.push(FurnitureItem {
                name: "Sofa".into(),
                position: Vec3::new(width * 0.5, 0.0, depth * 0.8),
                size: Vec3::new(width * 0.4, height * 0.15, depth * 0.12),
                material: 5, color: [120, 100, 80], value: 800.0, destructible: true,
            });
            // TV (always, opposite sofa)
            furniture.push(FurnitureItem {
                name: "TV".into(),
                position: Vec3::new(width * 0.5, height * 0.2, depth * 0.1),
                size: Vec3::new(width * 0.25, height * 0.12, depth * 0.02),
                material: 6, color: [30, 30, 35], value: 1200.0, destructible: true,
            });
            // Coffee table
            if rng.chance(0.8) {
                furniture.push(FurnitureItem {
                    name: "Coffee Table".into(),
                    position: Vec3::new(width * 0.5, 0.0, depth * 0.55),
                    size: Vec3::new(width * 0.2, height * 0.08, depth * 0.1),
                    material: 1, color: [110, 70, 35], value: 300.0, destructible: true,
                });
            }
            // Bookshelf
            if rng.chance(0.6) {
                let wall = rng.range(0, 1) as f32; // left or right
                furniture.push(FurnitureItem {
                    name: "Bookshelf".into(),
                    position: Vec3::new(width * wall, 0.0, depth * rng.frange(0.3, 0.7)),
                    size: Vec3::new(width * 0.06, height * 0.7, depth * 0.15),
                    material: 1, color: [95, 60, 28], value: 400.0, destructible: true,
                });
            }
            // Lamp
            if rng.chance(0.7) {
                furniture.push(FurnitureItem {
                    name: "Floor Lamp".into(),
                    position: Vec3::new(width * rng.frange(0.1, 0.9), 0.0, depth * rng.frange(0.1, 0.4)),
                    size: Vec3::new(width * 0.03, height * 0.5, depth * 0.03),
                    material: 4, color: [150, 150, 155], value: 150.0, destructible: true,
                });
            }
            // Random clutter
            let clutter_count = (clutter * 5.0) as usize;
            for _ in 0..clutter_count {
                let item_type = rng.range(0, 3);
                let (name, sz, mat, col, val) = match item_type {
                    0 => ("Vase", Vec3::new(0.04, 0.08, 0.04), 3u8, [180u8,60,50], 200.0),
                    1 => ("Book", Vec3::new(0.06, 0.02, 0.04), 9, [60,80,140], 30.0),
                    2 => ("Cup", Vec3::new(0.03, 0.04, 0.03), 3, [240,240,235], 15.0),
                    _ => ("Photo Frame", Vec3::new(0.06, 0.08, 0.01), 1, [100,80,50], 50.0),
                };
                furniture.push(FurnitureItem {
                    name: name.into(),
                    position: Vec3::new(
                        width * rng.frange(0.15, 0.85),
                        height * rng.frange(0.08, 0.3), // on surfaces
                        depth * rng.frange(0.15, 0.85),
                    ),
                    size: Vec3::new(sz.x * s, sz.y * s, sz.z * s),
                    material: mat, color: col, value: val, destructible: true,
                });
            }
        }
        RoomType::Kitchen => {
            // Counter (along back wall)
            furniture.push(FurnitureItem {
                name: "Kitchen Counter".into(),
                position: Vec3::new(width * 0.5, 0.0, depth * 0.9),
                size: Vec3::new(width * 0.7, height * 0.15, depth * 0.08),
                material: 7, color: [180, 175, 170], value: 2000.0, destructible: false,
            });
            // Fridge
            furniture.push(FurnitureItem {
                name: "Refrigerator".into(),
                position: Vec3::new(width * 0.9, 0.0, depth * 0.85),
                size: Vec3::new(width * 0.08, height * 0.35, depth * 0.08),
                material: 4, color: [210, 210, 215], value: 1500.0, destructible: true,
            });
            // Table
            furniture.push(FurnitureItem {
                name: "Kitchen Table".into(),
                position: Vec3::new(width * 0.4, 0.0, depth * 0.4),
                size: Vec3::new(width * 0.2, height * 0.13, depth * 0.15),
                material: 1, color: [130, 85, 45], value: 500.0, destructible: true,
            });
        }
        RoomType::Bedroom => {
            // Bed
            furniture.push(FurnitureItem {
                name: "Bed".into(),
                position: Vec3::new(width * 0.3, 0.0, depth * 0.7),
                size: Vec3::new(width * 0.35, height * 0.1, depth * 0.25),
                material: 5, color: [200, 200, 210], value: 1000.0, destructible: false,
            });
            // Nightstand
            furniture.push(FurnitureItem {
                name: "Nightstand".into(),
                position: Vec3::new(width * 0.1, 0.0, depth * 0.7),
                size: Vec3::new(width * 0.06, height * 0.1, depth * 0.06),
                material: 1, color: [100, 65, 30], value: 200.0, destructible: true,
            });
            // Wardrobe
            furniture.push(FurnitureItem {
                name: "Wardrobe".into(),
                position: Vec3::new(width * 0.85, 0.0, depth * 0.5),
                size: Vec3::new(width * 0.1, height * 0.6, depth * 0.2),
                material: 1, color: [90, 55, 25], value: 600.0, destructible: true,
            });
        }
        _ => {}
    }

    RoomSpec {
        room_type, width, depth, height,
        wall_thickness: s * 0.03,
        has_window: rng.chance(0.7),
        has_door: true,
        door_wall: rng.range(0, 3) as u8,
        furniture,
    }
}

/// Rasterize a room spec into a voxel grid via callback
pub fn rasterize_room<F>(spec: &RoomSpec, grid_size: usize, offset: Vec3, scale: f32, mut set_voxel: F)
where F: FnMut(usize, usize, usize, u8, u8, u8, u8) {
    let s = scale;
    let w = (spec.width * s) as usize;
    let d = (spec.depth * s) as usize;
    let h = (spec.height * s) as usize;
    let wt = (spec.wall_thickness * s).max(2.0) as usize;
    let ox = offset.x as usize;
    let oy = offset.y as usize;
    let oz = offset.z as usize;

    // Floor
    for x in ox..ox+w { for z in oz..oz+d {
        for y in oy..oy+wt {
            if x<grid_size && y<grid_size && z<grid_size {
                set_voxel(x, y, z, 7, 180, 155, 115); // wood
            }
        }
    }}

    // Ceiling
    for x in ox..ox+w { for z in oz..oz+d {
        for y in oy+h-wt..oy+h {
            if x<grid_size && y<grid_size && z<grid_size {
                set_voxel(x, y, z, 7, 235, 235, 240); // white
            }
        }
    }}

    // Walls (3 sides — front open for dollhouse view)
    // Back wall (high Z)
    for x in ox..ox+w { for y in oy..oy+h {
        for z in oz+d-wt..oz+d {
            if x<grid_size && y<grid_size && z<grid_size {
                set_voxel(x, y, z, 7, 220, 215, 205);
            }
        }
    }}
    // Left wall
    for z in oz..oz+d { for y in oy..oy+h {
        for x in ox..ox+wt {
            if x<grid_size && y<grid_size && z<grid_size {
                set_voxel(x, y, z, 7, 220, 215, 205);
            }
        }
    }}
    // Right wall
    for z in oz..oz+d { for y in oy..oy+h {
        for x in ox+w-wt..ox+w {
            if x<grid_size && y<grid_size && z<grid_size {
                set_voxel(x, y, z, 7, 220, 215, 205);
            }
        }
    }}

    // Window (if front wall existed — skip for dollhouse)

    // Furniture
    for item in &spec.furniture {
        let ix = ox + (item.position.x * s) as usize;
        let iy = oy + wt + (item.position.y * s) as usize;
        let iz = oz + (item.position.z * s) as usize;
        let iw = (item.size.x * s).max(1.0) as usize;
        let ih = (item.size.y * s).max(1.0) as usize;
        let id = (item.size.z * s).max(1.0) as usize;

        for x in ix..ix+iw { for y in iy..iy+ih { for z in iz..iz+id {
            if x<grid_size && y<grid_size && z<grid_size {
                set_voxel(x, y, z, item.material, item.color[0], item.color[1], item.color[2]);
            }
        }}}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_deterministic() {
        let mut rng1 = Rng::new(42);
        let mut rng2 = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next(), rng2.next());
        }
    }

    #[test]
    fn test_generate_living_room() {
        let spec = generate_room(RoomType::LivingRoom, 50.0, 0.5, 12345);
        assert!(spec.furniture.len() >= 2); // at least sofa + TV
        println!("Living room: {} items", spec.furniture.len());
        for f in &spec.furniture {
            println!("  {} at {:?} val=${:.0}", f.name, f.position, f.value);
        }
    }

    #[test]
    fn test_same_seed_same_room() {
        let spec1 = generate_room(RoomType::Kitchen, 40.0, 0.3, 99999);
        let spec2 = generate_room(RoomType::Kitchen, 40.0, 0.3, 99999);
        assert_eq!(spec1.furniture.len(), spec2.furniture.len());
    }

    #[test]
    fn test_rasterize_produces_voxels() {
        let spec = generate_room(RoomType::Bedroom, 60.0, 0.5, 777);
        let mut count = 0;
        rasterize_room(&spec, 128, Vec3::new(10.0, 0.0, 10.0), 1.0, |_,_,_,_,_,_,_| count += 1);
        println!("Bedroom voxels: {}", count);
        assert!(count > 5000);
    }
}
