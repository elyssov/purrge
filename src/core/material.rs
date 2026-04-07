// ═══════════════════════════════════════════════════════════════
// PROMETHEUS ENGINE — Material Registry
//
// Physical properties of materials. Each voxel has a material_id (u8).
// The registry maps IDs to properties: hardness, density, brittleness,
// sound, particles, texture mode.
//
// 14 base materials from PURRGE design doc + extensible.
// ═══════════════════════════════════════════════════════════════

/// Physical and visual properties of a material
#[derive(Clone, Debug)]
pub struct MaterialDef {
    pub id: u8,
    pub name: &'static str,
    /// Resistance to destruction (0.0 = paper, 1.0 = diamond)
    pub hardness: f32,
    /// Mass per cubic meter (kg/m³)
    pub density: f32,
    /// How it breaks: 0.0 = bends/deforms, 1.0 = shatters into pieces
    pub brittleness: f32,
    /// Sound level when destroyed (0.0 = silent, 1.0 = loud crash)
    pub noise: f32,
    /// Friction coefficient (0.0 = ice, 1.0 = rubber)
    pub friction: f32,
    /// Is this material flammable?
    pub flammable: bool,
    /// Is this material transparent? (glass, water)
    pub transparent: bool,
    /// Default color [R, G, B]
    pub color: [u8; 3],
    /// Color of particles/debris when broken
    pub particle_color: [u8; 3],
    /// How textures are applied
    pub texture_mode: TextureMode,
}

/// How to texture voxels of this material
#[derive(Clone, Debug, PartialEq)]
pub enum TextureMode {
    /// Single solid color (from voxel data)
    Solid,
    /// GPU triplanar mapping from texture atlas (good for repeating: wood, brick)
    TriplanarGPU,
    /// CPU per-voxel coloring from texture (good for organic: fur, skin)
    PerVoxelCPU,
}

/// The material registry — lookup by material_id
pub struct MaterialRegistry {
    materials: Vec<MaterialDef>,
}

impl MaterialRegistry {
    /// Create registry with default 14 materials from PURRGE design doc
    pub fn default() -> Self {
        let mut reg = Self { materials: Vec::with_capacity(16) };

        // ID 0: Empty (air)
        reg.register(MaterialDef {
            id: 0, name: "Air", hardness: 0.0, density: 0.0, brittleness: 0.0,
            noise: 0.0, friction: 0.0, flammable: false, transparent: true,
            color: [0,0,0], particle_color: [0,0,0], texture_mode: TextureMode::Solid,
        });

        // ID 1: Wood
        reg.register(MaterialDef {
            id: 1, name: "Wood", hardness: 0.5, density: 600.0, brittleness: 0.3,
            noise: 0.5, friction: 0.6, flammable: true, transparent: false,
            color: [140,95,50], particle_color: [120,80,40],
            texture_mode: TextureMode::TriplanarGPU,
        });

        // ID 2: Glass
        reg.register(MaterialDef {
            id: 2, name: "Glass", hardness: 0.7, density: 2500.0, brittleness: 1.0,
            noise: 0.9, friction: 0.3, flammable: false, transparent: true,
            color: [200,220,240], particle_color: [220,235,250],
            texture_mode: TextureMode::Solid,
        });

        // ID 3: Ceramic
        reg.register(MaterialDef {
            id: 3, name: "Ceramic", hardness: 0.6, density: 2300.0, brittleness: 0.9,
            noise: 0.8, friction: 0.4, flammable: false, transparent: false,
            color: [230,225,215], particle_color: [210,205,195],
            texture_mode: TextureMode::Solid,
        });

        // ID 4: Metal
        reg.register(MaterialDef {
            id: 4, name: "Metal", hardness: 0.9, density: 7800.0, brittleness: 0.1,
            noise: 0.7, friction: 0.5, flammable: false, transparent: false,
            color: [160,160,170], particle_color: [180,180,190],
            texture_mode: TextureMode::Solid,
        });

        // ID 5: Fabric
        reg.register(MaterialDef {
            id: 5, name: "Fabric", hardness: 0.1, density: 200.0, brittleness: 0.0,
            noise: 0.2, friction: 0.8, flammable: true, transparent: false,
            color: [140,130,120], particle_color: [160,150,140],
            texture_mode: TextureMode::TriplanarGPU,
        });

        // ID 6: Plastic
        reg.register(MaterialDef {
            id: 6, name: "Plastic", hardness: 0.4, density: 1200.0, brittleness: 0.5,
            noise: 0.4, friction: 0.4, flammable: true, transparent: false,
            color: [180,180,185], particle_color: [160,160,165],
            texture_mode: TextureMode::Solid,
        });

        // ID 7: Stone
        reg.register(MaterialDef {
            id: 7, name: "Stone", hardness: 0.8, density: 2600.0, brittleness: 0.6,
            noise: 0.6, friction: 0.7, flammable: false, transparent: false,
            color: [170,165,160], particle_color: [150,145,140],
            texture_mode: TextureMode::TriplanarGPU,
        });

        // ID 8: Water
        reg.register(MaterialDef {
            id: 8, name: "Water", hardness: 0.0, density: 1000.0, brittleness: 0.0,
            noise: 0.3, friction: 0.1, flammable: false, transparent: true,
            color: [100,150,220], particle_color: [120,170,240],
            texture_mode: TextureMode::Solid,
        });

        // ID 9: Paper
        reg.register(MaterialDef {
            id: 9, name: "Paper", hardness: 0.05, density: 100.0, brittleness: 0.0,
            noise: 0.1, friction: 0.5, flammable: true, transparent: false,
            color: [240,235,220], particle_color: [230,225,210],
            texture_mode: TextureMode::Solid,
        });

        // ID 10: Leather/Skin
        reg.register(MaterialDef {
            id: 10, name: "Skin", hardness: 0.3, density: 900.0, brittleness: 0.1,
            noise: 0.2, friction: 0.7, flammable: false, transparent: false,
            color: [220,185,155], particle_color: [200,165,135],
            texture_mode: TextureMode::PerVoxelCPU,
        });

        // ID 11: Earth/Dirt
        reg.register(MaterialDef {
            id: 11, name: "Earth", hardness: 0.2, density: 1500.0, brittleness: 0.2,
            noise: 0.3, friction: 0.8, flammable: false, transparent: false,
            color: [120,95,60], particle_color: [100,80,50],
            texture_mode: TextureMode::TriplanarGPU,
        });

        // ID 12: Food
        reg.register(MaterialDef {
            id: 12, name: "Food", hardness: 0.1, density: 800.0, brittleness: 0.3,
            noise: 0.2, friction: 0.5, flammable: false, transparent: false,
            color: [200,160,80], particle_color: [180,140,60],
            texture_mode: TextureMode::Solid,
        });

        // ID 13: Fur/Wool
        reg.register(MaterialDef {
            id: 13, name: "Fur", hardness: 0.05, density: 300.0, brittleness: 0.0,
            noise: 0.1, friction: 0.9, flammable: true, transparent: false,
            color: [200,140,60], particle_color: [180,120,40],
            texture_mode: TextureMode::PerVoxelCPU,
        });

        // ID 14: Bone (for ORPP monsters / anatomy)
        reg.register(MaterialDef {
            id: 14, name: "Bone", hardness: 0.7, density: 1900.0, brittleness: 0.4,
            noise: 0.5, friction: 0.4, flammable: false, transparent: false,
            color: [230,225,210], particle_color: [210,205,190],
            texture_mode: TextureMode::Solid,
        });

        // ID 15: Ritual Stone (for ORPP paranormal)
        reg.register(MaterialDef {
            id: 15, name: "Ritual Stone", hardness: 0.95, density: 3000.0, brittleness: 0.3,
            noise: 0.4, friction: 0.6, flammable: false, transparent: false,
            color: [80,60,90], particle_color: [100,80,120],
            texture_mode: TextureMode::TriplanarGPU,
        });

        reg
    }

    /// Register a new material
    pub fn register(&mut self, mat: MaterialDef) {
        let id = mat.id as usize;
        while self.materials.len() <= id {
            self.materials.push(MaterialDef {
                id: self.materials.len() as u8, name: "Unknown",
                hardness: 0.5, density: 1000.0, brittleness: 0.5,
                noise: 0.5, friction: 0.5, flammable: false, transparent: false,
                color: [128,128,128], particle_color: [128,128,128],
                texture_mode: TextureMode::Solid,
            });
        }
        self.materials[id] = mat;
    }

    /// Get material by ID
    pub fn get(&self, id: u8) -> &MaterialDef {
        if (id as usize) < self.materials.len() {
            &self.materials[id as usize]
        } else {
            &self.materials[0] // fallback to air
        }
    }

    /// Get destruction properties for a hit
    pub fn hit_result(&self, id: u8, force: f32) -> HitResult {
        let mat = self.get(id);
        let destroyed = force > mat.hardness;
        let fragments = if destroyed && mat.brittleness > 0.5 {
            ((mat.brittleness * 10.0) as u32).max(2)
        } else {
            0
        };
        HitResult {
            destroyed,
            fragments,
            noise: if destroyed { mat.noise } else { mat.noise * 0.3 },
            particle_color: mat.particle_color,
        }
    }

    /// Number of registered materials
    pub fn count(&self) -> usize { self.materials.len() }
}

/// Result of hitting a material
#[derive(Debug)]
pub struct HitResult {
    /// Was the voxel destroyed?
    pub destroyed: bool,
    /// Number of fragment particles to spawn
    pub fragments: u32,
    /// Noise level (0.0-1.0) for annoyance/alerting
    pub noise: f32,
    /// Color of debris particles
    pub particle_color: [u8; 3],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_registry() {
        let reg = MaterialRegistry::default();
        assert!(reg.count() >= 14);
        assert_eq!(reg.get(0).name, "Air");
        assert_eq!(reg.get(1).name, "Wood");
        assert_eq!(reg.get(2).name, "Glass");
        assert_eq!(reg.get(13).name, "Fur");
    }

    #[test]
    fn test_hit_glass() {
        let reg = MaterialRegistry::default();
        let result = reg.hit_result(2, 1.0); // glass, strong hit
        assert!(result.destroyed);
        assert!(result.fragments >= 2); // glass shatters
        assert!(result.noise > 0.5);    // glass is loud
    }

    #[test]
    fn test_hit_metal_weak() {
        let reg = MaterialRegistry::default();
        let result = reg.hit_result(4, 0.3); // metal, weak hit
        assert!(!result.destroyed); // metal survives weak hit
    }

    #[test]
    fn test_fur_texture_mode() {
        let reg = MaterialRegistry::default();
        assert_eq!(reg.get(13).texture_mode, TextureMode::PerVoxelCPU);
    }

    #[test]
    fn test_wood_properties() {
        let reg = MaterialRegistry::default();
        let wood = reg.get(1);
        assert!(wood.flammable);
        assert_eq!(wood.texture_mode, TextureMode::TriplanarGPU);
        assert!(wood.hardness > 0.3 && wood.hardness < 0.7);
    }
}
