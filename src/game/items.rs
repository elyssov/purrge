// ═══════════════════════════════════════════════════════════════
// PURRGE — Item & Material System
//
// Each piece of furniture has: material, HP, value, noise.
// Material defines how it breaks, sounds, and what particles fly.
// ═══════════════════════════════════════════════════════════════

/// Material physical properties (from design doc — 14 materials)
#[derive(Clone, Debug)]
pub struct Material {
    pub id: u8,
    pub name: &'static str,
    pub hardness: f32,      // 0.0-1.0, how many hits to break
    pub density: f32,       // kg/m³, affects debris weight
    pub brittleness: f32,   // 0.0=bends, 1.0=shatters
    pub noise: f32,         // 0.0-1.0, how loud when broken
    pub color: [u8; 3],     // default color
}

/// All 14 materials from the design doc
pub fn materials() -> Vec<Material> {
    vec![
        Material { id: 1,  name: "Wood",    hardness: 0.5, density: 600.0,  brittleness: 0.3, noise: 0.4, color: [140, 90, 45] },
        Material { id: 2,  name: "Glass",   hardness: 0.7, density: 2500.0, brittleness: 1.0, noise: 0.9, color: [180, 210, 230] },
        Material { id: 3,  name: "Ceramic", hardness: 0.6, density: 2300.0, brittleness: 0.9, noise: 0.8, color: [220, 200, 180] },
        Material { id: 4,  name: "Metal",   hardness: 0.9, density: 7800.0, brittleness: 0.1, noise: 0.7, color: [150, 150, 155] },
        Material { id: 5,  name: "Fabric",  hardness: 0.1, density: 200.0,  brittleness: 0.0, noise: 0.05, color: [120, 100, 80] },
        Material { id: 6,  name: "Plastic", hardness: 0.4, density: 1200.0, brittleness: 0.5, noise: 0.3, color: [180, 180, 170] },
        Material { id: 7,  name: "Stone",   hardness: 0.8, density: 2600.0, brittleness: 0.6, noise: 0.6, color: [160, 155, 150] },
        Material { id: 8,  name: "Water",   hardness: 0.0, density: 1000.0, brittleness: 0.0, noise: 0.2, color: [80, 140, 200] },
        Material { id: 9,  name: "Paper",   hardness: 0.05,density: 100.0,  brittleness: 0.0, noise: 0.02, color: [240, 235, 220] },
        Material { id: 10, name: "Leather", hardness: 0.3, density: 900.0,  brittleness: 0.1, noise: 0.15, color: [120, 70, 40] },
        Material { id: 11, name: "Earth",   hardness: 0.2, density: 1500.0, brittleness: 0.2, noise: 0.1, color: [100, 80, 50] },
        Material { id: 12, name: "Food",    hardness: 0.1, density: 800.0,  brittleness: 0.3, noise: 0.1, color: [200, 170, 100] },
        Material { id: 13, name: "Fur",     hardness: 0.05,density: 300.0,  brittleness: 0.0, noise: 0.01, color: [200, 140, 60] },
        Material { id: 14, name: "Rubber",  hardness: 0.2, density: 1100.0, brittleness: 0.0, noise: 0.1, color: [60, 60, 60] },
    ]
}

/// Get material by ID (returns Wood as fallback)
pub fn material_by_id(id: u8) -> Material {
    materials().into_iter().find(|m| m.id == id)
        .unwrap_or(Material { id: 0, name: "Unknown", hardness: 0.5, density: 500.0, brittleness: 0.5, noise: 0.3, color: [128,128,128] })
}

/// A destructible furniture item in the apartment
#[derive(Clone, Debug)]
pub struct FurnitureItem {
    pub name: String,
    pub x: f32, pub y: f32, pub z: f32,  // world position (center)
    pub width: f32, pub height: f32, pub depth: f32,  // size
    pub material_id: u8,
    pub value: f32,         // monetary value ($) for repair bill
    pub hp: f32,            // hit points (derived from material hardness × size)
    pub max_hp: f32,
    pub destroyed: bool,
}

impl FurnitureItem {
    pub fn new(name: &str, x: f32, y: f32, z: f32, w: f32, h: f32, d: f32, material_id: u8, value: f32) -> Self {
        let mat = material_by_id(material_id);
        let volume = w * h * d;
        let hp = mat.hardness * volume.cbrt() * 10.0; // bigger + harder = more HP
        Self {
            name: name.to_string(),
            x, y, z, width: w, height: h, depth: d,
            material_id, value,
            hp, max_hp: hp, destroyed: false,
        }
    }

    /// Apply damage. Returns true if destroyed this hit.
    pub fn damage(&mut self, amount: f32) -> bool {
        if self.destroyed { return false; }
        self.hp -= amount;
        if self.hp <= 0.0 {
            self.destroyed = true;
            return true;
        }
        false
    }

    /// Damage fraction (for visual: cracks, color change)
    pub fn damage_frac(&self) -> f32 {
        1.0 - (self.hp / self.max_hp).max(0.0)
    }

    /// Does a point fall within this item's bounding box?
    pub fn contains(&self, px: f32, py: f32, pz: f32) -> bool {
        let hw = self.width / 2.0;
        let hd = self.depth / 2.0;
        px >= self.x - hw && px <= self.x + hw &&
        py >= self.y && py <= self.y + self.height &&
        pz >= self.z - hd && pz <= self.z + hd
    }

    /// Get the noise this item makes when destroyed
    pub fn destruction_noise(&self) -> f32 {
        material_by_id(self.material_id).noise
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_materials_count() {
        assert_eq!(materials().len(), 14);
    }

    #[test]
    fn test_glass_is_brittle() {
        let glass = material_by_id(2);
        assert_eq!(glass.name, "Glass");
        assert_eq!(glass.brittleness, 1.0);
    }

    #[test]
    fn test_furniture_damage() {
        let mut vase = FurnitureItem::new("Vase", 100.0, 50.0, 100.0, 6.0, 10.0, 6.0, 3, 200.0);
        assert!(!vase.destroyed);
        // Ceramic vase: hardness 0.6, small → low HP
        let destroyed = vase.damage(vase.max_hp + 1.0);
        assert!(destroyed);
        assert!(vase.destroyed);
    }

    #[test]
    fn test_metal_is_tough() {
        let fridge = FurnitureItem::new("Fridge", 50.0, 0.0, 50.0, 30.0, 60.0, 30.0, 4, 1500.0);
        // Metal (hardness 0.9) + large → lots of HP
        assert!(fridge.max_hp > 20.0);
    }

    #[test]
    fn test_contains() {
        let table = FurnitureItem::new("Table", 100.0, 30.0, 100.0, 40.0, 2.0, 30.0, 1, 300.0);
        assert!(table.contains(100.0, 31.0, 100.0)); // center
        assert!(!table.contains(200.0, 31.0, 100.0)); // outside
    }

    #[test]
    fn test_fabric_is_quiet() {
        let sofa = FurnitureItem::new("Sofa", 50.0, 0.0, 50.0, 60.0, 30.0, 25.0, 5, 800.0);
        assert!(sofa.destruction_noise() < 0.1);
    }
}
