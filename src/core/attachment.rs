// ═══════════════════════════════════════════════════════════════
// PROMETHEUS ENGINE — Attachment System
//
// Objects attached to skeleton bones: weapons, gear, items.
// Each attachment has named points (muzzle, scope, grip).
// World positions computed automatically from skeleton.
// ═══════════════════════════════════════════════════════════════

use glam::{Quat, Vec3};
use super::skeleton::{Skeleton, BoneId};

/// A point on an attached object (muzzle, scope, grip, etc.)
#[derive(Clone, Debug)]
pub struct ObjectPoint {
    pub name: String,
    /// Position relative to attachment origin
    pub local_pos: Vec3,
    /// Direction relative to attachment (e.g., muzzle forward = +Z)
    pub local_dir: Vec3,
}

/// A voxel model that can be attached to a bone
#[derive(Clone, Debug)]
pub struct AttachedObject {
    /// Name of this object
    pub name: String,
    /// Which bone to attach to
    pub bone_id: BoneId,
    /// Offset from bone joint in bone-local space
    pub local_offset: Vec3,
    /// Rotation relative to bone
    pub local_rotation: Quat,
    /// Named points on this object
    pub points: Vec<ObjectPoint>,
    /// Voxel segments: (start, end, radius, material, color)
    pub segments: Vec<ObjectSegment>,

    // --- Computed each frame ---
    /// World position of attachment origin
    pub world_position: Vec3,
    /// World rotation
    pub world_rotation: Quat,
}

/// A segment of the attached object (barrel, stock, magazine, etc.)
#[derive(Clone, Debug)]
pub struct ObjectSegment {
    pub start: Vec3,    // local space
    pub end: Vec3,      // local space
    pub radius: f32,
    pub material: u8,
    pub color: [u8; 3],
}

impl AttachedObject {
    pub fn new(name: &str, bone_id: BoneId, offset: Vec3, rotation: Quat) -> Self {
        Self {
            name: name.to_string(),
            bone_id, local_offset: offset, local_rotation: rotation,
            points: Vec::new(), segments: Vec::new(),
            world_position: Vec3::ZERO, world_rotation: Quat::IDENTITY,
        }
    }

    /// Add a named point
    pub fn add_point(&mut self, name: &str, pos: Vec3, dir: Vec3) -> &mut Self {
        self.points.push(ObjectPoint {
            name: name.to_string(), local_pos: pos, local_dir: dir.normalize(),
        });
        self
    }

    /// Add a visual segment
    pub fn add_segment(&mut self, start: Vec3, end: Vec3, radius: f32, material: u8, color: [u8; 3]) -> &mut Self {
        self.segments.push(ObjectSegment { start, end, radius, material, color });
        self
    }

    /// Update world transform from skeleton
    pub fn update(&mut self, skeleton: &Skeleton) {
        let bone = skeleton.bone_by_id(self.bone_id);
        self.world_rotation = bone.world_rotation * self.local_rotation;
        self.world_position = bone.world_position + bone.world_rotation * self.local_offset;
    }

    /// Get world position of a named point
    pub fn get_point_world(&self, name: &str) -> Option<(Vec3, Vec3)> {
        for p in &self.points {
            if p.name == name {
                let pos = self.world_position + self.world_rotation * p.local_pos;
                let dir = self.world_rotation * p.local_dir;
                return Some((pos, dir));
            }
        }
        None
    }

    /// Get muzzle position and direction (convenience)
    pub fn muzzle(&self) -> Option<(Vec3, Vec3)> {
        self.get_point_world("muzzle")
    }

    /// Rasterize this object into a voxel grid
    pub fn rasterize<F>(&self, grid_size: usize, mut set_voxel: F)
    where F: FnMut(usize, usize, usize, u8, u8, u8, u8) {
        for seg in &self.segments {
            let world_start = self.world_position + self.world_rotation * seg.start;
            let world_end = self.world_position + self.world_rotation * seg.end;
            let dir = world_end - world_start;
            let len = dir.length();
            let steps = (len * 2.0).max(2.0) as usize;

            for i in 0..=steps {
                let t = i as f32 / steps as f32;
                let center = world_start + dir * t;
                let r = seg.radius;
                let ri = r.ceil() as i32;

                for dx in -ri..=ri {
                    for dy in -ri..=ri {
                        for dz in -ri..=ri {
                            if (dx*dx + dy*dy + dz*dz) as f32 <= r*r {
                                let x = (center.x as i32 + dx) as usize;
                                let y = (center.y as i32 + dy) as usize;
                                let z = (center.z as i32 + dz) as usize;
                                if x < grid_size && y < grid_size && z < grid_size {
                                    set_voxel(x, y, z, seg.material,
                                        seg.color[0], seg.color[1], seg.color[2]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// PREFAB WEAPONS
// ═══════════════════════════════════════════════════════════════

/// Create an AK-style assault rifle
pub fn weapon_ak(bone_id: BoneId, scale: f32) -> AttachedObject {
    let s = scale;
    let gun_metal: [u8; 3] = [55, 55, 62];
    let gun_dark: [u8; 3] = [35, 35, 42];
    let wood: [u8; 3] = [90, 60, 30];

    let mut w = AttachedObject::new("AK-12", bone_id,
        Vec3::new(0.0, -2.0*s, 0.0),  // offset from hand
        Quat::IDENTITY,
    );

    // Receiver (main body)
    w.add_segment(Vec3::new(0.0, 0.0, -8.0*s), Vec3::new(0.0, 0.0, 12.0*s), 2.0*s, 4, gun_metal);
    // Barrel
    w.add_segment(Vec3::new(0.0, 0.0, 12.0*s), Vec3::new(0.0, 0.0, 28.0*s), 1.0*s, 4, gun_dark);
    // Magazine
    w.add_segment(Vec3::new(0.0, -1.0*s, 2.0*s), Vec3::new(1.0*s, -9.0*s, 3.0*s), 1.5*s, 4, gun_dark);
    // Stock
    w.add_segment(Vec3::new(0.0, 1.0*s, -8.0*s), Vec3::new(0.0, 3.0*s, -18.0*s), 1.8*s, 1, wood);
    // Grip
    w.add_segment(Vec3::new(0.0, 0.0, -2.0*s), Vec3::new(0.0, -5.0*s, -3.0*s), 1.2*s, 4, gun_dark);

    // Points
    w.add_point("muzzle", Vec3::new(0.0, 0.0, 28.0*s), Vec3::Z);
    w.add_point("scope", Vec3::new(0.0, 2.0*s, 6.0*s), Vec3::Z);
    w.add_point("grip", Vec3::ZERO, Vec3::NEG_Y);
    w.add_point("support_hand", Vec3::new(0.0, 0.0, 10.0*s), Vec3::NEG_Y);
    w.add_point("stock", Vec3::new(0.0, 2.0*s, -16.0*s), Vec3::NEG_Z);

    w
}

/// Create a submachine gun (Vikhr/Vortex style)
pub fn weapon_vikhr(bone_id: BoneId, scale: f32) -> AttachedObject {
    let s = scale;
    let metal: [u8; 3] = [60, 60, 68];
    let dark: [u8; 3] = [40, 40, 48];

    let mut w = AttachedObject::new("Vikhr", bone_id,
        Vec3::new(0.0, -2.0*s, 0.0),
        Quat::IDENTITY,
    );

    // Compact receiver
    w.add_segment(Vec3::new(0.0, 0.0, -5.0*s), Vec3::new(0.0, 0.0, 8.0*s), 1.8*s, 4, metal);
    // Short barrel
    w.add_segment(Vec3::new(0.0, 0.0, 8.0*s), Vec3::new(0.0, 0.0, 18.0*s), 0.8*s, 4, dark);
    // Magazine
    w.add_segment(Vec3::new(0.0, -1.0*s, 1.0*s), Vec3::new(0.0, -7.0*s, 2.0*s), 1.2*s, 4, dark);
    // Folding stock
    w.add_segment(Vec3::new(0.0, 0.5*s, -5.0*s), Vec3::new(0.0, 2.0*s, -12.0*s), 1.5*s, 4, metal);

    w.add_point("muzzle", Vec3::new(0.0, 0.0, 18.0*s), Vec3::Z);
    w.add_point("grip", Vec3::ZERO, Vec3::NEG_Y);
    w.add_point("support_hand", Vec3::new(0.0, 0.0, 6.0*s), Vec3::NEG_Y);
    w.add_point("stock", Vec3::new(0.0, 1.5*s, -10.0*s), Vec3::NEG_Z);

    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::skeleton::Skeleton;

    #[test]
    fn test_weapon_creation() {
        let sk = Skeleton::human(1.0);
        let hand_id = sk.bone("hand_r").id;
        let ak = weapon_ak(hand_id, 1.0);

        assert_eq!(ak.name, "AK-12");
        assert!(ak.segments.len() >= 4); // receiver, barrel, magazine, stock, grip
        assert!(ak.points.len() >= 4);   // muzzle, scope, grip, support, stock
    }

    #[test]
    fn test_weapon_world_position() {
        let mut sk = Skeleton::human(2.0);
        sk.root_position = Vec3::new(128.0, 92.0, 128.0);
        sk.solve_forward();

        let hand_id = sk.bone("hand_r").id;
        let mut ak = weapon_ak(hand_id, 2.0);
        ak.update(&sk);

        // Muzzle should exist and be in front of the hand
        let muzzle = ak.muzzle();
        assert!(muzzle.is_some());
        let (pos, dir) = muzzle.unwrap();
        println!("Muzzle pos: {:?}, dir: {:?}", pos, dir);
        // Muzzle should be somewhere in the grid
        assert!(pos.x > 0.0 && pos.y > 0.0);
    }

    #[test]
    fn test_weapon_rasterize() {
        let mut sk = Skeleton::human(1.0);
        sk.root_position = Vec3::new(32.0, 20.0, 32.0);
        sk.solve_forward();

        let hand_id = sk.bone("hand_r").id;
        let mut ak = weapon_ak(hand_id, 1.0);
        ak.update(&sk);

        let mut count = 0;
        ak.rasterize(64, |_,_,_,_,_,_,_| count += 1);
        println!("Weapon voxels: {}", count);
        assert!(count > 100);
    }
}
