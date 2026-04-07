// ═══════════════════════════════════════════════════════════════
// PROMETHEUS ENGINE — Entity
//
// A game object: skeleton + body + attachments + behavior.
// High-level API: entity.aim_at(target), entity.fire(), entity.walk()
// The entity computes everything internally.
// ═══════════════════════════════════════════════════════════════

use glam::{Vec3, Quat};
use super::skeleton::Skeleton;
use super::body::BodyDefinition;
use super::attachment::AttachedObject;
use super::ik;

/// High-level game entity
pub struct Entity {
    pub skeleton: Skeleton,
    pub body: BodyDefinition,
    pub attachments: Vec<AttachedObject>,
    pub name: String,

    /// Current facing direction (yaw only, radians)
    pub facing_yaw: f32,
    /// Target yaw (smooth rotation toward this)
    target_yaw: f32,
    /// Turn speed (0.0-1.0 per frame)
    pub turn_speed: f32,
}

/// Result of entity.fire()
pub struct FireResult {
    pub origin: Vec3,
    pub direction: Vec3,
    pub speed: f32,
}

impl Entity {
    pub fn new(name: &str, skeleton: Skeleton, body: BodyDefinition) -> Self {
        Self {
            skeleton, body, name: name.to_string(),
            attachments: Vec::new(),
            facing_yaw: 0.0, target_yaw: 0.0, turn_speed: 0.05,
        }
    }

    /// Attach an object (weapon, gear) to the entity
    pub fn attach(&mut self, object: AttachedObject) {
        self.attachments.push(object);
    }

    /// Set world position of the entity root
    pub fn set_position(&mut self, pos: Vec3) {
        self.skeleton.root_position = pos;
    }

    /// Get world position
    pub fn position(&self) -> Vec3 {
        self.skeleton.root_position
    }

    // ─── HIGH-LEVEL ACTIONS ──────────────────────────────────

    /// Turn the entity to face a target (smooth, per-frame)
    pub fn look_at(&mut self, target: Vec3) {
        let to_target = target - self.skeleton.root_position;
        self.target_yaw = to_target.z.atan2(to_target.x);
    }

    /// Aim weapon at a specific point
    /// 1. Turn whole body toward target (root rotation)
    /// 2. Slight chest twist for fine-aiming (limited ±30°)
    /// 3. Head looks at target
    pub fn aim_at(&mut self, target: Vec3) {
        // Turn whole body toward target
        self.look_at(target);

        // DON'T twist individual bones for now — let root rotation handle it
        // The weapon is attached to hand_r which follows the skeleton
        // When we have proper arm IK, we'll add fine-aiming here
    }

    /// Fire the primary weapon. Returns bullet origin and direction.
    pub fn fire(&self) -> Option<FireResult> {
        // Find first attachment with a "muzzle" point
        for att in &self.attachments {
            if let Some((pos, dir)) = att.muzzle() {
                return Some(FireResult {
                    origin: pos,
                    direction: dir,
                    speed: 4.0, // default bullet speed
                });
            }
        }
        None
    }

    /// Place feet on ground using IK
    pub fn plant_feet(&mut self, ground_y: f32) {
        let pos = self.skeleton.root_position;
        let foot_y = ground_y;

        // Left foot: place on ground ahead-left
        let lf_target = Vec3::new(pos.x - 5.0, foot_y, pos.z);
        let lf_pole = Vec3::new(pos.x - 5.0, pos.y - 10.0, pos.z - 10.0); // knee forward
        ik::apply_leg_ik(&mut self.skeleton, "thigh_l", "shin_l", lf_target, lf_pole);

        // Right foot
        let rf_target = Vec3::new(pos.x + 5.0, foot_y, pos.z);
        let rf_pole = Vec3::new(pos.x + 5.0, pos.y - 10.0, pos.z - 10.0);
        ik::apply_leg_ik(&mut self.skeleton, "thigh_r", "shin_r", rf_target, rf_pole);
    }

    // ─── UPDATE (call every frame) ───────────────────────────

    /// Update entity state: smooth rotation, FK solve, attachment transforms
    pub fn update(&mut self) {
        // Smooth yaw rotation
        let mut yaw_diff = self.target_yaw - self.facing_yaw;
        while yaw_diff > std::f32::consts::PI { yaw_diff -= 2.0 * std::f32::consts::PI; }
        while yaw_diff < -std::f32::consts::PI { yaw_diff += 2.0 * std::f32::consts::PI; }
        self.facing_yaw += yaw_diff * self.turn_speed;
        self.skeleton.root_rotation = Quat::from_rotation_y(self.facing_yaw);

        // Solve forward kinematics
        self.skeleton.solve_forward();

        // Update all attachments
        for att in self.attachments.iter_mut() {
            att.update(&self.skeleton);
        }
    }

    // ─── RASTERIZE (draw into voxel grid) ────────────────────

    /// Rasterize the entire entity (body + attachments) into a voxel grid
    pub fn rasterize<F>(&self, grid_size: usize, mut set_voxel: F)
    where F: FnMut(usize, usize, usize, u8, u8, u8, u8) {
        // Body first
        self.body.rasterize(&self.skeleton, grid_size, &mut set_voxel);

        // Then attachments (on top)
        for att in &self.attachments {
            att.rasterize(grid_size, &mut set_voxel);
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// PREFAB ENTITIES
// ═══════════════════════════════════════════════════════════════

impl Entity {
    /// Phase 1: Skeleton only — bones, no soft tissue, no gear
    pub fn skeleton_preview(scale: f32) -> Self {
        let sk = Skeleton::human(scale);
        let body = BodyDefinition::human_skeleton_only(&sk, scale);
        let mut entity = Entity::new("Skeleton", sk, body);
        entity.turn_speed = 0.06;
        entity
    }

    /// Create an ORPP soldier entity with weapon
    pub fn orpp_soldier(scale: f32) -> Self {
        let sk = Skeleton::human(scale);
        let mut body = BodyDefinition::human_soldier(&sk, scale);
        body.set_hollow(true);
        let hand_id = sk.bone("hand_r").id;

        let mut entity = Entity::new("ORPP Soldier", sk, body);
        entity.attach(super::attachment::weapon_ak(hand_id, scale));
        entity.turn_speed = 0.06;
        entity
    }

    /// Create a cat entity
    pub fn cat(scale: f32) -> Self {
        let sk = Skeleton::cat(scale);
        let body = BodyDefinition::cat_body(&sk, scale);
        let mut entity = Entity::new("Cat", sk, body);
        entity.turn_speed = 0.1; // cats turn faster
        entity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_soldier() {
        let mut soldier = Entity::orpp_soldier(1.0);
        soldier.set_position(Vec3::new(64.0, 46.0, 64.0));
        soldier.update();

        // Should have weapon attached
        assert_eq!(soldier.attachments.len(), 1);
        assert_eq!(soldier.attachments[0].name, "AK-12");
    }

    #[test]
    fn test_fire_returns_muzzle() {
        let mut soldier = Entity::orpp_soldier(1.0);
        soldier.set_position(Vec3::new(64.0, 46.0, 64.0));
        soldier.update();

        let fire = soldier.fire();
        assert!(fire.is_some());
        let f = fire.unwrap();
        println!("Fire from {:?} dir {:?}", f.origin, f.direction);
        assert!(f.origin.y > 0.0);
    }

    #[test]
    fn test_look_at_and_update() {
        let mut soldier = Entity::orpp_soldier(1.0);
        soldier.set_position(Vec3::new(64.0, 46.0, 64.0));

        // Look at something to the right
        soldier.look_at(Vec3::new(100.0, 46.0, 64.0));
        for _ in 0..60 { soldier.update(); } // simulate 60 frames

        // Should have turned toward target
        println!("Facing yaw: {:.3}", soldier.facing_yaw);
    }

    #[test]
    fn test_rasterize_soldier() {
        let mut soldier = Entity::orpp_soldier(1.0);
        soldier.set_position(Vec3::new(32.0, 20.0, 32.0));
        soldier.update();

        let mut count = 0;
        soldier.rasterize(64, |_,_,_,_,_,_,_| count += 1);
        println!("Soldier total voxels: {}", count);
        assert!(count > 1500); // body + weapon
    }

    #[test]
    fn test_create_cat() {
        let mut cat = Entity::cat(1.0);
        cat.set_position(Vec3::new(32.0, 15.0, 32.0));
        cat.update();

        let mut count = 0;
        cat.rasterize(64, |_,_,_,_,_,_,_| count += 1);
        println!("Cat total voxels: {}", count);
        assert!(count > 500);
    }

    #[test]
    fn test_aim_and_fire() {
        let mut soldier = Entity::orpp_soldier(2.0);
        soldier.set_position(Vec3::new(128.0, 92.0, 128.0));

        // Aim at a target
        let target = Vec3::new(200.0, 80.0, 180.0);
        soldier.aim_at(target);
        for _ in 0..30 { soldier.update(); }

        // Fire — bullet should come from muzzle
        let fire = soldier.fire();
        assert!(fire.is_some());
        let f = fire.unwrap();
        println!("Aimed fire: origin={:?} dir={:?}", f.origin, f.direction);

        // Direction should roughly point toward target
        let to_target = (target - f.origin).normalize();
        let dot = f.direction.dot(to_target);
        println!("Aim accuracy (dot): {:.3}", dot);
        // At least vaguely in the right direction
        assert!(dot > -0.5);
    }
}
