// ═══════════════════════════════════════════════════════════════
// PROMETHEUS ENGINE — Skeleton System
//
// A tree of bones with joint constraints, forward kinematics,
// and world-space resolution. Foundation of all animation.
//
// Key principles:
// - Rotations are quaternions (no gimbal lock, smooth slerp)
// - Constraints are anatomical (knee: hinge 5°-130°, shoulder: ball-socket)
// - Forward kinematics cascades parent → child automatically
// - World positions computed, never manually set
// ═══════════════════════════════════════════════════════════════

use glam::{Quat, Vec3};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Unique identifier for a bone
pub type BoneId = u16;

/// Joint constraint — defines how a bone can rotate relative to its parent
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum JointConstraint {
    /// No rotation allowed (fused joint)
    Fixed,

    /// Full freedom of rotation (wrist, neck — limited by cone)
    Free,

    /// Single-axis rotation (knee, elbow)
    Hinge {
        /// Local axis of rotation (e.g., Vec3::X for knee)
        axis: Vec3,
        /// Minimum angle in radians (e.g., 0.087 = 5° for knee)
        min_angle: f32,
        /// Maximum angle in radians (e.g., 2.27 = 130° for knee)
        max_angle: f32,
    },

    /// Ball-and-socket (shoulder, hip)
    BallSocket {
        /// Maximum cone angle from rest direction (radians)
        cone_angle: f32,
        /// Twist limits around bone axis
        twist_min: f32,
        twist_max: f32,
    },
}

/// A single bone in the skeleton hierarchy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bone {
    /// Unique ID
    pub id: BoneId,
    /// Human-readable name
    pub name: String,
    /// Parent bone (None = root)
    pub parent: Option<BoneId>,
    /// Children
    pub children: Vec<BoneId>,

    // --- Rest pose (T-pose, never changes) ---
    /// Length of this bone
    pub rest_length: f32,
    /// Local rotation in rest pose relative to parent
    pub rest_rotation: Quat,
    /// Direction from this bone's joint to the next (local space, rest)
    pub rest_direction: Vec3,

    // --- Constraint ---
    pub constraint: JointConstraint,

    // --- Current state (changes every frame) ---
    /// Local rotation applied on top of rest_rotation
    pub local_rotation: Quat,

    // --- Computed (from forward kinematics) ---
    /// World-space position of this bone's joint
    pub world_position: Vec3,
    /// World-space rotation (accumulated from root)
    pub world_rotation: Quat,
    /// World-space position of the END of this bone (start of children)
    pub world_end_position: Vec3,
}

impl Bone {
    fn new(id: BoneId, name: &str, parent: Option<BoneId>, length: f32, direction: Vec3, constraint: JointConstraint) -> Self {
        Self {
            id, name: name.to_string(), parent, children: Vec::new(),
            rest_length: length,
            rest_rotation: Quat::IDENTITY,
            rest_direction: direction.normalize(),
            constraint,
            local_rotation: Quat::IDENTITY,
            world_position: Vec3::ZERO,
            world_rotation: Quat::IDENTITY,
            world_end_position: Vec3::ZERO,
        }
    }
}

/// Named attachment point on a bone
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttachPoint {
    pub name: String,
    pub bone_id: BoneId,
    /// Offset from bone joint in bone-local space
    pub local_offset: Vec3,
    /// Rotation relative to bone
    pub local_rotation: Quat,
}

/// The full skeleton — a tree of bones
#[derive(Clone, Serialize, Deserialize)]
pub struct Skeleton {
    bones: Vec<Bone>,
    name_to_id: HashMap<String, BoneId>,
    root: BoneId,
    /// Position of the root in world space
    pub root_position: Vec3,
    /// Rotation of the entire skeleton (facing direction)
    pub root_rotation: Quat,
    /// Attachment points
    attach_points: Vec<AttachPoint>,
}

impl Skeleton {
    /// Create an empty skeleton with a root bone
    pub fn new(root_name: &str) -> Self {
        let root_bone = Bone::new(0, root_name, None, 0.0, Vec3::Y, JointConstraint::Free);
        let mut name_to_id = HashMap::new();
        name_to_id.insert(root_name.to_string(), 0);
        Self {
            bones: vec![root_bone],
            name_to_id,
            root: 0,
            root_position: Vec3::ZERO,
            root_rotation: Quat::IDENTITY,
            attach_points: Vec::new(),
        }
    }

    /// Add a bone to the skeleton. Returns the new bone's ID.
    pub fn add_bone(&mut self, name: &str, parent_name: &str, length: f32, direction: Vec3, constraint: JointConstraint) -> BoneId {
        let parent_id = self.name_to_id[parent_name];
        let id = self.bones.len() as BoneId;
        let bone = Bone::new(id, name, Some(parent_id), length, direction, constraint);
        self.bones.push(bone);
        self.bones[parent_id as usize].children.push(id);
        self.name_to_id.insert(name.to_string(), id);
        id
    }

    /// Add an attachment point to a bone
    pub fn add_attach_point(&mut self, name: &str, bone_name: &str, offset: Vec3) {
        let bone_id = self.name_to_id[bone_name];
        self.attach_points.push(AttachPoint {
            name: name.to_string(),
            bone_id,
            local_offset: offset,
            local_rotation: Quat::IDENTITY,
        });
    }

    /// Get bone by name
    pub fn bone(&self, name: &str) -> &Bone {
        &self.bones[self.name_to_id[name] as usize]
    }

    /// Get mutable bone by name
    pub fn bone_mut(&mut self, name: &str) -> &mut Bone {
        let id = self.name_to_id[name];
        &mut self.bones[id as usize]
    }

    /// Get bone by ID
    pub fn bone_by_id(&self, id: BoneId) -> &Bone {
        &self.bones[id as usize]
    }

    /// Set local rotation of a bone (will be clamped by constraint)
    pub fn set_rotation(&mut self, name: &str, rotation: Quat) {
        let id = self.name_to_id[name];
        let bone = &mut self.bones[id as usize];
        bone.local_rotation = Self::clamp_rotation(&bone.constraint, rotation);
    }

    /// Set rotation by angle around bone's constraint axis (for hinge joints)
    pub fn set_hinge_angle(&mut self, name: &str, angle: f32) {
        let id = self.name_to_id[name];
        let bone = &self.bones[id as usize];
        if let JointConstraint::Hinge { axis, min_angle, max_angle } = &bone.constraint {
            let clamped = angle.clamp(*min_angle, *max_angle);
            let rotation = Quat::from_axis_angle(*axis, clamped);
            self.bones[id as usize].local_rotation = rotation;
        }
    }

    /// Clamp rotation to constraint limits
    fn clamp_rotation(constraint: &JointConstraint, rotation: Quat) -> Quat {
        match constraint {
            JointConstraint::Fixed => Quat::IDENTITY,
            JointConstraint::Free => rotation,
            JointConstraint::Hinge { axis, min_angle, max_angle } => {
                // Project rotation onto hinge axis
                let (rot_axis, mut angle) = rotation.to_axis_angle();
                // Check if rotation is roughly around the correct axis
                if rot_axis.dot(*axis) < 0.0 { angle = -angle; }
                let clamped = angle.clamp(*min_angle, *max_angle);
                Quat::from_axis_angle(*axis, clamped)
            }
            JointConstraint::BallSocket { cone_angle, .. } => {
                let (axis, angle) = rotation.to_axis_angle();
                let clamped = angle.clamp(-*cone_angle, *cone_angle);
                Quat::from_axis_angle(axis, clamped)
            }
        }
    }

    /// Solve forward kinematics — compute all world positions from root down.
    /// Call this after setting rotations and before reading world positions.
    pub fn solve_forward(&mut self) {
        self.solve_bone(self.root, self.root_position, self.root_rotation);
    }

    fn solve_bone(&mut self, id: BoneId, parent_world_pos: Vec3, parent_world_rot: Quat) {
        let bone = &self.bones[id as usize];

        // World rotation = parent's world rotation × rest rotation × local rotation
        let world_rot = parent_world_rot * bone.rest_rotation * bone.local_rotation;

        // World position = parent's position (bone starts where parent ends)
        let world_pos = parent_world_pos;

        // End position = start + rotated direction × length
        let world_end = world_pos + world_rot * (bone.rest_direction * bone.rest_length);

        // Store
        let children = bone.children.clone();
        let bone = &mut self.bones[id as usize];
        bone.world_position = world_pos;
        bone.world_rotation = world_rot;
        bone.world_end_position = world_end;

        // Recurse to children (they start where this bone ends)
        for child_id in children {
            self.solve_bone(child_id, world_end, world_rot);
        }
    }

    /// Get world position of a named attachment point
    pub fn get_attach_world_pos(&self, name: &str) -> Option<Vec3> {
        for ap in &self.attach_points {
            if ap.name == name {
                let bone = &self.bones[ap.bone_id as usize];
                return Some(bone.world_position + bone.world_rotation * ap.local_offset);
            }
        }
        None
    }

    /// Get world position and forward direction of an attachment point
    pub fn get_attach_world_transform(&self, name: &str) -> Option<(Vec3, Vec3)> {
        for ap in &self.attach_points {
            if ap.name == name {
                let bone = &self.bones[ap.bone_id as usize];
                let pos = bone.world_position + bone.world_rotation * ap.local_offset;
                let fwd = bone.world_rotation * ap.local_rotation * Vec3::Z;
                return Some((pos, fwd));
            }
        }
        None
    }

    /// Get all bones (for iteration)
    pub fn bones(&self) -> &[Bone] {
        &self.bones
    }

    /// Number of bones
    pub fn bone_count(&self) -> usize {
        self.bones.len()
    }

    /// Print skeleton hierarchy (debug)
    pub fn print_hierarchy(&self) {
        self.print_bone(self.root, 0);
    }

    fn print_bone(&self, id: BoneId, depth: usize) {
        let bone = &self.bones[id as usize];
        let indent = "  ".repeat(depth);
        println!("{}├── {} (len={:.1}, pos={:.1},{:.1},{:.1})",
            indent, bone.name, bone.rest_length,
            bone.world_position.x, bone.world_position.y, bone.world_position.z);
        for &child in &bone.children {
            self.print_bone(child, depth + 1);
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// PREFAB SKELETONS
// ═══════════════════════════════════════════════════════════════

impl Skeleton {
    /// Create a standard human skeleton (22 bones)
    /// Scale: 1.0 = 180cm human at 128³ grid
    pub fn human(scale: f32) -> Self {
        let mut sk = Skeleton::new("pelvis");

        // Spine chain (up from pelvis) — Aelis proportions
        sk.add_bone("spine", "pelvis", 11.0*scale, Vec3::Y,
            JointConstraint::BallSocket { cone_angle: 0.3, twist_min: -0.2, twist_max: 0.2 });
        sk.add_bone("chest", "spine", 15.0*scale, Vec3::Y,
            JointConstraint::BallSocket { cone_angle: 0.25, twist_min: -0.3, twist_max: 0.3 });
        sk.add_bone("neck", "chest", 5.0*scale, Vec3::Y,
            JointConstraint::BallSocket { cone_angle: 0.5, twist_min: -0.8, twist_max: 0.8 });
        sk.add_bone("head", "neck", 9.0*scale, Vec3::Y,
            JointConstraint::BallSocket { cone_angle: 0.3, twist_min: -0.5, twist_max: 0.5 });

        // Left arm — shoulder shortened to kill "airplane" (6→4)
        sk.add_bone("shoulder_l", "chest", 4.0*scale, Vec3::NEG_X,
            JointConstraint::Fixed);
        sk.add_bone("upper_arm_l", "shoulder_l", 11.0*scale, Vec3::NEG_X,
            JointConstraint::BallSocket { cone_angle: 1.2, twist_min: -1.0, twist_max: 1.0 });
        sk.add_bone("forearm_l", "upper_arm_l", 9.0*scale, Vec3::NEG_X,
            JointConstraint::Hinge { axis: Vec3::Y, min_angle: 0.0, max_angle: 2.6 });
        sk.add_bone("hand_l", "forearm_l", 4.0*scale, Vec3::NEG_X,
            JointConstraint::BallSocket { cone_angle: 1.0, twist_min: -0.5, twist_max: 0.5 });

        // Right arm (mirror)
        sk.add_bone("shoulder_r", "chest", 4.0*scale, Vec3::X,
            JointConstraint::Fixed);
        sk.add_bone("upper_arm_r", "shoulder_r", 11.0*scale, Vec3::X,
            JointConstraint::BallSocket { cone_angle: 1.2, twist_min: -1.0, twist_max: 1.0 });
        sk.add_bone("forearm_r", "upper_arm_r", 9.0*scale, Vec3::X,
            JointConstraint::Hinge { axis: Vec3::Y, min_angle: 0.0, max_angle: 2.6 });
        sk.add_bone("hand_r", "forearm_r", 4.0*scale, Vec3::X,
            JointConstraint::BallSocket { cone_angle: 1.0, twist_min: -0.5, twist_max: 0.5 });

        // Left leg — hip shortened (4→3)
        sk.add_bone("hip_l", "pelvis", 3.0*scale, Vec3::NEG_X,
            JointConstraint::Fixed);
        sk.add_bone("thigh_l", "hip_l", 22.0*scale, Vec3::new(-0.15, -1.0, 0.0).normalize(),
            JointConstraint::BallSocket { cone_angle: 1.0, twist_min: -0.3, twist_max: 0.3 });
        sk.add_bone("shin_l", "thigh_l", 20.0*scale, Vec3::NEG_Y,
            JointConstraint::Hinge { axis: Vec3::X, min_angle: 0.087, max_angle: 2.27 }); // knee: 5°-130°
        sk.add_bone("foot_l", "shin_l", 8.0*scale, Vec3::NEG_Z,
            JointConstraint::Hinge { axis: Vec3::X, min_angle: -0.35, max_angle: 0.87 }); // ankle: -20° to 50°

        // Right leg (Vitruvian: mirror spread)
        sk.add_bone("hip_r", "pelvis", 3.0*scale, Vec3::X,
            JointConstraint::Fixed);
        sk.add_bone("thigh_r", "hip_r", 22.0*scale, Vec3::new(0.15, -1.0, 0.0).normalize(),
            JointConstraint::BallSocket { cone_angle: 1.0, twist_min: -0.3, twist_max: 0.3 });
        sk.add_bone("shin_r", "thigh_r", 20.0*scale, Vec3::NEG_Y,
            JointConstraint::Hinge { axis: Vec3::X, min_angle: 0.087, max_angle: 2.27 });
        sk.add_bone("foot_r", "shin_r", 8.0*scale, Vec3::NEG_Z,
            JointConstraint::Hinge { axis: Vec3::X, min_angle: -0.35, max_angle: 0.87 });

        // Attachment points (offsets along bone direction)
        sk.add_attach_point("hand_r.grip", "hand_r", Vec3::new(3.0*scale, 0.0, 0.0));
        sk.add_attach_point("hand_l.support", "hand_l", Vec3::new(-3.0*scale, 0.0, 0.0));
        sk.add_attach_point("hip_r.holster", "hip_r", Vec3::new(5.0*scale, 0.0, 0.0));
        sk.add_attach_point("hip_l.holster", "hip_l", Vec3::new(-5.0*scale, 0.0, 0.0));
        sk.add_attach_point("back.sling", "chest", Vec3::new(0.0, 5.0*scale, 5.0*scale));

        sk
    }

    /// Create a cat skeleton (26 bones)
    pub fn cat(scale: f32) -> Self {
        let mut sk = Skeleton::new("pelvis");

        // Spine (horizontal — cat walks on all fours)
        sk.add_bone("spine1", "pelvis", 8.0*scale, Vec3::Z,
            JointConstraint::BallSocket { cone_angle: 0.3, twist_min: -0.2, twist_max: 0.2 });
        sk.add_bone("spine2", "spine1", 8.0*scale, Vec3::Z,
            JointConstraint::BallSocket { cone_angle: 0.3, twist_min: -0.2, twist_max: 0.2 });
        sk.add_bone("neck", "spine2", 5.0*scale, Vec3::new(0.0, 0.3, 0.7).normalize(),
            JointConstraint::BallSocket { cone_angle: 0.6, twist_min: -0.5, twist_max: 0.5 });
        sk.add_bone("head", "neck", 5.0*scale, Vec3::new(0.0, 0.2, 0.8).normalize(),
            JointConstraint::BallSocket { cone_angle: 0.5, twist_min: -0.3, twist_max: 0.3 });

        // Ears
        sk.add_bone("ear_l", "head", 2.0*scale, Vec3::new(-0.3, 0.9, 0.0).normalize(), JointConstraint::Free);
        sk.add_bone("ear_r", "head", 2.0*scale, Vec3::new(0.3, 0.9, 0.0).normalize(), JointConstraint::Free);

        // Jaw
        sk.add_bone("jaw", "head", 2.0*scale, Vec3::new(0.0, -0.3, 0.7).normalize(),
            JointConstraint::Hinge { axis: Vec3::X, min_angle: 0.0, max_angle: 0.5 });

        // Front legs
        sk.add_bone("shoulder_l", "spine2", 3.0*scale, Vec3::NEG_X, JointConstraint::Fixed);
        sk.add_bone("upper_arm_l", "shoulder_l", 7.0*scale, Vec3::NEG_Y,
            JointConstraint::BallSocket { cone_angle: 0.8, twist_min: -0.3, twist_max: 0.3 });
        sk.add_bone("forearm_l", "upper_arm_l", 6.0*scale, Vec3::NEG_Y,
            JointConstraint::Hinge { axis: Vec3::X, min_angle: 0.0, max_angle: 2.0 });
        sk.add_bone("paw_fl", "forearm_l", 3.0*scale, Vec3::NEG_Y, JointConstraint::Free);

        sk.add_bone("shoulder_r", "spine2", 3.0*scale, Vec3::X, JointConstraint::Fixed);
        sk.add_bone("upper_arm_r", "shoulder_r", 7.0*scale, Vec3::NEG_Y,
            JointConstraint::BallSocket { cone_angle: 0.8, twist_min: -0.3, twist_max: 0.3 });
        sk.add_bone("forearm_r", "upper_arm_r", 6.0*scale, Vec3::NEG_Y,
            JointConstraint::Hinge { axis: Vec3::X, min_angle: 0.0, max_angle: 2.0 });
        sk.add_bone("paw_fr", "forearm_r", 3.0*scale, Vec3::NEG_Y, JointConstraint::Free);

        // Back legs
        sk.add_bone("hip_l", "pelvis", 3.0*scale, Vec3::NEG_X, JointConstraint::Fixed);
        sk.add_bone("thigh_l", "hip_l", 8.0*scale, Vec3::NEG_Y,
            JointConstraint::BallSocket { cone_angle: 0.8, twist_min: -0.3, twist_max: 0.3 });
        sk.add_bone("shin_l", "thigh_l", 7.0*scale, Vec3::NEG_Y,
            JointConstraint::Hinge { axis: Vec3::X, min_angle: 0.087, max_angle: 2.0 });
        sk.add_bone("paw_bl", "shin_l", 4.0*scale, Vec3::NEG_Y, JointConstraint::Free);

        sk.add_bone("hip_r", "pelvis", 3.0*scale, Vec3::X, JointConstraint::Fixed);
        sk.add_bone("thigh_r", "hip_r", 8.0*scale, Vec3::NEG_Y,
            JointConstraint::BallSocket { cone_angle: 0.8, twist_min: -0.3, twist_max: 0.3 });
        sk.add_bone("shin_r", "thigh_r", 7.0*scale, Vec3::NEG_Y,
            JointConstraint::Hinge { axis: Vec3::X, min_angle: 0.087, max_angle: 2.0 });
        sk.add_bone("paw_br", "shin_r", 4.0*scale, Vec3::NEG_Y, JointConstraint::Free);

        // Tail (4 segments, decreasing length)
        sk.add_bone("tail1", "pelvis", 5.0*scale, Vec3::new(0.0, 0.3, -0.9).normalize(),
            JointConstraint::BallSocket { cone_angle: 0.8, twist_min: -0.5, twist_max: 0.5 });
        sk.add_bone("tail2", "tail1", 4.0*scale, Vec3::new(0.0, 0.3, -0.9).normalize(),
            JointConstraint::BallSocket { cone_angle: 0.8, twist_min: -0.5, twist_max: 0.5 });
        sk.add_bone("tail3", "tail2", 3.0*scale, Vec3::new(0.0, 0.3, -0.9).normalize(),
            JointConstraint::BallSocket { cone_angle: 0.8, twist_min: -0.5, twist_max: 0.5 });
        sk.add_bone("tail4", "tail3", 2.0*scale, Vec3::new(0.0, 0.3, -0.9).normalize(),
            JointConstraint::Free);

        sk
    }

    /// Dog skeleton — uses same bone names as cat for animation compatibility
    pub fn dog(scale: f32) -> Self {
        let mut sk = Skeleton::new("pelvis");
        // Spine (horizontal)
        sk.add_bone("spine1", "pelvis", 6.0*scale, Vec3::Z,
            JointConstraint::BallSocket { cone_angle: 0.3, twist_min: -0.2, twist_max: 0.2 });
        sk.add_bone("spine2", "spine1", 6.0*scale, Vec3::Z,
            JointConstraint::BallSocket { cone_angle: 0.3, twist_min: -0.2, twist_max: 0.2 });
        sk.add_bone("neck", "spine2", 4.0*scale, Vec3::new(0.0, 0.4, 0.6).normalize(),
            JointConstraint::BallSocket { cone_angle: 0.5, twist_min: -0.3, twist_max: 0.3 });
        sk.add_bone("head", "neck", 4.0*scale, Vec3::new(0.0, 0.2, 0.8).normalize(),
            JointConstraint::BallSocket { cone_angle: 0.4, twist_min: -0.3, twist_max: 0.3 });
        sk.add_bone("ear_l", "head", 1.5*scale, Vec3::new(-0.3, 0.9, 0.0).normalize(), JointConstraint::Free);
        sk.add_bone("ear_r", "head", 1.5*scale, Vec3::new(0.3, 0.9, 0.0).normalize(), JointConstraint::Free);
        // Front legs (same names as cat for compat: shoulder → upper_arm → forearm)
        for (suffix, dir) in [("_l", Vec3::NEG_X), ("_r", Vec3::X)] {
            sk.add_bone(&format!("shoulder{}", suffix), "spine2", 2.0*scale, dir, JointConstraint::Fixed);
            sk.add_bone(&format!("upper_arm{}", suffix), &format!("shoulder{}", suffix), 7.0*scale, Vec3::NEG_Y,
                JointConstraint::BallSocket { cone_angle: 0.8, twist_min: -0.3, twist_max: 0.3 });
            sk.add_bone(&format!("forearm{}", suffix), &format!("upper_arm{}", suffix), 5.0*scale, Vec3::NEG_Y,
                JointConstraint::Hinge { axis: Vec3::X, min_angle: 0.0, max_angle: 2.0 });
        }
        // Back legs (hip → thigh → shin)
        for (suffix, dir) in [("_l", Vec3::NEG_X), ("_r", Vec3::X)] {
            sk.add_bone(&format!("hip{}", suffix), "pelvis", 2.0*scale, dir, JointConstraint::Fixed);
            sk.add_bone(&format!("thigh{}", suffix), &format!("hip{}", suffix), 7.0*scale, Vec3::NEG_Y,
                JointConstraint::BallSocket { cone_angle: 0.8, twist_min: -0.3, twist_max: 0.3 });
            sk.add_bone(&format!("shin{}", suffix), &format!("thigh{}", suffix), 5.0*scale, Vec3::NEG_Y,
                JointConstraint::Hinge { axis: Vec3::X, min_angle: 0.087, max_angle: 2.0 });
        }
        // Tail (3 segments like cat for wag animation)
        sk.add_bone("tail1", "pelvis", 4.0*scale, Vec3::new(0.0, 0.5, -0.9).normalize(),
            JointConstraint::BallSocket { cone_angle: 0.8, twist_min: -0.5, twist_max: 0.5 });
        sk.add_bone("tail2", "tail1", 3.0*scale, Vec3::new(0.0, 0.3, -0.9).normalize(),
            JointConstraint::BallSocket { cone_angle: 0.8, twist_min: -0.5, twist_max: 0.5 });
        sk.add_bone("tail3", "tail2", 2.0*scale, Vec3::new(0.0, 0.3, -0.9).normalize(),
            JointConstraint::Free);
        sk
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_human_skeleton_creation() {
        let sk = Skeleton::human(1.0);
        assert!(sk.bone_count() >= 21); // 21 bones in current human skeleton
        assert_eq!(sk.bone("pelvis").id, 0);
        assert_eq!(sk.bone("head").parent, Some(sk.bone("neck").id));
    }

    #[test]
    fn test_forward_kinematics() {
        let mut sk = Skeleton::human(1.0);
        sk.root_position = Vec3::new(64.0, 46.0, 64.0);
        sk.solve_forward();

        // Head should be above pelvis
        let head = sk.bone("head");
        let pelvis = sk.bone("pelvis");
        assert!(head.world_position.y > pelvis.world_position.y);
        println!("Pelvis: {:?}", pelvis.world_position);
        println!("Head: {:?}", head.world_position);
    }

    #[test]
    fn test_knee_constraint() {
        let mut sk = Skeleton::human(1.0);
        // Try to set knee to negative angle (impossible!)
        sk.set_hinge_angle("shin_l", -0.5);
        let bone = sk.bone("shin_l");
        let (_, angle) = bone.local_rotation.to_axis_angle();
        // Should be clamped to minimum (5°)
        assert!(angle >= 0.08);
    }

    #[test]
    fn test_cat_skeleton() {
        let sk = Skeleton::cat(1.0);
        assert!(sk.bone_count() >= 26);
        // Cat has tail
        assert!(sk.name_to_id.contains_key("tail4"));
        // Cat has ears
        assert!(sk.name_to_id.contains_key("ear_l"));
    }

    #[test]
    fn test_attach_points() {
        let mut sk = Skeleton::human(1.0);
        sk.root_position = Vec3::new(64.0, 46.0, 64.0);
        sk.solve_forward();

        let grip = sk.get_attach_world_pos("hand_r.grip");
        assert!(grip.is_some());
        println!("Grip position: {:?}", grip.unwrap());
    }
}
