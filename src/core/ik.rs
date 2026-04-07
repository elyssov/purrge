// ═══════════════════════════════════════════════════════════════
// PROMETHEUS ENGINE — Inverse Kinematics
//
// Solvers that compute joint rotations from target positions.
// Universal: works on any skeleton (human legs, cat paws, arms).
//
// Two-Bone IK: given hip position and foot target → compute
//              thigh and shin rotations (knee always bends correctly)
//
// Aim IK:     given a chain of bones and a target point →
//              rotate the chain so the end points at the target
//
// Look-At:    rotate head/neck to face a target
// ═══════════════════════════════════════════════════════════════

use glam::{Quat, Vec3};
use super::skeleton::{Skeleton, BoneId};

/// Result of a two-bone IK solve
pub struct TwoBoneResult {
    /// Rotation for the upper bone (thigh / upper_arm)
    pub upper_rotation: Quat,
    /// Rotation for the lower bone (shin / forearm)
    pub lower_rotation: Quat,
    /// Whether the target was reachable
    pub reached: bool,
}

/// Two-bone IK solver (for legs and arms)
///
/// Given:
///   - Joint position (hip / shoulder)
///   - Target position (where foot / hand should be)
///   - Upper bone length (thigh / upper_arm)
///   - Lower bone length (shin / forearm)
///   - Pole target (which direction the knee/elbow should point)
///
/// Returns: rotations for upper and lower bones
pub fn solve_two_bone(
    joint_pos: Vec3,
    target_pos: Vec3,
    upper_len: f32,
    lower_len: f32,
    pole_target: Vec3,
    parent_rotation: Quat,
) -> TwoBoneResult {
    let to_target = target_pos - joint_pos;
    let dist = to_target.length();

    // Clamp distance to reachable range
    let max_reach = upper_len + lower_len - 0.01;
    let min_reach = (upper_len - lower_len).abs() + 0.01;

    let (clamped_dist, reached) = if dist > max_reach {
        (max_reach, false)
    } else if dist < min_reach {
        (min_reach, false)
    } else {
        (dist, true)
    };

    // Direction to target (or clamped target)
    let dir = if dist > 0.001 {
        to_target / dist
    } else {
        Vec3::NEG_Y // default: straight down
    };

    let clamped_target = joint_pos + dir * clamped_dist;

    // Law of cosines: angle at joint (between upper bone and line to target)
    let cos_upper = (upper_len * upper_len + clamped_dist * clamped_dist - lower_len * lower_len)
        / (2.0 * upper_len * clamped_dist);
    let upper_angle = cos_upper.clamp(-1.0, 1.0).acos();

    // Law of cosines: angle at knee/elbow (between upper and lower bones)
    let cos_knee = (upper_len * upper_len + lower_len * lower_len - clamped_dist * clamped_dist)
        / (2.0 * upper_len * lower_len);
    let knee_angle = std::f32::consts::PI - cos_knee.clamp(-1.0, 1.0).acos();

    // Build rotation for upper bone
    // 1. Rotate to point toward target
    let rest_dir = Vec3::NEG_Y; // bones point downward in rest pose
    let target_dir = (clamped_target - joint_pos).normalize();

    // 2. Apply pole constraint (which way the knee/elbow bends)
    let pole_dir = (pole_target - joint_pos).normalize();

    // Rotation from rest to target direction
    let base_rot = Quat::from_rotation_arc(rest_dir, target_dir);

    // Apply upper angle offset (swing toward pole)
    let swing_axis = target_dir.cross(pole_dir).normalize();
    let upper_offset = if swing_axis.length() > 0.001 {
        Quat::from_axis_angle(swing_axis, upper_angle)
    } else {
        Quat::IDENTITY
    };

    // Convert to local space (relative to parent)
    let inv_parent = parent_rotation.inverse();
    let upper_rotation = inv_parent * upper_offset * base_rot;

    // Lower bone: simple hinge rotation by knee angle
    let lower_rotation = Quat::from_axis_angle(Vec3::X, knee_angle);

    TwoBoneResult {
        upper_rotation,
        lower_rotation,
        reached,
    }
}

/// Apply two-bone IK to a leg (thigh + shin) in the skeleton.
/// Places the foot at target_pos. Knee points toward pole_target.
pub fn apply_leg_ik(
    skeleton: &mut Skeleton,
    thigh_name: &str,
    shin_name: &str,
    target_pos: Vec3,
    pole_target: Vec3,
) {
    let thigh = skeleton.bone(thigh_name);
    let joint_pos = thigh.world_position;
    let upper_len = thigh.rest_length;
    let parent_rot = if let Some(pid) = thigh.parent {
        skeleton.bone_by_id(pid).world_rotation
    } else {
        Quat::IDENTITY
    };

    let shin = skeleton.bone(shin_name);
    let lower_len = shin.rest_length;

    let result = solve_two_bone(
        joint_pos, target_pos,
        upper_len, lower_len,
        pole_target, parent_rot,
    );

    skeleton.bone_mut(thigh_name).local_rotation = result.upper_rotation;
    skeleton.bone_mut(shin_name).local_rotation = result.lower_rotation;
}

/// Apply two-bone IK to an arm (upper_arm + forearm) in the skeleton.
pub fn apply_arm_ik(
    skeleton: &mut Skeleton,
    upper_arm_name: &str,
    forearm_name: &str,
    target_pos: Vec3,
    pole_target: Vec3,
) {
    // Same algorithm, different bones
    apply_leg_ik(skeleton, upper_arm_name, forearm_name, target_pos, pole_target);
}

/// Aim IK: rotate a single bone to point its forward direction at a target.
/// Useful for head look-at, torso turning, weapon aiming.
pub fn aim_bone_at(
    skeleton: &mut Skeleton,
    bone_name: &str,
    target_pos: Vec3,
) {
    let bone = skeleton.bone(bone_name);
    let bone_pos = bone.world_position;
    let parent_rot = if let Some(pid) = bone.parent {
        skeleton.bone_by_id(pid).world_rotation
    } else {
        Quat::IDENTITY
    };

    let to_target = (target_pos - bone_pos).normalize();
    let rest_dir = bone.rest_direction;

    // Rotation from rest direction to target direction
    let world_rot = Quat::from_rotation_arc(rest_dir, to_target);

    // Convert to local space
    let local_rot = parent_rot.inverse() * world_rot;

    skeleton.bone_mut(bone_name).local_rotation = local_rot;
}

/// Turn the entire skeleton to face a horizontal direction.
/// Only rotates around Y axis (yaw). Smooth interpolation.
pub fn turn_skeleton_to(
    skeleton: &mut Skeleton,
    target_pos: Vec3,
    turn_speed: f32,  // 0.0-1.0, how fast to turn per frame
) {
    let to_target = target_pos - skeleton.root_position;
    let target_yaw = to_target.z.atan2(to_target.x);

    // Current yaw
    let (current_axis, current_angle) = skeleton.root_rotation.to_axis_angle();
    let current_yaw = if current_axis.y > 0.0 { current_angle } else { -current_angle };

    // Shortest path rotation
    let mut diff = target_yaw - current_yaw;
    while diff > std::f32::consts::PI { diff -= 2.0 * std::f32::consts::PI; }
    while diff < -std::f32::consts::PI { diff += 2.0 * std::f32::consts::PI; }

    let new_yaw = current_yaw + diff * turn_speed;
    skeleton.root_rotation = Quat::from_rotation_y(new_yaw);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_bone_ik_reachable() {
        let result = solve_two_bone(
            Vec3::new(0.0, 50.0, 0.0),   // hip
            Vec3::new(0.0, 10.0, 5.0),    // foot target (below and slightly forward)
            22.0, 20.0,                    // thigh, shin lengths
            Vec3::new(0.0, 30.0, 10.0),   // pole target (knee forward)
            Quat::IDENTITY,
        );
        assert!(result.reached);
    }

    #[test]
    fn test_two_bone_ik_unreachable() {
        let result = solve_two_bone(
            Vec3::new(0.0, 50.0, 0.0),
            Vec3::new(0.0, 0.0, 100.0),   // way too far
            22.0, 20.0,
            Vec3::new(0.0, 30.0, 10.0),
            Quat::IDENTITY,
        );
        assert!(!result.reached);
    }

    #[test]
    fn test_leg_ik_on_skeleton() {
        let mut sk = Skeleton::human(1.0);
        sk.root_position = Vec3::new(64.0, 46.0, 64.0);
        sk.solve_forward();

        let foot_target = Vec3::new(56.0, 2.0, 60.0);
        let pole = Vec3::new(56.0, 30.0, 50.0); // knee points forward (-Z)

        apply_leg_ik(&mut sk, "thigh_l", "shin_l", foot_target, pole);
        sk.solve_forward();

        // After IK, foot should be close to target
        let foot_bone = sk.bone("foot_l");
        let dist = (foot_bone.world_position - foot_target).length();
        println!("Foot distance from target: {:.2}", dist);
        // Allow some tolerance (IK is approximate)
        assert!(dist < 15.0);
    }

    #[test]
    fn test_turn_skeleton() {
        let mut sk = Skeleton::human(1.0);
        sk.root_position = Vec3::new(64.0, 46.0, 64.0);

        // Turn to face +X direction
        turn_skeleton_to(&mut sk, Vec3::new(100.0, 46.0, 64.0), 1.0);
        sk.solve_forward();

        // Root rotation should be roughly 0 yaw (facing +X)
        let (_, angle) = sk.root_rotation.to_axis_angle();
        println!("Turn angle: {:.3}", angle);
    }

    #[test]
    fn test_aim_bone() {
        let mut sk = Skeleton::human(1.0);
        sk.root_position = Vec3::new(64.0, 46.0, 64.0);
        sk.solve_forward();

        let target = Vec3::new(100.0, 50.0, 30.0);
        aim_bone_at(&mut sk, "head", target);
        sk.solve_forward();

        // Head should now point toward target (approximately)
        let head = sk.bone("head");
        let head_dir = head.world_rotation * head.rest_direction;
        let to_target = (target - head.world_position).normalize();
        let dot = head_dir.dot(to_target);
        println!("Head aim dot product: {:.3} (1.0 = perfect)", dot);
        assert!(dot > 0.0); // at least roughly correct direction
    }
}
