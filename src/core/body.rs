// ═══════════════════════════════════════════════════════════════
// PROMETHEUS ENGINE — Voxel Body Builder
//
// Takes a skeleton with solved world positions and fills voxels
// around each bone using elliptical cross-section profiles.
// Supports layered rendering: body → clothing → armor → gear.
// ═══════════════════════════════════════════════════════════════

use glam::{Vec3, Quat, Vec2};
use super::skeleton::{Skeleton, BoneId};

/// Cross-section at a point along a bone
#[derive(Clone, Debug)]
pub struct BodySection {
    /// Position along bone: 0.0 = joint start, 1.0 = joint end
    pub t: f32,
    /// Radius in bone-local X (width)
    pub radius_x: f32,
    /// Radius in bone-local Z (depth)
    pub radius_z: f32,
    /// Center offset from bone axis (for asymmetric shapes)
    pub offset: Vec2,
    /// Superellipse exponent: 2.0 = ellipse, 2.5 = rounded rect, 3.0+ = boxy
    pub n: f32,
}

/// Profile for one bone — how "thick" the body is around it
#[derive(Clone, Debug)]
pub struct BoneProfile {
    pub bone_id: BoneId,
    pub sections: Vec<BodySection>,
    pub material_id: u8,
    pub color: [u8; 3],
}

impl BoneProfile {
    /// Simple cylindrical profile (uniform radius)
    pub fn cylinder(bone_id: BoneId, radius: f32, mat: u8, color: [u8; 3]) -> Self {
        Self {
            bone_id,
            sections: vec![
                BodySection { t: 0.0, radius_x: radius, radius_z: radius, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 1.0, radius_x: radius, radius_z: radius, offset: Vec2::ZERO, n: 2.0 },
            ],
            material_id: mat,
            color,
        }
    }

    /// Tapered profile (thicker at start, thinner at end)
    pub fn tapered(bone_id: BoneId, r_start: f32, r_end: f32, mat: u8, color: [u8; 3]) -> Self {
        Self {
            bone_id,
            sections: vec![
                BodySection { t: 0.0, radius_x: r_start, radius_z: r_start, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 1.0, radius_x: r_end, radius_z: r_end, offset: Vec2::ZERO, n: 2.0 },
            ],
            material_id: mat,
            color,
        }
    }

    /// Elliptical profile (different X and Z radii — flat torso)
    pub fn elliptical(bone_id: BoneId, sections: Vec<BodySection>, mat: u8, color: [u8; 3]) -> Self {
        Self { bone_id, sections, material_id: mat, color }
    }

    /// Interpolate cross-section at position t (0.0-1.0)
    fn section_at(&self, t: f32) -> BodySection {
        if self.sections.is_empty() {
            return BodySection { t, radius_x: 1.0, radius_z: 1.0, offset: Vec2::ZERO, n: 2.0 };
        }
        if self.sections.len() == 1 || t <= self.sections[0].t {
            return self.sections[0].clone();
        }
        if t >= self.sections.last().unwrap().t {
            return self.sections.last().unwrap().clone();
        }

        // Find surrounding sections and interpolate
        for i in 0..self.sections.len() - 1 {
            let s0 = &self.sections[i];
            let s1 = &self.sections[i + 1];
            if t >= s0.t && t <= s1.t {
                let frac = (t - s0.t) / (s1.t - s0.t);
                return BodySection {
                    t,
                    radius_x: s0.radius_x + (s1.radius_x - s0.radius_x) * frac,
                    radius_z: s0.radius_z + (s1.radius_z - s0.radius_z) * frac,
                    offset: s0.offset + (s1.offset - s0.offset) * frac,
                    n: s0.n + (s1.n - s0.n) * frac,
                };
            }
        }
        self.sections.last().unwrap().clone()
    }
}

/// A decal (detail) placed on a bone surface: eyes, nose, whiskers, scars, patches
#[derive(Clone, Debug)]
pub struct BoneDecal {
    pub bone_id: BoneId,
    /// Position relative to bone joint (bone-local space)
    pub local_pos: Vec3,
    /// Type of decal
    pub shape: DecalShape,
    pub material: u8,
    pub color: [u8; 3],
}

#[derive(Clone, Debug)]
pub enum DecalShape {
    /// Single voxel point
    Point,
    /// Sphere of given radius
    Sphere(f32),
    /// Horizontal line (width)
    LineH(f32),
    /// Vertical line (height)
    LineV(f32),
    /// Ellipse (rx, rz)
    Ellipse(f32, f32),
}

/// Complete body definition — profiles for all bones + face/detail decals
pub struct BodyDefinition {
    pub profiles: Vec<BoneProfile>,
    pub decals: Vec<BoneDecal>,
    /// If true, only render the outer shell of each profile (1 voxel thick).
    /// Interior is empty. Massive memory savings at high resolution.
    /// 100³ cube: 1M voxels full vs 59K shell = 94% savings.
    pub hollow: bool,
}

impl BodyDefinition {
    pub fn new() -> Self {
        Self { profiles: Vec::new(), decals: Vec::new(), hollow: false }
    }

    /// Enable hollow mode — only render outer shell, interior empty
    pub fn set_hollow(&mut self, hollow: bool) { self.hollow = hollow; }

    pub fn add(&mut self, profile: BoneProfile) {
        self.profiles.push(profile);
    }

    /// Add a face/body detail (eye, nose, whisker, scar, visor, etc.)
    pub fn add_decal(&mut self, bone_id: BoneId, local_pos: Vec3, shape: DecalShape, material: u8, color: [u8; 3]) {
        self.decals.push(BoneDecal { bone_id, local_pos, shape, material, color });
    }

    /// Rasterize body + decals into a voxel grid using current skeleton pose.
    pub fn rasterize<F>(&self, skeleton: &Skeleton, grid_size: usize, mut set_voxel: F)
    where F: FnMut(usize, usize, usize, u8, u8, u8, u8) {
        // First: bone profiles (body volume)
        // Bone blending: extend t slightly for smooth joint transitions
        let t_extend = 0.08;
        for profile in &self.profiles {
            let bone = skeleton.bone_by_id(profile.bone_id);
            let start = bone.world_position;
            let end = bone.world_end_position;
            let bone_dir = end - start;
            let bone_len = bone_dir.length();
            if bone_len < 0.01 { continue; }

            let bone_fwd = bone_dir / bone_len;
            let (bone_right, bone_up) = build_perpendicular_frame(bone_fwd);

            // Extended range for bone blending (overlap with neighbors)
            let steps = (bone_len * 2.0).max(4.0) as usize;
            let extended_steps = ((1.0 + t_extend * 2.0) * steps as f32) as usize;
            for i in 0..=extended_steps {
                let t = -t_extend + i as f32 / steps as f32;

                // Fade at bone boundaries (smooth blending)
                let fade = if t < 0.0 { (1.0 + t / t_extend).max(0.0) }
                      else if t > 1.0 { (1.0 - (t - 1.0) / t_extend).max(0.0) }
                      else { 1.0 };
                if fade < 0.01 { continue; }

                let section = profile.section_at(t.clamp(0.0, 1.0));
                let rx = section.radius_x * fade;
                let rz = section.radius_z * fade;
                if rx < 0.5 || rz < 0.5 { continue; }

                let center = start + bone_dir * t
                    + bone_right * section.offset.x
                    + bone_up * section.offset.y;

                let ri = rx.max(rz).ceil() as i32;
                let n = section.n;

                for dx in -ri..=ri {
                    for dz in -ri..=ri {
                        // Superellipse: |x/rx|^n + |z/rz|^n <= 1
                        let ex = (dx as f32 / rx).abs();
                        let ez = (dz as f32 / rz).abs();
                        let dist = ex.powf(n) + ez.powf(n);
                        if dist > 1.0 { continue; }

                        if self.hollow && rx > 3.0 && rz > 3.0 {
                            let shell_thickness = 2.5 / rx.min(rz);
                            let inner = 1.0 - shell_thickness;
                            if inner > 0.1 && dist < inner.powf(n) { continue; }
                        }

                        let world_pos = center
                            + bone_right * dx as f32
                            + bone_up * dz as f32;

                        let x = world_pos.x as usize;
                        let y = world_pos.y as usize;
                        let z = world_pos.z as usize;

                        if x < grid_size && y < grid_size && z < grid_size {
                            set_voxel(x, y, z, profile.material_id,
                                profile.color[0], profile.color[1], profile.color[2]);
                        }
                    }
                }
            }
        }

        // Second: decals (eyes, nose, whiskers, visor, etc.) — drawn ON TOP of body
        for decal in &self.decals {
            let bone = skeleton.bone_by_id(decal.bone_id);
            let world_pos = bone.world_position + bone.world_rotation * decal.local_pos;

            match &decal.shape {
                DecalShape::Point => {
                    let (x,y,z) = (world_pos.x as usize, world_pos.y as usize, world_pos.z as usize);
                    if x < grid_size && y < grid_size && z < grid_size {
                        set_voxel(x, y, z, decal.material, decal.color[0], decal.color[1], decal.color[2]);
                    }
                }
                DecalShape::Sphere(r) => {
                    let ri = r.ceil() as i32;
                    let r2 = r * r;
                    for dz in -ri..=ri { for dy in -ri..=ri { for dx in -ri..=ri {
                        if (dx*dx+dy*dy+dz*dz) as f32 <= r2 {
                            let x = (world_pos.x as i32 + dx) as usize;
                            let y = (world_pos.y as i32 + dy) as usize;
                            let z = (world_pos.z as i32 + dz) as usize;
                            if x < grid_size && y < grid_size && z < grid_size {
                                set_voxel(x, y, z, decal.material, decal.color[0], decal.color[1], decal.color[2]);
                            }
                        }
                    }}}
                }
                DecalShape::LineH(w) => {
                    let hw = (*w / 2.0) as i32;
                    for dx in -hw..=hw {
                        let x = (world_pos.x as i32 + dx) as usize;
                        let (y,z) = (world_pos.y as usize, world_pos.z as usize);
                        if x < grid_size && y < grid_size && z < grid_size {
                            set_voxel(x, y, z, decal.material, decal.color[0], decal.color[1], decal.color[2]);
                        }
                    }
                }
                DecalShape::LineV(h) => {
                    let hh = (*h / 2.0) as i32;
                    for dy in -hh..=hh {
                        let (x,z) = (world_pos.x as usize, world_pos.z as usize);
                        let y = (world_pos.y as i32 + dy) as usize;
                        if x < grid_size && y < grid_size && z < grid_size {
                            set_voxel(x, y, z, decal.material, decal.color[0], decal.color[1], decal.color[2]);
                        }
                    }
                }
                DecalShape::Ellipse(rx, rz) => {
                    let ri = rx.max(*rz).ceil() as i32;
                    for dx in -ri..=ri { for dz in -ri..=ri {
                        let ex = dx as f32 / rx;
                        let ez = dz as f32 / rz;
                        if ex*ex + ez*ez <= 1.0 {
                            let x = (world_pos.x as i32 + dx) as usize;
                            let y = world_pos.y as usize;
                            let z = (world_pos.z as i32 + dz) as usize;
                            if x < grid_size && y < grid_size && z < grid_size {
                                set_voxel(x, y, z, decal.material, decal.color[0], decal.color[1], decal.color[2]);
                            }
                        }
                    }}
                }
            }
        }
    }
}

/// Build two perpendicular vectors to a given forward direction
fn build_perpendicular_frame(forward: Vec3) -> (Vec3, Vec3) {
    // Choose an "up" hint that isn't parallel to forward
    let up_hint = if forward.y.abs() > 0.9 { Vec3::Z } else { Vec3::Y };
    let right = forward.cross(up_hint).normalize();
    let up = right.cross(forward).normalize();
    (right, up)
}

// ═══════════════════════════════════════════════════════════════
// PREFAB BODY DEFINITIONS
// ═══════════════════════════════════════════════════════════════

impl BodyDefinition {
    /// Phase 1: Bare skeleton — bones only, no soft tissue
    /// Skull, ribcage, pelvis, long bones with epiphyses
    pub fn human_skeleton_only(skeleton: &Skeleton, scale: f32) -> Self {
        let mut body = BodyDefinition::new();
        let s = scale;
        let bone_c: [u8; 3] = [235, 225, 200]; // ivory bone color

        // ── SKULL — from reference: elongated back, flat face, bulging occiput ──
        // Key: rz > rx at cranium level, center shifts BACKWARD at t=0.5-0.7
        body.add(BoneProfile::elliptical(skeleton.bone("head").id, vec![
            BodySection { t: 0.0,  radius_x: 2.0*s, radius_z: 2.0*s, offset: Vec2::ZERO, n: 2.0 },             // chin (narrow V)
            BodySection { t: 0.1,  radius_x: 3.5*s, radius_z: 2.5*s, offset: Vec2::ZERO, n: 2.3 },             // mandible (wide, shallow, angular)
            BodySection { t: 0.25, radius_x: 4.5*s, radius_z: 3.5*s, offset: Vec2::new(0.0, 0.3*s), n: 2.0 },  // zygomatic/cheekbones (face fwd)
            BodySection { t: 0.35, radius_x: 4.5*s, radius_z: 4.0*s, offset: Vec2::new(0.0, 0.2*s), n: 2.0 },  // orbit level
            BodySection { t: 0.50, radius_x: 4.3*s, radius_z: 5.0*s, offset: Vec2::new(0.0, -0.5*s), n: 2.0 }, // temporal (rz > rx! center back)
            BodySection { t: 0.65, radius_x: 4.2*s, radius_z: 5.5*s, offset: Vec2::new(0.0, -1.0*s), n: 2.0 }, // parietal MAX (occiput bulge)
            BodySection { t: 0.80, radius_x: 3.8*s, radius_z: 4.5*s, offset: Vec2::new(0.0, -0.5*s), n: 2.0 }, // upper parietal
            BodySection { t: 0.95, radius_x: 2.5*s, radius_z: 3.0*s, offset: Vec2::ZERO, n: 2.0 },             // crown taper
            BodySection { t: 1.0,  radius_x: 2.0*s, radius_z: 2.5*s, offset: Vec2::ZERO, n: 2.0 },             // top
        ], 10, bone_c));

        // ── CERVICAL SPINE — thin vertebral column ──
        body.add(BoneProfile::elliptical(skeleton.bone("neck").id, vec![
            BodySection { t: 0.0, radius_x: 2.0*s, radius_z: 1.8*s, offset: Vec2::ZERO, n: 2.0 },
            BodySection { t: 0.5, radius_x: 1.8*s, radius_z: 1.5*s, offset: Vec2::ZERO, n: 2.0 },
            BodySection { t: 1.0, radius_x: 1.8*s, radius_z: 1.8*s, offset: Vec2::ZERO, n: 2.0 },
        ], 10, bone_c));

        // ── RIBCAGE (chest) — large barrel ──
        body.add(BoneProfile::elliptical(skeleton.bone("chest").id, vec![
            BodySection { t: 0.0, radius_x: 5.5*s, radius_z: 4.0*s, offset: Vec2::ZERO, n: 2.3 },
            BodySection { t: 0.3, radius_x: 7.5*s, radius_z: 5.0*s, offset: Vec2::ZERO, n: 2.3 },
            BodySection { t: 0.5, radius_x: 8.0*s, radius_z: 5.5*s, offset: Vec2::ZERO, n: 2.3 },  // max
            BodySection { t: 0.7, radius_x: 7.5*s, radius_z: 5.0*s, offset: Vec2::ZERO, n: 2.3 },
            BodySection { t: 1.0, radius_x: 5.5*s, radius_z: 4.0*s, offset: Vec2::ZERO, n: 2.0 },
        ], 10, bone_c));

        // ── LUMBAR SPINE — thin vertebral column ──
        body.add(BoneProfile::elliptical(skeleton.bone("spine").id, vec![
            BodySection { t: 0.0, radius_x: 2.0*s, radius_z: 1.8*s, offset: Vec2::ZERO, n: 2.0 },
            BodySection { t: 0.5, radius_x: 1.8*s, radius_z: 1.5*s, offset: Vec2::ZERO, n: 2.0 },
            BodySection { t: 1.0, radius_x: 1.8*s, radius_z: 1.5*s, offset: Vec2::ZERO, n: 2.0 },
        ], 10, bone_c));

        // ── PELVIS — LARGE butterfly, widest bone ──
        body.add(BoneProfile::elliptical(skeleton.bone("pelvis").id, vec![
            BodySection { t: 0.0, radius_x: 8.0*s, radius_z: 4.5*s, offset: Vec2::ZERO, n: 2.5 },
        ], 10, bone_c));

        // ── CLAVICLES — visible S-shaped rods ──
        for side in ["shoulder_l", "shoulder_r"] {
            body.add(BoneProfile::elliptical(skeleton.bone(side).id, vec![
                BodySection { t: 0.0, radius_x: 2.0*s, radius_z: 1.8*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 0.5, radius_x: 1.8*s, radius_z: 1.5*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 1.0, radius_x: 2.5*s, radius_z: 2.0*s, offset: Vec2::ZERO, n: 2.0 },
            ], 10, bone_c));
        }

        // ── HUMERUS — thick bone with ball joints ──
        for side in ["upper_arm_l", "upper_arm_r"] {
            body.add(BoneProfile::elliptical(skeleton.bone(side).id, vec![
                BodySection { t: 0.0,  radius_x: 3.0*s, radius_z: 3.0*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 0.1,  radius_x: 2.5*s, radius_z: 2.5*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 0.5,  radius_x: 2.0*s, radius_z: 2.0*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 0.9,  radius_x: 2.5*s, radius_z: 2.0*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 1.0,  radius_x: 3.0*s, radius_z: 2.5*s, offset: Vec2::ZERO, n: 2.0 },
            ], 10, bone_c));
        }

        // ── RADIUS/ULNA — two bones as one thicker profile ──
        for side in ["forearm_l", "forearm_r"] {
            body.add(BoneProfile::elliptical(skeleton.bone(side).id, vec![
                BodySection { t: 0.0, radius_x: 2.5*s, radius_z: 2.0*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 0.2, radius_x: 2.2*s, radius_z: 1.8*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 0.5, radius_x: 2.0*s, radius_z: 1.5*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 1.0, radius_x: 2.0*s, radius_z: 1.5*s, offset: Vec2::ZERO, n: 2.0 },
            ], 10, bone_c));
        }

        // ── HAND BONES ──
        for side in ["hand_l", "hand_r"] {
            body.add(BoneProfile::elliptical(skeleton.bone(side).id, vec![
                BodySection { t: 0.0, radius_x: 2.0*s, radius_z: 1.2*s, offset: Vec2::ZERO, n: 2.5 },
                BodySection { t: 0.5, radius_x: 2.5*s, radius_z: 1.2*s, offset: Vec2::ZERO, n: 2.5 },
                BodySection { t: 1.0, radius_x: 1.5*s, radius_z: 0.8*s, offset: Vec2::ZERO, n: 2.0 },
            ], 10, bone_c));
        }

        // ── HIP BONES — large iliac wings ──
        for side in ["hip_l", "hip_r"] {
            body.add(BoneProfile::elliptical(skeleton.bone(side).id, vec![
                BodySection { t: 0.0, radius_x: 4.0*s, radius_z: 3.0*s, offset: Vec2::ZERO, n: 2.5 },
                BodySection { t: 1.0, radius_x: 3.0*s, radius_z: 3.0*s, offset: Vec2::ZERO, n: 2.0 },
            ], 10, bone_c));
        }

        // ── FEMUR — thick, largest long bone ──
        for side in ["thigh_l", "thigh_r"] {
            body.add(BoneProfile::elliptical(skeleton.bone(side).id, vec![
                BodySection { t: 0.0,  radius_x: 3.5*s, radius_z: 3.5*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 0.1,  radius_x: 2.8*s, radius_z: 2.8*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 0.5,  radius_x: 2.2*s, radius_z: 2.2*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 0.9,  radius_x: 2.8*s, radius_z: 2.5*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 1.0,  radius_x: 3.5*s, radius_z: 3.0*s, offset: Vec2::ZERO, n: 2.0 },
            ], 10, bone_c));
        }

        // ── TIBIA/FIBULA — visible, solid ──
        for side in ["shin_l", "shin_r"] {
            body.add(BoneProfile::elliptical(skeleton.bone(side).id, vec![
                BodySection { t: 0.0,  radius_x: 3.0*s, radius_z: 2.8*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 0.15, radius_x: 2.5*s, radius_z: 2.0*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 0.5,  radius_x: 2.0*s, radius_z: 1.8*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 0.9,  radius_x: 2.0*s, radius_z: 1.5*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 1.0,  radius_x: 2.5*s, radius_z: 2.0*s, offset: Vec2::ZERO, n: 2.0 },
            ], 10, bone_c));
        }

        // ── FOOT BONES ──
        for side in ["foot_l", "foot_r"] {
            body.add(BoneProfile::elliptical(skeleton.bone(side).id, vec![
                BodySection { t: 0.0, radius_x: 2.5*s, radius_z: 2.5*s, offset: Vec2::ZERO, n: 2.5 },
                BodySection { t: 0.3, radius_x: 3.5*s, radius_z: 2.0*s, offset: Vec2::ZERO, n: 2.5 },
                BodySection { t: 0.7, radius_x: 3.5*s, radius_z: 1.5*s, offset: Vec2::ZERO, n: 3.0 },
                BodySection { t: 1.0, radius_x: 2.5*s, radius_z: 1.0*s, offset: Vec2::ZERO, n: 2.0 },
            ], 10, bone_c));
        }

        // Eye sockets + nasal aperture
        let head_id = skeleton.bone("head").id;
        body.add_decal(head_id, Vec3::new(-1.5*s, 2.5*s, -4.0*s), DecalShape::LineH(2.0*s), 1, [40, 35, 30]);
        body.add_decal(head_id, Vec3::new(1.5*s, 2.5*s, -4.0*s), DecalShape::LineH(2.0*s), 1, [40, 35, 30]);
        body.add_decal(head_id, Vec3::new(0.0, 1.5*s, -4.0*s), DecalShape::LineH(1.0*s), 1, [50, 45, 40]); // nose hole

        body
    }

    /// Phase 2: Bare body — soft tissue over skeleton (Barbie anatomy)
    /// To be built after skeleton proportions are approved
    pub fn human_soldier(skeleton: &Skeleton, scale: f32) -> Self {
        let mut body = BodyDefinition::new();
        let s = scale;

        let skin: [u8; 3] = [235, 200, 170];
        let pants: [u8; 3] = [95, 100, 110];
        let coat: [u8; 3] = [130, 135, 150];
        let boot: [u8; 3] = [80, 68, 55];
        let belt_c: [u8; 3] = [150, 130, 95];

        // ── HEAD — egg shape, bare (no helmet for now) ──
        body.add(BoneProfile::elliptical(skeleton.bone("head").id, vec![
            BodySection { t: 0.0,  radius_x: 3.0*s, radius_z: 3.0*s, offset: Vec2::ZERO, n: 2.0 },  // chin
            BodySection { t: 0.15, radius_x: 4.5*s, radius_z: 4.0*s, offset: Vec2::ZERO, n: 2.0 },  // jaw
            BodySection { t: 0.3,  radius_x: 5.5*s, radius_z: 5.5*s, offset: Vec2::ZERO, n: 2.0 },  // cheekbones
            BodySection { t: 0.5,  radius_x: 6.0*s, radius_z: 6.0*s, offset: Vec2::ZERO, n: 2.0 },  // temples
            BodySection { t: 0.7,  radius_x: 5.8*s, radius_z: 6.5*s, offset: Vec2::new(0.0, -0.5*s), n: 2.0 }, // occiput
            BodySection { t: 0.9,  radius_x: 4.5*s, radius_z: 5.0*s, offset: Vec2::ZERO, n: 2.0 },  // crown taper
            BodySection { t: 1.0,  radius_x: 3.0*s, radius_z: 3.5*s, offset: Vec2::ZERO, n: 2.0 },  // top
        ], 10, skin));

        // ── NECK — trapezoid, wider at base ──
        body.add(BoneProfile::elliptical(skeleton.bone("neck").id, vec![
            BodySection { t: 0.0, radius_x: 4.0*s, radius_z: 3.5*s, offset: Vec2::ZERO, n: 2.0 },
            BodySection { t: 0.5, radius_x: 3.5*s, radius_z: 3.0*s, offset: Vec2::ZERO, n: 2.0 },
            BodySection { t: 1.0, radius_x: 3.0*s, radius_z: 3.0*s, offset: Vec2::ZERO, n: 2.0 },
        ], 10, skin));

        // ── CHEST — barrel, light coat, superellipse n=2.3 ──
        body.add(BoneProfile::elliptical(skeleton.bone("chest").id, vec![
            BodySection { t: 0.0, radius_x: 7.0*s, radius_z: 5.0*s, offset: Vec2::ZERO, n: 2.3 },
            BodySection { t: 0.3, radius_x: 8.5*s, radius_z: 5.5*s, offset: Vec2::new(0.0, 0.5*s), n: 2.3 },
            BodySection { t: 0.5, radius_x: 9.0*s, radius_z: 6.0*s, offset: Vec2::new(0.0, 0.5*s), n: 2.3 }, // max
            BodySection { t: 0.7, radius_x: 8.5*s, radius_z: 5.5*s, offset: Vec2::ZERO, n: 2.3 },
            BodySection { t: 1.0, radius_x: 7.0*s, radius_z: 5.0*s, offset: Vec2::ZERO, n: 2.0 },
        ], 5, coat));

        // ── SPINE — waist! The key anatomical landmark ──
        body.add(BoneProfile::elliptical(skeleton.bone("spine").id, vec![
            BodySection { t: 0.0, radius_x: 8.0*s, radius_z: 5.0*s, offset: Vec2::ZERO, n: 2.3 },
            BodySection { t: 0.3, radius_x: 7.0*s, radius_z: 4.5*s, offset: Vec2::ZERO, n: 2.0 },
            BodySection { t: 0.7, radius_x: 6.5*s, radius_z: 4.0*s, offset: Vec2::ZERO, n: 2.0 },  // WAIST
            BodySection { t: 1.0, radius_x: 7.0*s, radius_z: 4.5*s, offset: Vec2::ZERO, n: 2.3 },
        ], 5, coat));

        // ── PELVIS — belt ──
        body.add(BoneProfile::elliptical(skeleton.bone("pelvis").id, vec![
            BodySection { t: 0.0, radius_x: 8.5*s, radius_z: 5.5*s, offset: Vec2::ZERO, n: 2.3 },
        ], 10, belt_c));

        // ── SHOULDERS — barrel/olive shape (deltoid muscle) ──
        for side in ["shoulder_l", "shoulder_r"] {
            body.add(BoneProfile::elliptical(skeleton.bone(side).id, vec![
                BodySection { t: 0.0, radius_x: 4.0*s, radius_z: 3.5*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 0.5, radius_x: 5.0*s, radius_z: 4.5*s, offset: Vec2::ZERO, n: 2.0 }, // deltoid peak
                BodySection { t: 1.0, radius_x: 4.0*s, radius_z: 3.5*s, offset: Vec2::ZERO, n: 2.0 },
            ], 5, coat));
        }

        // ── UPPER ARMS — spindle shape (bicep peak at t=0.2, thin elbow) ──
        for side in ["upper_arm_l", "upper_arm_r"] {
            body.add(BoneProfile::elliptical(skeleton.bone(side).id, vec![
                BodySection { t: 0.0, radius_x: 3.5*s, radius_z: 3.5*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 0.2, radius_x: 4.0*s, radius_z: 3.5*s, offset: Vec2::ZERO, n: 2.0 }, // bicep
                BodySection { t: 0.5, radius_x: 3.5*s, radius_z: 3.5*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 0.8, radius_x: 3.0*s, radius_z: 3.0*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 1.0, radius_x: 2.5*s, radius_z: 2.5*s, offset: Vec2::ZERO, n: 2.0 }, // elbow
            ], 5, coat));
        }

        // ── FOREARMS — spindle (muscle at top, thin wrist) ──
        for side in ["forearm_l", "forearm_r"] {
            body.add(BoneProfile::elliptical(skeleton.bone(side).id, vec![
                BodySection { t: 0.0, radius_x: 3.0*s, radius_z: 2.5*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 0.2, radius_x: 3.5*s, radius_z: 3.0*s, offset: Vec2::ZERO, n: 2.0 }, // forearm muscle
                BodySection { t: 0.5, radius_x: 3.0*s, radius_z: 2.5*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 1.0, radius_x: 2.5*s, radius_z: 2.5*s, offset: Vec2::ZERO, n: 2.0 }, // wrist (gloved)
            ], 5, coat));
        }

        // ── HANDS — flat paddle, fist in glove ──
        for side in ["hand_l", "hand_r"] {
            body.add(BoneProfile::elliptical(skeleton.bone(side).id, vec![
                BodySection { t: 0.0, radius_x: 2.5*s, radius_z: 2.5*s, offset: Vec2::ZERO, n: 2.5 },
                BodySection { t: 0.5, radius_x: 2.5*s, radius_z: 2.5*s, offset: Vec2::ZERO, n: 2.5 },
                BodySection { t: 1.0, radius_x: 1.5*s, radius_z: 1.5*s, offset: Vec2::ZERO, n: 2.0 },
            ], 10, skin));
        }

        // ── HIPS — transition from pelvis to thigh ──
        for side in ["hip_l", "hip_r"] {
            body.add(BoneProfile::elliptical(skeleton.bone(side).id, vec![
                BodySection { t: 0.0, radius_x: 4.0*s, radius_z: 3.0*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 1.0, radius_x: 5.0*s, radius_z: 5.0*s, offset: Vec2::ZERO, n: 2.0 },
            ], 5, pants));
        }

        // ── THIGHS — powerful spindle (quadriceps peak at t=0.2, thin knee) ──
        for side in ["thigh_l", "thigh_r"] {
            body.add(BoneProfile::elliptical(skeleton.bone(side).id, vec![
                BodySection { t: 0.0, radius_x: 5.5*s, radius_z: 5.5*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 0.2, radius_x: 6.0*s, radius_z: 5.5*s, offset: Vec2::new(0.0, 0.5*s), n: 2.0 }, // quad
                BodySection { t: 0.5, radius_x: 5.0*s, radius_z: 5.0*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 0.8, radius_x: 4.0*s, radius_z: 4.0*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 1.0, radius_x: 3.5*s, radius_z: 3.5*s, offset: Vec2::ZERO, n: 2.0 }, // knee
            ], 5, pants));
        }

        // ── SHINS — calf spindle (peak at t=0.15, thin ankle + boots) ──
        for side in ["shin_l", "shin_r"] {
            body.add(BoneProfile::elliptical(skeleton.bone(side).id, vec![
                BodySection { t: 0.0,  radius_x: 3.5*s, radius_z: 3.5*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 0.15, radius_x: 4.0*s, radius_z: 3.5*s, offset: Vec2::new(0.0, -1.0*s), n: 2.0 }, // calf (back!)
                BodySection { t: 0.3,  radius_x: 3.5*s, radius_z: 3.5*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 0.7,  radius_x: 3.5*s, radius_z: 3.5*s, offset: Vec2::ZERO, n: 2.0 }, // boots thicken
                BodySection { t: 1.0,  radius_x: 3.5*s, radius_z: 3.5*s, offset: Vec2::ZERO, n: 2.0 }, // boot ankle
            ], 5, pants));
        }

        // ── FEET — flat platform with boots ──
        for side in ["foot_l", "foot_r"] {
            body.add(BoneProfile::elliptical(skeleton.bone(side).id, vec![
                BodySection { t: 0.0, radius_x: 3.5*s, radius_z: 3.5*s, offset: Vec2::ZERO, n: 3.0 },
                BodySection { t: 0.3, radius_x: 4.5*s, radius_z: 4.0*s, offset: Vec2::ZERO, n: 3.0 },
                BodySection { t: 0.7, radius_x: 4.5*s, radius_z: 3.0*s, offset: Vec2::ZERO, n: 3.0 },
                BodySection { t: 1.0, radius_x: 3.5*s, radius_z: 2.0*s, offset: Vec2::ZERO, n: 2.5 },
            ], 10, boot));
        }

        // ── Face decals (adjusted for new head size) ──
        let head_id = skeleton.bone("head").id;
        // Visor at eye level (t≈0.35, so local Y ≈ 4.5*s from chin)
        body.add_decal(head_id, Vec3::new(0.0, 4.5*s, -7.0*s), DecalShape::LineH(10.0*s), 2, [30, 250, 255]);
        body.add_decal(head_id, Vec3::new(0.0, 5.0*s, -7.0*s), DecalShape::LineH(10.0*s), 2, [30, 250, 255]);

        body
    }

    /// Cat body
    pub fn cat_body(skeleton: &Skeleton, scale: f32) -> Self {
        let mut body = BodyDefinition::new();
        let s = scale;
        let fur: [u8; 3] = [200, 140, 60]; // orange tabby

        // Head (round)
        body.add(BoneProfile::cylinder(skeleton.bone("head").id, 4.0*s, 13, fur));

        // Ears (small cones)
        for ear in ["ear_l", "ear_r"] {
            body.add(BoneProfile::tapered(skeleton.bone(ear).id, 1.5*s, 0.5*s, 13, fur));
        }

        // Neck
        body.add(BoneProfile::tapered(skeleton.bone("neck").id, 3.0*s, 3.5*s, 13, fur));

        // Spine segments (body)
        for sp in ["spine2", "spine1"] {
            body.add(BoneProfile::elliptical(skeleton.bone(sp).id, vec![
                BodySection { t: 0.0, radius_x: 4.0*s, radius_z: 3.5*s, offset: Vec2::ZERO, n: 2.0 },
                BodySection { t: 1.0, radius_x: 4.5*s, radius_z: 4.0*s, offset: Vec2::ZERO, n: 2.0 },
            ], 13, fur));
        }

        // Pelvis
        body.add(BoneProfile::cylinder(skeleton.bone("pelvis").id, 4.0*s, 13, fur));

        // Front legs
        for (upper, lower, paw) in [("upper_arm_l","forearm_l","paw_fl"), ("upper_arm_r","forearm_r","paw_fr")] {
            body.add(BoneProfile::tapered(skeleton.bone(upper).id, 2.5*s, 2.0*s, 13, fur));
            body.add(BoneProfile::tapered(skeleton.bone(lower).id, 2.0*s, 1.8*s, 13, fur));
            body.add(BoneProfile::cylinder(skeleton.bone(paw).id, 2.0*s, 13, fur));
        }

        // Back legs
        for (thigh, shin, paw) in [("thigh_l","shin_l","paw_bl"), ("thigh_r","shin_r","paw_br")] {
            body.add(BoneProfile::tapered(skeleton.bone(thigh).id, 3.0*s, 2.5*s, 13, fur));
            body.add(BoneProfile::tapered(skeleton.bone(shin).id, 2.5*s, 2.0*s, 13, fur));
            body.add(BoneProfile::cylinder(skeleton.bone(paw).id, 2.0*s, 13, fur));
        }

        // Tail segments (decreasing)
        for (i, name) in ["tail1","tail2","tail3","tail4"].iter().enumerate() {
            let r = (2.5 - i as f32 * 0.5) * s;
            body.add(BoneProfile::tapered(skeleton.bone(name).id, r, r * 0.7, 13, fur));
        }

        // Cat face — decals near the END of head bone (snout tip)
        // Head bone: length=5*s, direction ~(0, 0.2, 0.8). Snout at ~80% = 4*s along bone.
        let head_id = skeleton.bone("head").id;
        let snout = 4.0; // distance along head bone to snout
        // Eyes (green, almond-shaped — two spheres side by side)
        body.add_decal(head_id, Vec3::new(-1.0*s, 0.8*s, snout*s), DecalShape::Sphere(0.5*s), 1, [80, 210, 60]);
        body.add_decal(head_id, Vec3::new(1.0*s, 0.8*s, snout*s), DecalShape::Sphere(0.5*s), 1, [80, 210, 60]);
        // Pupils (vertical slit — cat eyes!)
        body.add_decal(head_id, Vec3::new(-1.0*s, 0.8*s, (snout+0.3)*s), DecalShape::Point, 1, [5, 5, 5]);
        body.add_decal(head_id, Vec3::new(1.0*s, 0.8*s, (snout+0.3)*s), DecalShape::Point, 1, [5, 5, 5]);
        // Nose (pink triangle at tip)
        body.add_decal(head_id, Vec3::new(0.0, 0.1*s, (snout+0.5)*s), DecalShape::Sphere(0.25*s), 1, [255, 150, 150]);
        // Whiskers (3 per side, from nose area)
        for &(dy, len) in &[(-0.1, 2.5), (0.3, 2.0), (-0.4, 1.8)] {
            body.add_decal(head_id, Vec3::new(-1.8*s, dy*s, snout*s), DecalShape::LineH(len*s), 1, [230, 230, 230]);
            body.add_decal(head_id, Vec3::new(1.8*s, dy*s, snout*s), DecalShape::LineH(len*s), 1, [230, 230, 230]);
        }
        // Mouth (small inverted Y under nose)
        body.add_decal(head_id, Vec3::new(0.0, -0.3*s, snout*s), DecalShape::LineH(0.6*s), 1, [170, 90, 70]);

        body
    }

    /// Simple dog body
    pub fn dog_body(skeleton: &Skeleton, scale: f32) -> Self {
        let s = scale;
        let brown: [u8; 3] = [140, 100, 60];
        let mut body = BodyDefinition::new();

        let mk = |bid: BoneId, secs: Vec<BodySection>| -> BoneProfile {
            BoneProfile { bone_id: bid, sections: secs, material_id: 10, color: brown }
        };
        let sec = |t: f32, rx: f32, rz: f32| -> BodySection {
            BodySection { t, radius_x: rx, radius_z: rz, offset: Vec2::ZERO, n: 2.0 }
        };

        body.add(mk(skeleton.bone("spine1").id, vec![sec(0.0,4.0*s,3.5*s), sec(0.5,5.0*s,4.0*s), sec(1.0,4.5*s,3.5*s)]));
        body.add(mk(skeleton.bone("spine2").id, vec![sec(0.0,4.5*s,3.5*s), sec(0.5,4.0*s,3.5*s), sec(1.0,3.5*s,3.0*s)]));
        body.add(mk(skeleton.bone("neck").id, vec![sec(0.0,3.0*s,2.5*s), sec(1.0,2.5*s,2.0*s)]));
        body.add(mk(skeleton.bone("head").id, vec![sec(0.0,3.0*s,3.0*s), sec(0.5,3.5*s,3.0*s), sec(1.0,2.0*s,2.0*s)]));

        for suffix in ["_l", "_r"] {
            body.add(mk(skeleton.bone(&format!("upper_arm{}", suffix)).id, vec![sec(0.0,1.8*s,1.8*s), sec(1.0,1.2*s,1.2*s)]));
            body.add(mk(skeleton.bone(&format!("forearm{}", suffix)).id, vec![sec(0.0,1.2*s,1.2*s), sec(1.0,1.0*s,1.0*s)]));
            body.add(mk(skeleton.bone(&format!("thigh{}", suffix)).id, vec![sec(0.0,2.0*s,2.0*s), sec(1.0,1.3*s,1.3*s)]));
            body.add(mk(skeleton.bone(&format!("shin{}", suffix)).id, vec![sec(0.0,1.3*s,1.3*s), sec(1.0,1.0*s,1.0*s)]));
        }
        body.add(mk(skeleton.bone("tail1").id, vec![sec(0.0,1.2*s,1.2*s), sec(1.0,0.8*s,0.8*s)]));
        body.add(mk(skeleton.bone("tail2").id, vec![sec(0.0,0.8*s,0.8*s), sec(1.0,0.5*s,0.5*s)]));
        body
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_section_interpolation() {
        let profile = BoneProfile::elliptical(0, vec![
            BodySection { t: 0.0, radius_x: 10.0, radius_z: 5.0, offset: Vec2::ZERO, n: 2.0 },
            BodySection { t: 1.0, radius_x: 6.0, radius_z: 3.0, offset: Vec2::ZERO, n: 2.0 },
        ], 1, [255,255,255]);

        let mid = profile.section_at(0.5);
        assert!((mid.radius_x - 8.0).abs() < 0.01);
        assert!((mid.radius_z - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_rasterize_produces_voxels() {
        let mut sk = Skeleton::human(1.0);
        sk.root_position = Vec3::new(32.0, 20.0, 32.0);
        sk.solve_forward();

        let body = BodyDefinition::human_soldier(&sk, 1.0);
        let mut voxel_count = 0;
        body.rasterize(&sk, 64, |_x, _y, _z, _m, _r, _g, _b| {
            voxel_count += 1;
        });

        println!("Rasterized {} voxels", voxel_count);
        assert!(voxel_count > 1000); // should produce many voxels
    }

    #[test]
    fn test_cat_body() {
        let mut sk = Skeleton::cat(1.0);
        sk.root_position = Vec3::new(32.0, 15.0, 32.0);
        sk.solve_forward();

        let body = BodyDefinition::cat_body(&sk, 1.0);
        let mut voxel_count = 0;
        body.rasterize(&sk, 64, |_x, _y, _z, _m, _r, _g, _b| {
            voxel_count += 1;
        });

        println!("Cat: {} voxels", voxel_count);
        assert!(voxel_count > 500);
    }
}
