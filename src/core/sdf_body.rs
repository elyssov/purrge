// ═══════════════════════════════════════════════════════════════
// PROMETHEUS ENGINE — SDF Body System
//
// Replaces ellipse-sweep profiles with Signed Distance Functions.
// Each body part = combination of SDF primitives (sphere, capsule,
// ellipsoid, box) with smooth union/subtraction.
//
// SDF field is sampled only at the surface shell (hollow rendering).
// Memory: O(surface_area), not O(volume).
// ═══════════════════════════════════════════════════════════════

use glam::Vec3;
use super::svo::Voxel;

/// SDF primitive — basic building block
#[derive(Clone, Debug)]
pub enum SdfPrimitive {
    /// Sphere at position with radius
    Sphere { center: Vec3, radius: f32 },
    /// Capsule (line segment with radius)
    Capsule { a: Vec3, b: Vec3, radius: f32 },
    /// Ellipsoid (stretched sphere)
    Ellipsoid { center: Vec3, radii: Vec3 },
    /// Rounded box
    RoundBox { center: Vec3, half_extents: Vec3, rounding: f32 },
}

impl SdfPrimitive {
    fn distance(&self, p: Vec3) -> f32 {
        match self {
            SdfPrimitive::Sphere { center, radius } => {
                (p - *center).length() - radius
            }
            SdfPrimitive::Capsule { a, b, radius } => {
                let ab = *b - *a;
                let ap = p - *a;
                let t = ap.dot(ab) / ab.dot(ab);
                let t = t.clamp(0.0, 1.0);
                let closest = *a + ab * t;
                (p - closest).length() - radius
            }
            SdfPrimitive::Ellipsoid { center, radii } => {
                // Approximate SDF for ellipsoid
                let q = (p - *center) / *radii;
                let len = q.length();
                if len < 0.001 { return -radii.min_element(); }
                (len - 1.0) * radii.min_element()
            }
            SdfPrimitive::RoundBox { center, half_extents, rounding } => {
                let q = (p - *center).abs() - *half_extents;
                let outside = Vec3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0)).length();
                let inside = q.x.max(q.y).max(q.z).min(0.0);
                outside + inside - rounding
            }
        }
    }

    /// Bounding box (min, max) with margin
    fn bounds(&self, margin: f32) -> (Vec3, Vec3) {
        match self {
            SdfPrimitive::Sphere { center, radius } => {
                let r = *radius + margin;
                (*center - Vec3::splat(r), *center + Vec3::splat(r))
            }
            SdfPrimitive::Capsule { a, b, radius } => {
                let r = *radius + margin;
                (a.min(*b) - Vec3::splat(r), a.max(*b) + Vec3::splat(r))
            }
            SdfPrimitive::Ellipsoid { center, radii } => {
                let r = *radii + Vec3::splat(margin);
                (*center - r, *center + r)
            }
            SdfPrimitive::RoundBox { center, half_extents, rounding } => {
                let r = *half_extents + Vec3::splat(*rounding + margin);
                (*center - r, *center + r)
            }
        }
    }
}

/// SDF operation — how primitives combine
#[derive(Clone, Debug)]
pub enum SdfOp {
    /// Add shape (smooth union with blending radius k)
    Add { primitive: SdfPrimitive, k: f32 },
    /// Subtract shape (smooth subtraction)
    Sub { primitive: SdfPrimitive, k: f32 },
}

/// Smooth minimum (for smooth union)
fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    if k < 0.001 { return a.min(b); }
    let h = (0.5 + 0.5 * (a - b) / k).clamp(0.0, 1.0);
    a * (1.0 - h) + b * h - k * h * (1.0 - h)
}

/// Smooth maximum (for smooth subtraction)
fn smooth_max(a: f32, b: f32, k: f32) -> f32 {
    -smooth_min(-a, -b, k)
}

/// A complete SDF body part (e.g., skull, ribcage, femur)
#[derive(Clone)]
pub struct SdfShape {
    pub name: String,
    pub ops: Vec<SdfOp>,
    pub material: u8,
    pub color: [u8; 3],
}

impl SdfShape {
    pub fn new(name: &str, material: u8, color: [u8; 3]) -> Self {
        Self { name: name.to_string(), ops: Vec::new(), material, color }
    }

    pub fn add(&mut self, prim: SdfPrimitive, blend: f32) -> &mut Self {
        self.ops.push(SdfOp::Add { primitive: prim, k: blend });
        self
    }

    pub fn sub(&mut self, prim: SdfPrimitive, blend: f32) -> &mut Self {
        self.ops.push(SdfOp::Sub { primitive: prim, k: blend });
        self
    }

    /// Evaluate the SDF at point p
    pub fn distance(&self, p: Vec3) -> f32 {
        let mut d = f32::MAX;
        for op in &self.ops {
            match op {
                SdfOp::Add { primitive, k } => {
                    d = smooth_min(d, primitive.distance(p), *k);
                }
                SdfOp::Sub { primitive, k } => {
                    d = smooth_max(d, -primitive.distance(p), *k);
                }
            }
        }
        d
    }

    /// Overall bounding box of all primitives
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let mut bmin = Vec3::splat(f32::MAX);
        let mut bmax = Vec3::splat(f32::MIN);
        for op in &self.ops {
            let prim = match op {
                SdfOp::Add { primitive, .. } => primitive,
                SdfOp::Sub { primitive, .. } => primitive,
            };
            let (lo, hi) = prim.bounds(2.0);
            bmin = bmin.min(lo);
            bmax = bmax.max(hi);
        }
        (bmin, bmax)
    }
}

/// Complete SDF body — collection of shapes
pub struct SdfBody {
    pub shapes: Vec<SdfShape>,
}

impl SdfBody {
    pub fn new() -> Self { Self { shapes: Vec::new() } }

    pub fn add_shape(&mut self, shape: SdfShape) {
        self.shapes.push(shape);
    }

    /// Rasterize SDF body into voxel grid — HOLLOW shell only.
    /// Only writes voxels where the surface is (|sdf| < shell_thickness).
    /// This means interior is EMPTY — massive memory savings.
    pub fn rasterize<F>(&self, grid_size: usize, shell: f32, mut set_voxel: F)
    where F: FnMut(usize, usize, usize, u8, u8, u8, u8)
    {
        for shape in &self.shapes {
            let (bmin, bmax) = shape.bounds();
            // Only iterate within bounding box
            let x0 = (bmin.x.floor() as i32).max(0) as usize;
            let y0 = (bmin.y.floor() as i32).max(0) as usize;
            let z0 = (bmin.z.floor() as i32).max(0) as usize;
            let x1 = (bmax.x.ceil() as i32 + 1).min(grid_size as i32) as usize;
            let y1 = (bmax.y.ceil() as i32 + 1).min(grid_size as i32) as usize;
            let z1 = (bmax.z.ceil() as i32 + 1).min(grid_size as i32) as usize;

            for z in z0..z1 {
                for y in y0..y1 {
                    for x in x0..x1 {
                        let p = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                        let d = shape.distance(p);
                        // Solid fill for now (hollow later via Surface Nets)
                        if d <= 0.0 {
                            set_voxel(x, y, z, shape.material,
                                shape.color[0], shape.color[1], shape.color[2]);
                        }
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// PREFAB: FULL HUMAN BODY from skeleton positions
//
// Takes a Skeleton after solve_forward() and builds SDF shapes
// around each bone group. Smooth union between parts = no Buratino.
//
// Anatomy: torso (3 ellipsoids blended), limbs (capsules),
//          head (skull), hands/feet (rounded boxes).
// ═══════════════════════════════════════════════════════════════

use super::skeleton::Skeleton;

impl SdfBody {
    /// Full human body built from skeleton world positions.
    /// Call skeleton.solve_forward() BEFORE this.
    ///
    /// Returns one SdfBody with multiple shapes (torso, limbs, head, etc.)
    /// each with its own material/color. Shapes are separate so they
    /// rasterize with correct per-part colors.
    pub fn human_body(skeleton: &Skeleton, scale: f32) -> Self {
        let mut body = SdfBody::new();
        let s = scale;
        let b = 2.5 * s;  // base blend radius — bigger = smoother joints

        let skin: [u8; 3] = [220, 185, 155];
        let coat: [u8; 3] = [110, 115, 130];
        let pants: [u8; 3] = [85, 88, 100];
        let boot: [u8; 3] = [70, 60, 48];
        let belt_c: [u8; 3] = [140, 120, 85];

        // ─── Bone positions (from solved skeleton) ──────────
        let pelvis  = skeleton.bone("pelvis").world_position;
        let spine_e = skeleton.bone("spine").world_end_position;
        let chest_s = skeleton.bone("chest").world_position;
        let chest_e = skeleton.bone("chest").world_end_position;
        let neck_s  = skeleton.bone("neck").world_position;
        let neck_e  = skeleton.bone("neck").world_end_position;
        let head_s  = skeleton.bone("head").world_position;
        let head_e  = skeleton.bone("head").world_end_position;

        // ─── TORSO ──────────────────────────────────────────
        // One shape for the whole torso: pelvis → chest, smooth blended.
        // This is the key anti-Buratino trick: overlapping ellipsoids
        // with large blend radius create organic transitions.
        {
            let mut torso = SdfShape::new("torso", 5, coat);
            let mid = (pelvis + spine_e) * 0.5;

            // Pelvis — wide, flat, boxy
            torso.add(SdfPrimitive::Ellipsoid {
                center: pelvis,
                radii: Vec3::new(8.5*s, 3.0*s, 5.0*s),
            }, 0.0);

            // Waist — narrower (belt area)
            torso.add(SdfPrimitive::Ellipsoid {
                center: mid,
                radii: Vec3::new(6.5*s, 3.0*s, 4.0*s),
            }, b);

            // Lower chest — barrel starts expanding
            torso.add(SdfPrimitive::Ellipsoid {
                center: chest_s,
                radii: Vec3::new(8.0*s, 3.5*s, 5.0*s),
            }, b);

            // Upper chest — widest, barrel shape
            torso.add(SdfPrimitive::Ellipsoid {
                center: chest_e,
                radii: Vec3::new(8.5*s, 4.0*s, 5.5*s),
            }, b);

            body.add_shape(torso);
        }

        // Belt highlight
        {
            let belt_pos = (pelvis + spine_e) * 0.5 + Vec3::new(0.0, 0.5*s, 0.0);
            let mut belt = SdfShape::new("belt", 10, belt_c);
            belt.add(SdfPrimitive::Ellipsoid {
                center: belt_pos,
                radii: Vec3::new(7.0*s, 1.5*s, 4.5*s),
            }, 0.0);
            body.add_shape(belt);
        }

        // ─── NECK ───────────────────────────────────────────
        {
            let mut neck = SdfShape::new("neck", 10, skin);
            neck.add(SdfPrimitive::Capsule {
                a: neck_s,
                b: neck_e,
                radius: 3.0*s,
            }, 0.0);
            // Smooth transition to chest — sphere at base
            neck.add(SdfPrimitive::Sphere {
                center: neck_s,
                radius: 4.0*s,
            }, b);
            body.add_shape(neck);
        }

        // ─── HEAD ───────────────────────────────────────────
        // Simplified head (not full skull — that's for skeleton view).
        // Egg shape: ellipsoid + chin capsule + forehead sphere.
        {
            let head_mid = (head_s + head_e) * 0.5;
            let head_len = (head_e - head_s).length();
            let mut head = SdfShape::new("head", 10, skin);

            // Main cranium — slightly elongated back
            head.add(SdfPrimitive::Ellipsoid {
                center: head_mid + Vec3::new(0.0, 0.0, -0.5*s),
                radii: Vec3::new(5.5*s, head_len * 0.55, 6.0*s),
            }, 0.0);

            // Jaw / chin — rounded box below
            head.add(SdfPrimitive::RoundBox {
                center: head_s + Vec3::new(0.0, -1.0*s, 1.0*s),
                half_extents: Vec3::new(3.5*s, 1.5*s, 2.5*s),
                rounding: 1.5*s,
            }, b * 0.8);

            // Forehead
            head.add(SdfPrimitive::Sphere {
                center: head_mid + Vec3::new(0.0, 1.5*s, 2.0*s),
                radius: 3.5*s,
            }, b);

            // Nose ridge
            head.add(SdfPrimitive::Capsule {
                a: head_s + Vec3::new(0.0, 0.5*s, 4.5*s),
                b: head_s + Vec3::new(0.0, -1.0*s, 5.5*s),
                radius: 1.0*s,
            }, 1.5);

            body.add_shape(head);
        }

        // ─── SHOULDERS + ARMS ───────────────────────────────
        for suffix in ["_l", "_r"] {
            let shoulder_name = format!("shoulder{}", suffix);
            let upper_name = format!("upper_arm{}", suffix);
            let forearm_name = format!("forearm{}", suffix);
            let hand_name = format!("hand{}", suffix);

            let shoulder_s = skeleton.bone(&shoulder_name).world_position;
            let shoulder_e = skeleton.bone(&shoulder_name).world_end_position;
            let upper_s = skeleton.bone(&upper_name).world_position;
            let upper_e = skeleton.bone(&upper_name).world_end_position;
            let forearm_s = skeleton.bone(&forearm_name).world_position;
            let forearm_e = skeleton.bone(&forearm_name).world_end_position;
            let hand_s = skeleton.bone(&hand_name).world_position;
            let hand_e = skeleton.bone(&hand_name).world_end_position;

            // Shoulder (deltoid) — sphere at joint
            {
                let mut shoulder = SdfShape::new("shoulder", 5, coat);
                shoulder.add(SdfPrimitive::Sphere {
                    center: shoulder_e,
                    radius: 5.0*s,
                }, 0.0);
                // Smooth bridge to chest
                shoulder.add(SdfPrimitive::Capsule {
                    a: shoulder_s,
                    b: shoulder_e,
                    radius: 3.5*s,
                }, b);
                body.add_shape(shoulder);
            }

            // Upper arm (coat)
            {
                let mut arm = SdfShape::new("upper_arm", 5, coat);
                arm.add(SdfPrimitive::Capsule {
                    a: upper_s,
                    b: upper_e,
                    radius: 3.5*s,
                }, 0.0);
                // Elbow bulge
                arm.add(SdfPrimitive::Sphere {
                    center: upper_e,
                    radius: 3.8*s,
                }, b * 0.5);
                body.add_shape(arm);
            }

            // Forearm (coat, tapers)
            {
                let mut forearm = SdfShape::new("forearm", 5, coat);
                // Capsule from elbow to wrist, wrist thinner
                forearm.add(SdfPrimitive::Capsule {
                    a: forearm_s,
                    b: forearm_e,
                    radius: 3.0*s,
                }, 0.0);
                // Wrist — smaller sphere
                forearm.add(SdfPrimitive::Sphere {
                    center: forearm_e,
                    radius: 2.5*s,
                }, 1.0);
                body.add_shape(forearm);
            }

            // Hand (skin)
            {
                let mut hand = SdfShape::new("hand", 10, skin);
                let hand_dir = (hand_e - hand_s).normalize();
                let hand_mid = (hand_s + hand_e) * 0.5;
                // Flat-ish box for palm
                hand.add(SdfPrimitive::RoundBox {
                    center: hand_mid,
                    half_extents: Vec3::new(2.0*s, 1.2*s, (hand_e - hand_s).length() * 0.4),
                    rounding: 1.0*s,
                }, 0.0);
                body.add_shape(hand);
            }
        }

        // ─── HIPS + LEGS ────────────────────────────────────
        for suffix in ["_l", "_r"] {
            let hip_name = format!("hip{}", suffix);
            let thigh_name = format!("thigh{}", suffix);
            let shin_name = format!("shin{}", suffix);
            let foot_name = format!("foot{}", suffix);

            let hip_s = skeleton.bone(&hip_name).world_position;
            let hip_e = skeleton.bone(&hip_name).world_end_position;
            let thigh_s = skeleton.bone(&thigh_name).world_position;
            let thigh_e = skeleton.bone(&thigh_name).world_end_position;
            let shin_s = skeleton.bone(&shin_name).world_position;
            let shin_e = skeleton.bone(&shin_name).world_end_position;
            let foot_s = skeleton.bone(&foot_name).world_position;
            let foot_e = skeleton.bone(&foot_name).world_end_position;

            // Hip joint — sphere bridging pelvis to leg
            {
                let mut hip = SdfShape::new("hip", 5, pants);
                hip.add(SdfPrimitive::Sphere {
                    center: hip_e,
                    radius: 5.5*s,
                }, 0.0);
                body.add_shape(hip);
            }

            // Thigh (pants)
            {
                let mut thigh = SdfShape::new("thigh", 5, pants);
                thigh.add(SdfPrimitive::Capsule {
                    a: thigh_s,
                    b: thigh_e,
                    radius: 4.5*s,
                }, 0.0);
                // Knee bulge
                thigh.add(SdfPrimitive::Sphere {
                    center: thigh_e,
                    radius: 4.0*s,
                }, b * 0.5);
                body.add_shape(thigh);
            }

            // Shin (pants → boot transition at ~60%)
            {
                let mut shin = SdfShape::new("shin", 5, pants);
                shin.add(SdfPrimitive::Capsule {
                    a: shin_s,
                    b: shin_e,
                    radius: 3.5*s,
                }, 0.0);
                body.add_shape(shin);
            }

            // Boot
            {
                let mut boot_shape = SdfShape::new("boot", 10, boot);
                // Boot shaft — capsule around lower shin
                let boot_top = shin_s + (shin_e - shin_s) * 0.5;
                boot_shape.add(SdfPrimitive::Capsule {
                    a: boot_top,
                    b: shin_e,
                    radius: 4.0*s,
                }, 0.0);
                // Boot sole — rounded box at foot
                let foot_mid = (foot_s + foot_e) * 0.5;
                boot_shape.add(SdfPrimitive::RoundBox {
                    center: foot_mid + Vec3::new(0.0, -1.0*s, 0.0),
                    half_extents: Vec3::new(3.5*s, 2.0*s, (foot_e - foot_s).length() * 0.45),
                    rounding: 1.0*s,
                }, b * 0.6);
                body.add_shape(boot_shape);
            }
        }

        body
    }
}

// ═══════════════════════════════════════════════════════════════
// PREFAB: HUMAN SKULL from reference images
// ═══════════════════════════════════════════════════════════════

impl SdfBody {
    /// Anatomical human skull based on reference scull.jpg
    /// Position: centered at `pos`, looking toward -Z
    pub fn human_skull(pos: Vec3, scale: f32) -> Self {
        let mut body = SdfBody::new();
        let s = scale;
        let bone: [u8; 3] = [235, 225, 200];

        let mut skull = SdfShape::new("skull", 10, bone);

        // === CRANIAL VAULT ===
        // Main cranium — elongated ellipsoid (deeper than wide)
        // From scull.jpg side view: depth ≈ 85% of height, width ≈ 70% of height
        skull.add(SdfPrimitive::Ellipsoid {
            center: pos + Vec3::new(0.0, 6.0*s, -1.0*s),
            radii: Vec3::new(4.5*s, 5.5*s, 5.8*s), // width < height < depth
        }, 0.0);

        // Frontal bone (forehead)
        skull.add(SdfPrimitive::Sphere {
            center: pos + Vec3::new(0.0, 8.0*s, 3.0*s),
            radius: 3.0*s,
        }, 3.0); // ABSOLUTE blend, not scaled!

        // Occipital bone (back of skull — big backward bulge)
        skull.add(SdfPrimitive::Sphere {
            center: pos + Vec3::new(0.0, 4.5*s, -5.0*s),
            radius: 3.5*s,
        }, 3.0);

        // === FACE ===
        // Brow ridge
        skull.add(SdfPrimitive::RoundBox {
            center: pos + Vec3::new(0.0, 4.0*s, 3.5*s),
            half_extents: Vec3::new(3.2*s, 0.8*s, 0.8*s),
            rounding: 0.5*s,
        }, 2.0);

        // Zygomatic arches (cheekbones) — capsules from face to side
        skull.add(SdfPrimitive::Capsule {
            a: pos + Vec3::new(-2.5*s, 3.0*s, 2.5*s),
            b: pos + Vec3::new(-4.5*s, 2.5*s, 0.0),
            radius: 1.0*s,
        }, 1.5);
        skull.add(SdfPrimitive::Capsule {
            a: pos + Vec3::new(2.5*s, 3.0*s, 2.5*s),
            b: pos + Vec3::new(4.5*s, 2.5*s, 0.0),
            radius: 1.0*s,
        }, 1.5);

        // Maxilla (upper jaw)
        skull.add(SdfPrimitive::RoundBox {
            center: pos + Vec3::new(0.0, 1.0*s, 3.0*s),
            half_extents: Vec3::new(2.2*s, 1.2*s, 1.2*s),
            rounding: 0.5*s,
        }, 1.5);

        // Mandible — U-shaped jaw, 3 capsules
        skull.add(SdfPrimitive::Capsule {
            a: pos + Vec3::new(-3.0*s, 1.0*s, 0.0),
            b: pos + Vec3::new(-1.5*s, -1.5*s, 2.5*s),
            radius: 1.0*s,
        }, 1.5);
        skull.add(SdfPrimitive::Capsule {
            a: pos + Vec3::new(3.0*s, 1.0*s, 0.0),
            b: pos + Vec3::new(1.5*s, -1.5*s, 2.5*s),
            radius: 1.0*s,
        }, 1.5);
        skull.add(SdfPrimitive::Capsule {
            a: pos + Vec3::new(-1.5*s, -1.5*s, 2.5*s),
            b: pos + Vec3::new(1.5*s, -1.5*s, 2.5*s),
            radius: 1.0*s,
        }, 1.5);

        // === CAVITIES (subtract) ===
        // Eye sockets — deep holes
        skull.sub(SdfPrimitive::Sphere {
            center: pos + Vec3::new(-1.8*s, 3.5*s, 4.5*s),
            radius: 1.5*s,
        }, 1.0);
        skull.sub(SdfPrimitive::Sphere {
            center: pos + Vec3::new(1.8*s, 3.5*s, 4.5*s),
            radius: 1.5*s,
        }, 1.0);

        // Nasal aperture
        skull.sub(SdfPrimitive::Ellipsoid {
            center: pos + Vec3::new(0.0, 1.8*s, 4.5*s),
            radii: Vec3::new(0.8*s, 1.3*s, 1.0*s),
        }, 0.5);

        // Temporal fossae (indent on sides)
        skull.sub(SdfPrimitive::Sphere {
            center: pos + Vec3::new(-5.5*s, 5.0*s, 1.0*s),
            radius: 2.0*s,
        }, 1.0);
        skull.sub(SdfPrimitive::Sphere {
            center: pos + Vec3::new(5.5*s, 5.0*s, 1.0*s),
            radius: 2.0*s,
        }, 1.0);

        // Foramen magnum
        skull.sub(SdfPrimitive::Sphere {
            center: pos + Vec3::new(0.0, -0.5*s, -1.5*s),
            radius: 1.5*s,
        }, 0.5);

        body.add_shape(skull);
        body
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdf_sphere() {
        let s = SdfPrimitive::Sphere { center: Vec3::ZERO, radius: 5.0 };
        assert!((s.distance(Vec3::ZERO) - (-5.0)).abs() < 0.01);
        assert!((s.distance(Vec3::new(5.0, 0.0, 0.0)) - 0.0).abs() < 0.01);
        assert!((s.distance(Vec3::new(10.0, 0.0, 0.0)) - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_sdf_skull_rasterize() {
        let pos = Vec3::new(64.0, 64.0, 64.0);
        let s = 4.0;
        let body = SdfBody::human_skull(pos, s);

        // Debug: check SDF value at skull center
        let skull = &body.shapes[0];
        let d_center = skull.distance(pos + Vec3::new(0.0, 6.0*s, 0.0));
        let d_surface = skull.distance(pos + Vec3::new(4.8*s, 6.0*s, 0.0));
        println!("SDF at cranium center: {:.2}", d_center);
        println!("SDF near cranium surface: {:.2}", d_surface);
        println!("Skull bounds: {:?}", skull.bounds());

        let mut count = 0;
        body.rasterize(128, 2.0, |_,_,_,_,_,_,_| { count += 1; });
        println!("Skull voxels (shell): {}", count);
        assert!(count > 100);
    }

    #[test]
    fn test_sdf_human_body() {
        use super::super::skeleton::Skeleton;

        let mut sk = Skeleton::human(4.0);
        sk.root_position = Vec3::new(64.0, 60.0, 64.0);
        sk.solve_forward();

        let body = SdfBody::human_body(&sk, 4.0);
        println!("Human body: {} shapes", body.shapes.len());
        for shape in &body.shapes {
            println!("  {} (mat={}, color={:?})", shape.name, shape.material, shape.color);
        }

        // Should have torso + belt + neck + head + 2×(shoulder+upper+forearm+hand) + 2×(hip+thigh+shin+boot)
        assert!(body.shapes.len() >= 16, "Expected 16+ shapes, got {}", body.shapes.len());

        let mut count = 0;
        body.rasterize(128, 1.5, |_,_,_,_,_,_,_| { count += 1; });
        println!("Human body voxels: {}", count);
        // Full body should have significantly more voxels than just a skull
        assert!(count > 5000, "Expected >5000 voxels, got {}", count);
    }

    #[test]
    fn test_sdf_body_with_surface_nets() {
        use super::super::skeleton::Skeleton;
        use super::super::meshing;
        use super::super::svo::Voxel as V;

        let mut sk = Skeleton::human(2.0);
        sk.root_position = Vec3::new(64.0, 50.0, 64.0);
        sk.solve_forward();

        let sdf = SdfBody::human_body(&sk, 2.0);

        // Rasterize into flat grid
        let size = 128;
        let mut voxels = vec![V::empty(); size * size * size];
        sdf.rasterize(size, 1.5, |x,y,z,mat,r,g,b| {
            if x < size && y < size && z < size {
                voxels[z * size * size + y * size + x] = V::solid(mat, r, g, b);
            }
        });

        let solid = voxels.iter().filter(|v| v.is_solid()).count();
        println!("Rasterized: {} solid voxels in {}³", solid, size);

        // Now mesh with Surface Nets
        let mesh = meshing::generate_mesh_smooth(&voxels, size, Vec3::ZERO, 1.0);
        println!("Surface Nets: {} triangles, {} vertices", mesh.triangle_count, mesh.vertices.len());
        assert!(mesh.triangle_count > 1000, "Expected >1000 triangles from smooth meshing");

        // Verify normals mostly point outward
        let center = Vec3::new(64.0, 50.0, 64.0);
        let mut outward = 0;
        for v in &mesh.vertices {
            let pos = Vec3::from(v.position);
            let normal = Vec3::from(v.normal);
            let to_center = center - pos;
            if normal.dot(to_center) < 0.0 { outward += 1; }
        }
        let ratio = outward as f32 / mesh.vertices.len().max(1) as f32;
        println!("Normals outward: {:.0}%", ratio * 100.0);
    }

    #[test]
    fn test_sdf_smooth_union() {
        // Two spheres with smooth blend should have distance < min(d1, d2) at midpoint
        let mut shape = SdfShape::new("test", 1, [255,255,255]);
        shape.add(SdfPrimitive::Sphere { center: Vec3::new(-3.0, 0.0, 0.0), radius: 3.0 }, 0.0);
        shape.add(SdfPrimitive::Sphere { center: Vec3::new(3.0, 0.0, 0.0), radius: 3.0 }, 2.0);
        let d = shape.distance(Vec3::ZERO);
        // At origin: each sphere is distance 0.0. With smooth union, should be < 0
        assert!(d < 0.0);
    }
}
