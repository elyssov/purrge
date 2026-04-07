// ═══════════════════════════════════════════════════════════════
// PURRGE — Physics Engine
//
// Real physics, not hardcoded animations.
// Vase falls because gravity + no support. Not because we said so.
// Cat (10kg) knocks over chair (5kg) but bounces off fridge (80kg).
// Debris has weight. Everything obeys F=ma.
// ═══════════════════════════════════════════════════════════════

use glam::Vec3;

const GRAVITY_ACCEL: f32 = 9.81;  // m/s², but we scale to voxel units
const VOXEL_SIZE: f32 = 0.01;     // 1 voxel ≈ 1 cm (for mass calculations)

/// Axis-Aligned Bounding Box
#[derive(Clone, Debug)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn new(center: Vec3, half: Vec3) -> Self {
        Self { min: center - half, max: center + half }
    }

    pub fn center(&self) -> Vec3 { (self.min + self.max) * 0.5 }
    pub fn half_extents(&self) -> Vec3 { (self.max - self.min) * 0.5 }
    pub fn volume(&self) -> f32 {
        let s = self.max - self.min;
        s.x * s.y * s.z
    }

    pub fn overlaps(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x &&
        self.min.y <= other.max.y && self.max.y >= other.min.y &&
        self.min.z <= other.max.z && self.max.z >= other.min.z
    }

    /// Penetration depth along each axis (negative = no overlap)
    pub fn penetration(&self, other: &AABB) -> Vec3 {
        Vec3::new(
            (self.max.x.min(other.max.x) - self.min.x.max(other.min.x)).max(0.0),
            (self.max.y.min(other.max.y) - self.min.y.max(other.min.y)).max(0.0),
            (self.max.z.min(other.max.z) - self.min.z.max(other.min.z)).max(0.0),
        )
    }

    /// Minimum translation vector to separate two AABBs
    pub fn mtv(&self, other: &AABB) -> Option<Vec3> {
        let pen = self.penetration(other);
        if pen.x <= 0.0 || pen.y <= 0.0 || pen.z <= 0.0 {
            return None; // no overlap
        }
        // Push along axis of minimum penetration
        let dir = self.center() - other.center();
        if pen.x <= pen.y && pen.x <= pen.z {
            Some(Vec3::new(pen.x * dir.x.signum(), 0.0, 0.0))
        } else if pen.y <= pen.z {
            Some(Vec3::new(0.0, pen.y * dir.y.signum(), 0.0))
        } else {
            Some(Vec3::new(0.0, 0.0, pen.z * dir.z.signum()))
        }
    }
}

/// A physical body in the world
#[derive(Clone, Debug)]
pub struct RigidBody {
    pub id: u32,
    pub name: String,

    // ── Spatial ──
    pub pos: Vec3,           // center of mass (voxel units)
    pub vel: Vec3,           // velocity (voxels/sec)
    pub half_extents: Vec3,  // AABB half-sizes

    // ── Physical ──
    pub mass: f32,           // kg
    pub restitution: f32,    // bounciness: 0=dead stop, 1=perfect bounce
    pub friction: f32,       // 0=ice, 1=rubber

    // ── State ──
    pub grounded: bool,      // resting on something
    pub is_static: bool,     // immovable (walls, floor, heavy furniture)
    pub is_cat: bool,        // special: player-controlled, always has force
    pub active: bool,        // false = removed from simulation

    // ── Material ──
    pub material_id: u8,
    pub value: f32,          // $ for repair bill
}

impl RigidBody {
    pub fn new_dynamic(id: u32, name: &str, pos: Vec3, half: Vec3, mass: f32, mat_id: u8) -> Self {
        Self {
            id, name: name.to_string(),
            pos, vel: Vec3::ZERO, half_extents: half,
            mass, restitution: 0.3, friction: 0.5,
            grounded: false, is_static: false, is_cat: false, active: true,
            material_id: mat_id, value: 0.0,
        }
    }

    pub fn new_static(id: u32, name: &str, pos: Vec3, half: Vec3) -> Self {
        Self {
            id, name: name.to_string(),
            pos, vel: Vec3::ZERO, half_extents: half,
            mass: f32::MAX, restitution: 0.1, friction: 0.8,
            grounded: true, is_static: true, is_cat: false, active: true,
            material_id: 0, value: 0.0,
        }
    }

    pub fn new_cat(pos: Vec3) -> Self {
        Self {
            id: 0, name: "Cat".to_string(),
            pos, vel: Vec3::ZERO,
            half_extents: Vec3::new(3.0, 6.0, 10.0), // cat-sized AABB
            mass: 10.0, // 10 kg chonky boy
            restitution: 0.1, friction: 0.6,
            grounded: false, is_static: false, is_cat: true, active: true,
            material_id: 13, value: 0.0, // priceless
        }
    }

    pub fn aabb(&self) -> AABB {
        AABB::new(self.pos, self.half_extents)
    }

    pub fn inv_mass(&self) -> f32 {
        if self.is_static { 0.0 } else { 1.0 / self.mass }
    }

    /// Mass from material density and volume (for procedural objects)
    pub fn mass_from_density(density_kg_m3: f32, half: Vec3) -> f32 {
        let volume_voxels = half.x * half.y * half.z * 8.0; // full box volume
        let volume_m3 = volume_voxels * VOXEL_SIZE * VOXEL_SIZE * VOXEL_SIZE;
        density_kg_m3 * volume_m3
    }

    /// Apply impulse (instantaneous velocity change)
    pub fn apply_impulse(&mut self, impulse: Vec3) {
        if !self.is_static {
            self.vel += impulse * self.inv_mass();
        }
    }

    /// Kinetic energy (for chain reaction scoring)
    pub fn kinetic_energy(&self) -> f32 {
        0.5 * self.mass * self.vel.length_squared()
    }
}

/// Physics world — manages all bodies and simulation
pub struct PhysicsWorld {
    pub bodies: Vec<RigidBody>,
    pub gravity: f32,        // voxels/sec² (scaled from 9.81)
    pub floor_y: f32,        // floor plane Y coordinate
    next_id: u32,
}

/// Event emitted by physics (for sound, particles, scoring)
#[derive(Debug, Clone)]
pub enum PhysicsEvent {
    /// Two bodies collided
    Collision {
        id_a: u32, id_b: u32,
        point: Vec3,
        impulse: f32,   // magnitude — louder = bigger impact
    },
    /// Body hit the floor
    FloorHit {
        id: u32,
        speed: f32,     // impact speed — for damage/sound
    },
    /// Body came to rest
    Settled { id: u32 },
    /// Body fell off the world
    FellOff { id: u32 },
}

impl PhysicsWorld {
    pub fn new(floor_y: f32) -> Self {
        Self {
            bodies: Vec::new(),
            gravity: 120.0, // voxel-scaled gravity (feels right at our scale)
            floor_y,
            next_id: 1,
        }
    }

    /// Add a body, returns its ID
    pub fn add(&mut self, mut body: RigidBody) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        body.id = id;
        self.bodies.push(body);
        id
    }

    /// Spawn debris from destruction point
    pub fn spawn_debris(&mut self, pos: Vec3, velocity: Vec3, mass: f32, mat_id: u8, value: f32) -> u32 {
        let size = (mass * 0.5).cbrt().max(1.0); // bigger mass = bigger chunk
        let mut body = RigidBody::new_dynamic(
            0, "debris", pos,
            Vec3::splat(size), mass, mat_id,
        );
        body.vel = velocity;
        body.restitution = 0.4;
        body.value = value;
        self.add(body)
    }

    /// Main simulation step
    pub fn step(&mut self, dt: f32) -> Vec<PhysicsEvent> {
        let mut events = Vec::new();

        // 1. Gravity + integration
        for body in self.bodies.iter_mut() {
            if !body.active || body.is_static { continue; }

            // Gravity
            if !body.grounded {
                body.vel.y -= self.gravity * dt;
            }

            // Velocity damping (air resistance / friction)
            if body.grounded {
                body.vel.x *= 1.0 - body.friction * dt * 5.0;
                body.vel.z *= 1.0 - body.friction * dt * 5.0;
            } else {
                body.vel *= 1.0 - 0.01 * dt; // minimal air drag
            }

            // Integrate position
            body.pos += body.vel * dt;
        }

        // 2. Floor collision
        for body in self.bodies.iter_mut() {
            if !body.active || body.is_static { continue; }

            let bottom = body.pos.y - body.half_extents.y;
            if bottom < self.floor_y {
                let impact_speed = (-body.vel.y).max(0.0);
                body.pos.y = self.floor_y + body.half_extents.y;

                if impact_speed > 5.0 {
                    events.push(PhysicsEvent::FloorHit { id: body.id, speed: impact_speed });
                    body.vel.y = impact_speed * body.restitution; // bounce UP
                    body.grounded = false; // NOT grounded — still bouncing!
                } else {
                    body.vel.y = 0.0;
                    body.grounded = true; // settled
                }
            }

            // Fell off world
            if body.pos.y < -100.0 || body.pos.x < -50.0 || body.pos.x > 300.0
            || body.pos.z < -50.0 || body.pos.z > 300.0 {
                body.active = false;
                events.push(PhysicsEvent::FellOff { id: body.id });
            }
        }

        // 3. Body vs body collisions
        let n = self.bodies.len();
        for i in 0..n {
            for j in (i+1)..n {
                if !self.bodies[i].active || !self.bodies[j].active { continue; }
                if self.bodies[i].is_static && self.bodies[j].is_static { continue; }

                let aabb_a = self.bodies[i].aabb();
                let aabb_b = self.bodies[j].aabb();

                if let Some(mtv) = aabb_a.mtv(&aabb_b) {
                    // Collision detected!
                    let (body_a, body_b) = get_two_mut(&mut self.bodies, i, j);

                    // Separate bodies
                    let total_inv = body_a.inv_mass() + body_b.inv_mass();
                    if total_inv > 0.0 {
                        let ratio_a = body_a.inv_mass() / total_inv;
                        let ratio_b = body_b.inv_mass() / total_inv;
                        body_a.pos += mtv * ratio_a;
                        body_b.pos -= mtv * ratio_b;
                    }

                    // Impulse-based collision response
                    let normal = mtv.normalize();
                    let rel_vel = body_a.vel - body_b.vel;
                    let vel_along_normal = rel_vel.dot(normal);

                    // Only resolve if bodies are moving toward each other
                    if vel_along_normal < 0.0 {
                        let e = (body_a.restitution + body_b.restitution) * 0.5;
                        let j = -(1.0 + e) * vel_along_normal / total_inv.max(0.001);
                        let impulse = normal * j;

                        body_a.vel += impulse * body_a.inv_mass();
                        body_b.vel -= impulse * body_b.inv_mass();

                        events.push(PhysicsEvent::Collision {
                            id_a: body_a.id, id_b: body_b.id,
                            point: (body_a.pos + body_b.pos) * 0.5,
                            impulse: j.abs(),
                        });
                    }
                }
            }
        }

        // 4. Settle check — bodies nearly at rest
        for body in self.bodies.iter_mut() {
            if !body.active || body.is_static { continue; }
            if body.grounded && body.vel.length_squared() < 0.5 {
                body.vel = Vec3::ZERO;
                // Don't emit Settled every frame — only on transition
            }
        }

        // 5. Cleanup dead bodies
        self.bodies.retain(|b| b.active);

        events
    }

    /// Get body by ID
    pub fn get(&self, id: u32) -> Option<&RigidBody> {
        self.bodies.iter().find(|b| b.id == id)
    }

    pub fn get_mut(&mut self, id: u32) -> Option<&mut RigidBody> {
        self.bodies.iter_mut().find(|b| b.id == id)
    }

    /// Cat pushes against an object (called when cat walks into something)
    /// Returns true if the object moved
    pub fn cat_push(&mut self, cat_id: u32, target_id: u32, push_dir: Vec3) -> bool {
        let cat_mass;
        let cat_speed;
        if let Some(cat) = self.get(cat_id) {
            cat_mass = cat.mass;
            cat_speed = cat.vel.length().max(20.0); // minimum push force
        } else { return false; }

        let grav = self.gravity;
        if let Some(target) = self.get_mut(target_id) {
            if target.is_static { return false; }

            let force = cat_mass * cat_speed;
            let accel = force / target.mass;

            let friction_force = target.mass * grav * target.friction;
            if force > friction_force * 0.3 {
                target.vel += push_dir.normalize() * accel * 0.1;
                return true;
            }
        }
        false
    }

    /// Remove support: check if an object at position has anything below it
    /// Returns true if the object is now unsupported (should start falling)
    pub fn check_support(&self, body: &RigidBody) -> bool {
        let bottom = body.pos.y - body.half_extents.y;
        // Check if anything is below (floor or another body)
        if bottom <= self.floor_y + 0.5 {
            return true; // on floor
        }
        // Check against other static/grounded bodies
        let check_aabb = AABB::new(
            Vec3::new(body.pos.x, bottom - 1.0, body.pos.z),
            Vec3::new(body.half_extents.x * 0.8, 1.0, body.half_extents.z * 0.8),
        );
        for other in &self.bodies {
            if other.id == body.id || !other.active { continue; }
            if (other.is_static || other.grounded) && other.aabb().overlaps(&check_aabb) {
                return true; // supported by another body
            }
        }
        false // UNSUPPORTED — will fall!
    }

    /// After destruction: check all non-static bodies for support
    /// Returns IDs of bodies that lost support and started falling
    pub fn check_all_support(&mut self) -> Vec<u32> {
        let mut falling = Vec::new();

        for i in 0..self.bodies.len() {
            if self.bodies[i].is_static || !self.bodies[i].active { continue; }
            let body = self.bodies[i].clone();
            if !self.check_support(&body) {
                self.bodies[i].grounded = false;
                falling.push(self.bodies[i].id);
            }
        }
        falling
    }
}

/// Helper: get two mutable references from a vec by index (safe)
fn get_two_mut(v: &mut Vec<RigidBody>, i: usize, j: usize) -> (&mut RigidBody, &mut RigidBody) {
    assert!(i != j);
    if i < j {
        let (left, right) = v.split_at_mut(j);
        (&mut left[i], &mut right[0])
    } else {
        let (left, right) = v.split_at_mut(i);
        (&mut right[0], &mut left[j])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_overlap() {
        let a = AABB::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(5.0, 5.0, 5.0));
        let b = AABB::new(Vec3::new(3.0, 0.0, 0.0), Vec3::new(5.0, 5.0, 5.0));
        assert!(a.overlaps(&b));
        let c = AABB::new(Vec3::new(20.0, 0.0, 0.0), Vec3::new(5.0, 5.0, 5.0));
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_gravity_drops_object() {
        let mut world = PhysicsWorld::new(0.0);
        let id = world.add(RigidBody::new_dynamic(
            0, "ball", Vec3::new(50.0, 50.0, 50.0),
            Vec3::splat(2.0), 1.0, 1,
        ));
        for _ in 0..600 { world.step(1.0/60.0); } // 10 seconds
        let body = world.get(id).unwrap();
        assert!(body.pos.y < 10.0, "body.y={} should be near floor", body.pos.y);
    }

    #[test]
    fn test_floor_collision() {
        let mut world = PhysicsWorld::new(5.0);
        let id = world.add(RigidBody::new_dynamic(
            0, "box", Vec3::new(50.0, 20.0, 50.0),
            Vec3::splat(3.0), 2.0, 1,
        ));
        for _ in 0..600 { world.step(1.0/60.0); }
        let body = world.get(id).unwrap();
        assert!((body.pos.y - (5.0 + 3.0)).abs() < 2.0, "body.y={}", body.pos.y);
    }

    #[test]
    fn test_cat_pushes_light_object() {
        let mut world = PhysicsWorld::new(0.0);
        let cat_id = world.add(RigidBody::new_cat(Vec3::new(50.0, 10.0, 50.0)));
        let chair_id = world.add({
            let mut c = RigidBody::new_dynamic(0, "chair",
                Vec3::new(55.0, 10.0, 50.0), Vec3::new(5.0, 8.0, 5.0), 5.0, 1);
            c.grounded = true;
            c
        });
        // Cat pushes chair
        let moved = world.cat_push(cat_id, chair_id, Vec3::X);
        assert!(moved); // 10kg cat pushes 5kg chair
    }

    #[test]
    fn test_cat_cant_push_fridge() {
        let mut world = PhysicsWorld::new(0.0);
        let cat_id = world.add(RigidBody::new_cat(Vec3::new(50.0, 10.0, 50.0)));
        let fridge_id = world.add({
            let mut f = RigidBody::new_dynamic(0, "fridge",
                Vec3::new(55.0, 10.0, 50.0), Vec3::new(10.0, 20.0, 10.0), 80.0, 4);
            f.friction = 0.9; // heavy + high friction
            f.grounded = true;
            f
        });
        let moved = world.cat_push(cat_id, fridge_id, Vec3::X);
        assert!(!moved); // 10kg cat can't push 80kg fridge
    }

    #[test]
    fn test_collision_impulse() {
        let mut world = PhysicsWorld::new(0.0);
        // Two objects moving toward each other
        let id_a = world.add({
            let mut a = RigidBody::new_dynamic(0, "a", Vec3::new(45.0, 5.0, 50.0),
                Vec3::splat(3.0), 5.0, 1);
            a.vel = Vec3::new(50.0, 0.0, 0.0); // fast enough to reach B in one step
            a.grounded = true;
            a
        });
        let id_b = world.add({
            let mut b = RigidBody::new_dynamic(0, "b", Vec3::new(50.0, 5.0, 50.0), // closer!
                Vec3::splat(3.0), 5.0, 1);
            b.grounded = true;
            b
        });
        let events = world.step(1.0/60.0);
        // Should have collision event
        assert!(events.iter().any(|e| matches!(e, PhysicsEvent::Collision { .. })));
        // B should have gained velocity
        let b = world.get(id_b).unwrap();
        assert!(b.vel.x > 0.0);
    }

    #[test]
    fn test_support_check() {
        let mut world = PhysicsWorld::new(0.0);
        // Table (static, acts as support)
        world.add(RigidBody::new_static(0, "table",
            Vec3::new(50.0, 25.0, 50.0), Vec3::new(15.0, 1.0, 10.0)));
        // Vase on table
        let vase_id = world.add(RigidBody::new_dynamic(0, "vase",
            Vec3::new(50.0, 30.0, 50.0), Vec3::new(2.0, 4.0, 2.0), 3.0, 3));

        let vase = world.get(vase_id).unwrap();
        assert!(world.check_support(vase)); // supported by table
    }

    #[test]
    fn test_unsupported_falls() {
        let mut world = PhysicsWorld::new(0.0);
        // Vase floating in air (no support)
        let vase_id = world.add(RigidBody::new_dynamic(0, "vase",
            Vec3::new(50.0, 50.0, 50.0), Vec3::new(2.0, 4.0, 2.0), 3.0, 3));

        let vase = world.get(vase_id).unwrap();
        assert!(!world.check_support(vase)); // NOT supported!

        // Simulate — should fall
        for _ in 0..300 {
            world.step(1.0/60.0);
        }
        let vase = world.get(vase_id).unwrap();
        assert!(vase.pos.y < 10.0); // fell down
    }

    #[test]
    fn test_mass_from_density() {
        // Ceramic vase: density 2300 kg/m³, half_extents (2, 4, 2) voxels
        let mass = RigidBody::mass_from_density(2300.0, Vec3::new(2.0, 4.0, 2.0));
        // 4*8*4 = 128 voxels, each 0.01³ m³ = 1e-6 m³, total = 128e-6 m³
        // 2300 * 128e-6 ≈ 0.294 kg
        assert!(mass > 0.1 && mass < 1.0);
    }

    #[test]
    fn test_debris_spawning() {
        let mut world = PhysicsWorld::new(0.0);
        let id = world.spawn_debris(
            Vec3::new(50.0, 30.0, 50.0),
            Vec3::new(10.0, 20.0, -5.0),
            0.5, 3, 50.0,
        );
        assert!(world.get(id).is_some());
        // Simulate
        for _ in 0..60 {
            world.step(1.0/60.0);
        }
        let debris = world.get(id).unwrap();
        assert!(debris.pos.y < 30.0); // fell from impact point
    }
}
