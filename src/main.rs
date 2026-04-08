// ═══════════════════════════════════════════════════════════════
// PURRGE — Build 12 "Leopold"
// Mesh pipeline (Surface Nets + PBR). No more cubes.
// Procedural apartment, dog AI, parrot, meters, scoring.
// ═══════════════════════════════════════════════════════════════

mod core;
mod game;
mod render;
mod apartment;
mod furniture;

use crate::apartment::{VoxelGrid, generate_apartment_v2, GRID, FLOOR_Y};
use crate::furniture::FurnitureObj;
use crate::core::body::BodyDefinition;
use crate::core::skeleton::Skeleton;
use crate::core::svo::Voxel;
use crate::game::dog::{Dog, DogEvent};
use crate::game::meters::{Meters, GameOver};
use crate::game::scoring::{RepairBill, OwnerType};
use crate::game::timer::GameTimer;
use crate::game::items;
use crate::game::physics::{PhysicsWorld, RigidBody, PhysicsEvent};
use crate::render::Renderer;
use crate::core::render_mesh::GpuMesh;
use glam::{Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, MouseButton, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window},
};

// Cat is rasterized at high internal scale for voxel detail,
// then the mesh is scaled down to world size.
const CAT_INTERNAL_SCALE: f32 = 2.0;   // high-res rasterization
const CAT_WORLD_SCALE: f32 = 0.35;     // mesh shrink (effective size ≈ 0.7)
const DOG_INTERNAL_SCALE: f32 = 1.8;
const DOG_WORLD_SCALE: f32 = 0.30;     // slightly smaller than cat

const LEG_HEIGHT: f32 = 13.5;
const MOVE_SPEED: f32 = 55.0;
const GRAVITY: f32 = 55.0;
const JUMP_VEL: f32 = 52.0;
const SCRATCH_SPEED: f32 = 1.1;
const MOUSE_SENS: f32 = 0.003;
const SCRATCH_RANGE: f32 = 35.0;
const CAT_GRID: usize = 160;  // was 96 — more voxels = more detail
const DOG_GRID: usize = 96;   // was 64

// ─── Cat ────────────────────────────────────────────────────
struct CatState {
    x:f32, y:f32, z:f32, facing:f32, vy:f32,
    walk_phase:f32, moving:bool,
    scratching:bool, scratch_phase:f32, scratch_fired:bool,
    in_air:bool, land_timer:f32,
}
impl CatState {
    fn new() -> Self {
        Self { x:65.0, y:FLOOR_Y as f32+1.0+LEG_HEIGHT, z:75.0, facing:0.5,
               vy:0.0, walk_phase:0.0, moving:false,
               scratching:false, scratch_phase:0.0, scratch_fired:false,
               in_air:false, land_timer:0.0 }
    }
    fn forward(&self) -> Vec3 { Vec3::new(self.facing.sin(), 0.0, self.facing.cos()) }
    fn right(&self) -> Vec3 { let f = self.forward(); Vec3::new(f.z, 0.0, -f.x) }
}

#[derive(Default)]
struct Input { forward:bool, back:bool, left:bool, right:bool, jump:bool, scratch_pressed:bool, sprint:bool }

// ─── HUD helpers (NDC screen-space quads) ──────────────────
/// Push a filled rectangle in NDC coords (-1..1) with color
fn hud_bar(verts: &mut Vec<f32>, x: f32, y: f32, w: f32, h: f32, r: f32, g: f32, b: f32, a: f32) {
    // Two triangles = 6 vertices, each = x,y, r,g,b,a
    for &(vx, vy) in &[(x,y), (x+w,y), (x+w,y+h), (x,y), (x+w,y+h), (x,y+h)] {
        verts.extend_from_slice(&[vx, vy, r, g, b, a]);
    }
}

// ─── Debris Particles ──────────────────────────────────────
struct Particle {
    x: f32, y: f32, z: f32,
    vx: f32, vy: f32, vz: f32,
    color: [u8; 3],
    life: f32,
}

struct ParticleSystem {
    particles: Vec<Particle>,
}

impl ParticleSystem {
    fn new() -> Self { Self { particles: Vec::new() } }

    fn spawn_debris(&mut self, hit: Vec3, debris: &[(Vec3, Voxel)], forward: Vec3) {
        for (pos, vox) in debris.iter().take(50) { // cap at 50 particles per hit
            let color = [(vox.packed >> 16) as u8, (vox.packed >> 8) as u8, vox.packed as u8];
            // Random-ish velocity: outward from hit + upward + slight scatter
            let dx = pos.x - hit.x;
            let dz = pos.z - hit.z;
            let scatter = ((pos.x * 7.0 + pos.z * 13.0) % 3.0) - 1.5;
            self.particles.push(Particle {
                x: pos.x, y: pos.y, z: pos.z,
                vx: forward.x * 30.0 + dx * 5.0 + scatter * 10.0,
                vy: 20.0 + ((pos.y * 11.0) % 20.0),
                vz: forward.z * 30.0 + dz * 5.0 + scatter * 10.0,
                color,
                life: 1.5,
            });
        }
    }

    fn update(&mut self, dt: f32) {
        for p in &mut self.particles {
            p.x += p.vx * dt;
            p.y += p.vy * dt;
            p.z += p.vz * dt;
            p.vy -= 80.0 * dt; // gravity
            p.life -= dt;
            // Bounce off floor
            if p.y < 4.0 { p.y = 4.0; p.vy = p.vy.abs() * 0.3; p.vx *= 0.8; p.vz *= 0.8; }
        }
        self.particles.retain(|p| p.life > 0.0);
    }

    /// Rasterize particles into a small voxel grid for mesh rendering
    fn rasterize(&self, grid: &mut [Voxel], grid_size: usize) {
        for p in &self.particles {
            let alpha = (p.life / 1.5).clamp(0.0, 1.0);
            let x = p.x as usize; let y = p.y as usize; let z = p.z as usize;
            if x < grid_size && y < grid_size && z < grid_size {
                let r = (p.color[0] as f32 * alpha) as u8;
                let g = (p.color[1] as f32 * alpha) as u8;
                let b = (p.color[2] as f32 * alpha) as u8;
                // 2x2x2 cube per particle for visibility
                for dz in 0..2 { for dy in 0..2 { for dx in 0..2 {
                    let px = x+dx; let py = y+dy; let pz = z+dz;
                    if px < grid_size && py < grid_size && pz < grid_size {
                        grid[pz * grid_size * grid_size + py * grid_size + px] = Voxel::solid(1, r, g, b);
                    }
                }}}
            }
        }
    }
}

#[derive(PartialEq)]
enum GameState { Menu, Playing, Paused, Over(String), Bill }
impl GameState {
    fn is_playing(&self) -> bool { matches!(self, GameState::Playing) }
    fn is_menu(&self) -> bool { matches!(self, GameState::Menu) }
}

// ─── Cat Animation ──────────────────────────────────────────
fn animate_cat(sk: &mut Skeleton, cat: &CatState, t: f32) {
    let tau = std::f32::consts::PI * 2.0;

    // Legs
    if cat.in_air {
        if cat.vy > 0.0 {
            for u in ["upper_arm_l","upper_arm_r"] { sk.set_rotation(u, Quat::from_axis_angle(Vec3::X, -0.5)); }
            for l in ["forearm_l","forearm_r"] { sk.set_rotation(l, Quat::from_axis_angle(Vec3::X, 0.3)); }
            for u in ["thigh_l","thigh_r"] { sk.set_rotation(u, Quat::from_axis_angle(Vec3::X, 0.6)); }
            sk.set_hinge_angle("shin_l", 1.2); sk.set_hinge_angle("shin_r", 1.2);
        } else {
            for u in ["upper_arm_l","upper_arm_r"] { sk.set_rotation(u, Quat::from_axis_angle(Vec3::X, 0.3)); }
            for l in ["forearm_l","forearm_r"] { sk.set_rotation(l, Quat::from_axis_angle(Vec3::X, 0.1)); }
            for u in ["thigh_l","thigh_r"] { sk.set_rotation(u, Quat::from_axis_angle(Vec3::X, 0.2)); }
            sk.set_hinge_angle("shin_l", 0.3); sk.set_hinge_angle("shin_r", 0.3);
        }
    } else if cat.land_timer > 0.0 {
        let sq = (cat.land_timer / 0.2).min(1.0) * 0.6;
        for u in ["upper_arm_l","upper_arm_r"] { sk.set_rotation(u, Quat::from_axis_angle(Vec3::X, sq)); }
        for l in ["forearm_l","forearm_r"] { sk.set_rotation(l, Quat::from_axis_angle(Vec3::X, sq*1.5)); }
        for u in ["thigh_l","thigh_r"] { sk.set_rotation(u, Quat::from_axis_angle(Vec3::X, -sq*0.5)); }
        sk.set_hinge_angle("shin_l", 0.5+sq); sk.set_hinge_angle("shin_r", 0.5+sq);
    } else if cat.moving {
        let p = cat.walk_phase;
        let pi = std::f32::consts::PI;
        for (u, l, ph) in [("upper_arm_r","forearm_r",p),("upper_arm_l","forearm_l",p+0.5)] {
            let pp = ph.rem_euclid(1.0);
            sk.set_rotation(u, Quat::from_axis_angle(Vec3::X, 0.35*(pp*pi*2.0).sin()));
            let b = if pp > 0.25 && pp < 0.75 { 0.1+0.6*((pp-0.25)/0.5*pi).sin() } else { 0.1 };
            sk.set_rotation(l, Quat::from_axis_angle(Vec3::X, b));
        }
        for (u, l, ph) in [("thigh_r","shin_r",p+0.5),("thigh_l","shin_l",p)] {
            let pp = ph.rem_euclid(1.0);
            sk.set_rotation(u, Quat::from_axis_angle(Vec3::X, 0.3*(pp*pi*2.0).sin()));
            let b = if pp > 0.25 && pp < 0.75 { 0.15+0.7*((pp-0.25)/0.5*pi).sin() } else { 0.15 };
            sk.set_hinge_angle(l, b.max(0.087));
        }
    } else {
        let s = 0.04 * (t * 0.8).sin();
        for u in ["upper_arm_l","upper_arm_r"] { sk.set_rotation(u, Quat::from_axis_angle(Vec3::X, s)); }
        for u in ["thigh_l","thigh_r"] { sk.set_rotation(u, Quat::from_axis_angle(Vec3::X, -s)); }
        for l in ["forearm_l","forearm_r"] { sk.set_rotation(l, Quat::from_axis_angle(Vec3::X, 0.05)); }
        sk.set_hinge_angle("shin_l", 0.1); sk.set_hinge_angle("shin_r", 0.1);
    }

    // Tail
    for (i, n) in ["tail1","tail2","tail3","tail4"].iter().enumerate() {
        let d = i as f32 * 0.4;
        sk.set_rotation(n, Quat::from_axis_angle(Vec3::Y, 0.35*(t*0.7*tau+d).sin())
            * Quat::from_axis_angle(Vec3::X, 0.1*(i as f32+1.0)*(t*0.5).sin()));
    }
    // Head
    let nod = if cat.moving { 0.08*(cat.walk_phase*tau).sin() } else { 0.03*(t*1.2).sin() };
    sk.set_rotation("head", Quat::from_axis_angle(Vec3::X, nod));
    sk.set_rotation("neck", Quat::from_axis_angle(Vec3::X, -nod*0.4));
    sk.set_rotation("ear_l", Quat::from_axis_angle(Vec3::Z, 0.2*(t*3.7).sin()));
    sk.set_rotation("ear_r", Quat::from_axis_angle(Vec3::Z, 0.2*(t*4.3+1.5).sin()));
    // Spine
    if cat.moving && !cat.in_air {
        let fx = 0.06 * (cat.walk_phase*tau).sin();
        sk.set_rotation("spine1", Quat::from_axis_angle(Vec3::Y, fx));
        sk.set_rotation("spine2", Quat::from_axis_angle(Vec3::Y, -fx*0.7));
    }
    // Scratch: crouch on all fours, then swipe with front paw
    if cat.scratching {
        let p = cat.scratch_phase;
        // Phase 1 (0.0-0.4): CROUCH — all legs bend, body lowers, arm winds up
        if p < 0.4 {
            let s = { let l = p/0.4; l*l*(3.0-2.0*l) }; // smoothstep
            // Crouch: bend all legs
            for leg in ["upper_arm_l","upper_arm_r"] {
                sk.set_rotation(leg, Quat::from_axis_angle(Vec3::X, 0.4*s));
            }
            for leg in ["forearm_l","forearm_r"] {
                sk.set_hinge_angle(leg, 0.6*s);
            }
            for leg in ["thigh_l","thigh_r"] {
                sk.set_rotation(leg, Quat::from_axis_angle(Vec3::X, 0.3*s));
            }
            sk.set_hinge_angle("shin_l", 0.6*s + 0.087);
            sk.set_hinge_angle("shin_r", 0.6*s + 0.087);
            // Wind up right paw
            sk.set_rotation("upper_arm_r", Quat::from_axis_angle(Vec3::X, -1.6*s) * Quat::from_axis_angle(Vec3::Z, 0.6*s));
            sk.set_rotation("forearm_r", Quat::from_axis_angle(Vec3::X, 1.5*s));
            // Spine dips, ears back
            sk.set_rotation("spine2", Quat::from_axis_angle(Vec3::X, -0.15*s));
            sk.set_rotation("ear_l", Quat::from_axis_angle(Vec3::X, 0.4*s));
            sk.set_rotation("ear_r", Quat::from_axis_angle(Vec3::X, 0.4*s));
        }
        // Phase 2 (0.4-0.55): STRIKE — paw swipes forward explosively
        else if p < 0.55 {
            let s = { let l = (p-0.4)/0.15; l*l }; // quadratic
            // Maintain crouch
            for leg in ["thigh_l","thigh_r"] { sk.set_rotation(leg, Quat::from_axis_angle(Vec3::X, 0.3)); }
            sk.set_hinge_angle("shin_l", 0.687); sk.set_hinge_angle("shin_r", 0.687);
            // Paw strikes forward
            sk.set_rotation("upper_arm_r", Quat::from_axis_angle(Vec3::X, -1.6+3.2*s)*Quat::from_axis_angle(Vec3::Z, 0.6*(1.0-s)));
            sk.set_rotation("forearm_r", Quat::from_axis_angle(Vec3::X, 1.5*(1.0-s)));
            sk.set_rotation("spine2", Quat::from_axis_angle(Vec3::X, -0.15+0.3*s));
        }
        // Phase 3 (0.55-1.0): RECOVER — stand back up smoothly
        else {
            let s = { let l = ((p-0.55)/0.45).min(1.0); l*l*(3.0-2.0*l) }; // smoothstep
            // Uncrouch legs
            for leg in ["thigh_l","thigh_r"] { sk.set_rotation(leg, Quat::from_axis_angle(Vec3::X, 0.3*(1.0-s))); }
            let knee = 0.687 * (1.0-s) + 0.087;
            sk.set_hinge_angle("shin_l", knee); sk.set_hinge_angle("shin_r", knee);
            // Arm returns
            sk.set_rotation("upper_arm_r", Quat::from_axis_angle(Vec3::X, 1.6*(1.0-s)));
        }
    }
}

/// Rasterize cat into a small voxel grid for meshing.
/// Uses high internal scale for voxel detail; mesh is shrunk via CAT_WORLD_SCALE.
fn rasterize_cat_to_grid(cat: &CatState, time: f32) -> Vec<Voxel> {
    let mut grid = vec![Voxel::empty(); CAT_GRID * CAT_GRID * CAT_GRID];
    let half = CAT_GRID as f32 / 2.0;

    let mut sk = Skeleton::cat(CAT_INTERNAL_SCALE);
    // Root position: center X/Z, Y high enough for legs to fit
    sk.root_position = Vec3::new(half, half * 0.55, half);
    sk.root_rotation = Quat::from_rotation_y(cat.facing);
    animate_cat(&mut sk, cat, time);
    sk.solve_forward();

    let body = BodyDefinition::cat_body(&sk, CAT_INTERNAL_SCALE);
    body.rasterize(&sk, CAT_GRID, |x, y, z, mat, r, g, b| {
        if x < CAT_GRID && y < CAT_GRID && z < CAT_GRID {
            grid[z * CAT_GRID * CAT_GRID + y * CAT_GRID + x] = Voxel::solid(mat, r, g, b);
        }
    });
    grid
}

/// Rasterize dog using proper skeleton + body definition.
fn rasterize_dog_to_grid(dog: &crate::game::dog::Dog, time: f32) -> Vec<Voxel> {
    let mut grid = vec![Voxel::empty(); DOG_GRID * DOG_GRID * DOG_GRID];
    let half = DOG_GRID as f32 / 2.0;

    let mut sk = Skeleton::dog(DOG_INTERNAL_SCALE);
    sk.root_position = Vec3::new(half, half * 0.5, half);
    sk.root_rotation = Quat::IDENTITY;

    // Simple animation: tail wag, breathing, sleeping pose
    if dog.is_sleeping() {
        // Curl up: legs tucked, head down
        for u in ["upper_arm_l","upper_arm_r"] { sk.set_rotation(u, Quat::from_axis_angle(Vec3::X, 0.6)); }
        for l in ["forearm_l","forearm_r"] { sk.set_rotation(l, Quat::from_axis_angle(Vec3::X, 1.2)); }
        for u in ["thigh_l","thigh_r"] { sk.set_rotation(u, Quat::from_axis_angle(Vec3::X, 0.5)); }
        sk.set_hinge_angle("shin_l", 1.4); sk.set_hinge_angle("shin_r", 1.4);
        sk.set_rotation("neck", Quat::from_axis_angle(Vec3::X, 0.3));
        sk.set_rotation("head", Quat::from_axis_angle(Vec3::X, 0.2));
        // Gentle breathing
        let breath = 0.02 * (time * 0.5).sin();
        sk.set_rotation("spine1", Quat::from_axis_angle(Vec3::X, breath));
    } else {
        // Standing / alert — legs shift weight, body sways
        let breath = 0.03 * (time * 1.0).sin();
        sk.set_rotation("spine2", Quat::from_axis_angle(Vec3::X, breath));
        sk.set_rotation("head", Quat::from_axis_angle(Vec3::X, 0.05 * (time * 1.5).sin()));
        // Idle leg shifting (subtle weight transfer)
        let leg_t = time * 0.8;
        sk.set_rotation("upper_arm_l", Quat::from_axis_angle(Vec3::X, 0.08 * leg_t.sin()));
        sk.set_rotation("upper_arm_r", Quat::from_axis_angle(Vec3::X, 0.08 * (leg_t + 1.5).sin()));
        sk.set_rotation("thigh_l", Quat::from_axis_angle(Vec3::X, 0.06 * (leg_t + 0.7).sin()));
        sk.set_rotation("thigh_r", Quat::from_axis_angle(Vec3::X, 0.06 * (leg_t + 2.2).sin()));
        // Tail wag
        for (i, name) in ["tail1","tail2","tail3"].iter().enumerate() {
            let phase = i as f32 * 0.3;
            sk.set_rotation(name, Quat::from_axis_angle(Vec3::Y, 0.4 * (time * 4.0 + phase).sin()));
        }
        // Ear flop
        sk.set_rotation("ear_l", Quat::from_axis_angle(Vec3::Z, 0.1 * (time * 2.0).sin()));
        sk.set_rotation("ear_r", Quat::from_axis_angle(Vec3::Z, 0.1 * (time * 2.3 + 1.0).sin()));
    }

    sk.solve_forward();

    let body = BodyDefinition::dog_body(&sk, DOG_INTERNAL_SCALE);
    body.rasterize(&sk, DOG_GRID, |x, y, z, mat, r, g, b| {
        if x < DOG_GRID && y < DOG_GRID && z < DOG_GRID {
            grid[z * DOG_GRID * DOG_GRID + y * DOG_GRID + x] = Voxel::solid(mat, r, g, b);
        }
    });
    grid
}

// ─── App ────────────────────────────────────────────────────
struct App {
    win: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    room: VoxelGrid,
    cat: CatState,
    input: Input,
    dog: Dog,
    meters: Meters,
    timer: GameTimer,
    bill: RepairBill,
    state: GameState,
    cam_pitch: f32,
    cam_yaw: f32,
    cam_dist: f32,  // scroll zoom distance
    time: f32,
    frame_count: u32,
    screen_shake: f32,
    hitstop: f32,
    focused: bool,
    room_dirty: bool,
    particles: ParticleSystem,
    physics: PhysicsWorld,
    furniture: Vec<FurnitureObj>,
    // Attack targeting
    target_marker: Option<Vec3>,  // world position of attack target (mouse raycast)
    mouse_screen: (f32, f32),     // screen pixel coords of mouse
}

impl App {
    fn new() -> Self {
        let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs()).unwrap_or(42);
        let (room, furniture) = generate_apartment_v2(seed);
        let owners = [OwnerType::Normal, OwnerType::Gamer, OwnerType::Bookworm, OwnerType::Minimalist];
        let owner = owners[(seed % owners.len() as u64) as usize].clone();
        println!("  Owner: {:?}, Seed: {}", owner, seed);

        Self {
            win: None, renderer: None,
            room, cat: CatState::new(), input: Input::default(),
            dog: Dog::new(148.0, 22.0),
            meters: Meters::new(), timer: GameTimer::new(),
            bill: RepairBill::new(owner),
            state: GameState::Menu,
            cam_pitch: 0.55, cam_yaw: 0.0, cam_dist: 200.0, time: 0.0, frame_count: 0,
            screen_shake: 0.0, hitstop: 0.0,
            focused: true, room_dirty: true,
            particles: ParticleSystem::new(),
            physics: PhysicsWorld::new(FLOOR_Y as f32 + 1.0),
            furniture,
            target_marker: None,
            mouse_screen: (640.0, 360.0),
        }
    }

    fn update(&mut self, dt: f32) {
        if !self.state.is_playing() { return; }
        if self.hitstop > 0.0 { self.hitstop -= dt; return; }
        if self.screen_shake > 0.0 { self.screen_shake -= dt; }

        let cat = &mut self.cat;
        let spd = if self.input.sprint { MOVE_SPEED * 2.2 } else { MOVE_SPEED };

        // AD = turn cat (responsive turning)
        let turn_speed = 4.5 * dt;
        if self.input.left  { cat.facing += turn_speed; }
        if self.input.right { cat.facing -= turn_speed; }

        // WS = move forward/backward along cat's facing direction
        let fwd = cat.forward();
        let (mut nx, mut nz) = (cat.x, cat.z);
        let mut moved = false;
        if self.input.forward { nx += fwd.x*spd*dt; nz += fwd.z*spd*dt; moved = true; }
        if self.input.back    { nx -= fwd.x*spd*dt*0.6; nz -= fwd.z*spd*dt*0.6; moved = true; }
        nx = nx.clamp(14.0, GRID as f32 - 14.0);
        nz = nz.clamp(14.0, GRID as f32 - 14.0);

        let cat_r = 5.0; // small enough to fit through doors easily
        let can_x = !self.room.collides(nx, cat.z, cat.y, cat_r);
        let can_z = !self.room.collides(cat.x, nz, cat.y, cat_r);

        // BODY SLAM: running into objects knocks them over
        if moved && self.input.sprint {
            let bump_pos = Vec3::new(
                if !can_x { nx } else { cat.x } + fwd.x * 3.0,
                cat.y,
                if !can_z { nz } else { cat.z } + fwd.z * 3.0,
            );
            if !can_x || !can_z {
                // Destroy voxels at collision point
                let fwd_v = cat.forward();
                let rgt_v = cat.right();
                let debris = self.room.scratch_at(bump_pos, fwd_v, rgt_v);
                if !debris.is_empty() {
                    self.particles.spawn_debris(bump_pos, &debris, fwd_v);
                    // Rebuild affected chunk
                    if let Some(r) = self.renderer.as_mut() {
                        r.rebuild_chunk_at(&self.room, GRID, bump_pos.x, bump_pos.y, bump_pos.z);
                    }
                    self.screen_shake = 0.1;
                    let value = debris.len() as f32 * 1.5;
                    self.meters.on_destroy(value, 0.3);
                    self.bill.record("Furniture", "body slam", value, 1.0);
                }
            }
        }

        if can_x { cat.x = nx; }
        if can_z { cat.z = nz; }
        cat.moving = moved && (can_x || can_z);
        let walk_speed = if self.input.sprint { 2.5 } else { 1.2 };
        if cat.moving { cat.walk_phase += dt * walk_speed; }

        // Jump
        if self.input.jump && cat.vy.abs() < 1.0 {
            let fl = self.room.floor_at(cat.x, cat.z, cat.y);
            if (cat.y - LEG_HEIGHT - fl).abs() < 3.0 { cat.vy = JUMP_VEL; }
        }
        cat.vy -= GRAVITY * dt;
        cat.y += cat.vy * dt;
        let fl = self.room.floor_at(cat.x, cat.z, cat.y + LEG_HEIGHT);
        let min_y = fl + LEG_HEIGHT;
        cat.in_air = cat.vy.abs() > 2.0 || (cat.y - min_y) > 1.5;
        if cat.y < min_y {
            if cat.vy < -10.0 { cat.land_timer = 0.2; }
            cat.y = min_y; cat.vy = 0.0; cat.in_air = false;
        }
        if cat.land_timer > 0.0 { cat.land_timer -= dt; }

        // Dog AI
        let noise = if cat.scratching && cat.scratch_fired { 0.6 } else { 0.0 };
        let cat_x = cat.x; let cat_z = cat.z;
        let dog_events = self.dog.update(dt, cat_x, cat_z, noise);
        for ev in &dog_events {
            if let DogEvent::WokeUp = ev {
                self.meters.annoyance = (self.meters.annoyance + 5.0).min(100.0);
            }
        }
        if self.dog.is_blocking() && self.dog.distance_to(cat.x, cat.z) < 12.0 {
            let dx = cat.x - self.dog.x;
            let dz = cat.z - self.dog.z;
            let d = (dx*dx+dz*dz).sqrt().max(0.1);
            cat.x += dx/d * 0.5;
            cat.z += dz/d * 0.5;
        }

        // Parrot attraction
        let parrot_dist = ((cat.x - 105.0).powi(2) + (cat.z - 120.0).powi(2)).sqrt();
        if parrot_dist < 15.0 { self.meters.mild_entertainment(dt * 2.0); }

        // Scratch
        if self.input.scratch_pressed {
            self.input.scratch_pressed = false;
            self.cat.scratching = true; self.cat.scratch_phase = 0.0; self.cat.scratch_fired = false;
        }
        if self.cat.scratching {
            let old = self.cat.scratch_phase;
            self.cat.scratch_phase += dt * SCRATCH_SPEED;
            if !self.cat.scratch_fired && old < 0.45 && self.cat.scratch_phase >= 0.45 {
                self.cat.scratch_fired = true;
                // Hit where the marker is (mouse-aimed), or fallback to forward
                let cat_pos = Vec3::new(self.cat.x, self.cat.y, self.cat.z);
                let hit = self.target_marker.unwrap_or(cat_pos + self.cat.forward() * SCRATCH_RANGE);
                let hit_low = Vec3::new(hit.x, hit.y - LEG_HEIGHT * 0.3, hit.z);
                let fwd = (hit - cat_pos).normalize();
                let rgt = fwd.cross(Vec3::Y).normalize();
                let d1 = self.room.scratch_at(hit, fwd, rgt);
                let d2 = self.room.scratch_at(hit_low, fwd, rgt);
                // Also scratch furniture objects!
                for f in &mut self.furniture {
                    if f.shattered { continue; }
                    f.scratch_world(hit, 4.0);
                    f.scratch_world(hit_low, 4.0);
                }

                // VKUSNO: spawn debris particles flying outward!
                self.particles.spawn_debris(hit, &d1, fwd);
                self.particles.spawn_debris(hit_low, &d2, fwd);

                self.screen_shake = 0.15;
                self.hitstop = 0.05;

                // Furniture objects handle their own falling via support check.
                // Room grid connectivity not needed — walls don't float.

                // Chunk rebuild at hit point
                if let Some(r) = self.renderer.as_mut() {
                    r.rebuild_chunk_at(&self.room, GRID, hit.x, hit.y, hit.z);
                    r.rebuild_chunk_at(&self.room, GRID, hit_low.x, hit_low.y, hit_low.z);
                }

                let destroyed = d1.len() + d2.len();
                if destroyed > 0 {
                    let avg_mat = if !d1.is_empty() { (d1[0].1.packed & 0xFF) as u8 } else { 1 };
                    let mat_info = items::material_by_id(avg_mat);
                    let value = destroyed as f32 * 2.0;
                    self.meters.on_destroy(value, mat_info.noise);
                    self.bill.record("Furniture", mat_info.name, value, 1.0);
                }
            }
            if self.cat.scratch_phase >= 1.0 { self.cat.scratching = false; }
        }

        // Physics step — falling objects, collisions
        let phys_events = self.physics.step(dt);
        for ev in &phys_events {
            match ev {
                PhysicsEvent::FloorHit { id, speed } => {
                    // Falling object hit floor → screen shake + particles
                    if *speed > 20.0 {
                        self.screen_shake = (*speed / 100.0).min(0.4);
                    }
                    // Find the body and spawn impact particles
                    if let Some(body) = self.physics.bodies.iter().find(|b| b.id == *id) {
                        for _ in 0..(*speed as usize / 5).min(20) {
                            let scatter = ((body.pos.x * 13.0 + body.pos.z * 7.0) % 5.0) - 2.5;
                            self.particles.particles.push(Particle {
                                x: body.pos.x + scatter, y: body.pos.y, z: body.pos.z + scatter,
                                vx: scatter * 8.0, vy: *speed * 0.3, vz: scatter * 8.0,
                                color: [180, 160, 130], life: 1.0,
                            });
                        }
                    }
                }
                PhysicsEvent::Collision { impulse, .. } => {
                    if *impulse > 50.0 { self.screen_shake = 0.08; }
                }
                _ => {}
            }
        }
        // Clean up settled/inactive physics bodies
        self.physics.bodies.retain(|b| b.active && b.pos.y > -50.0);

        // Furniture support check — objects without support START FALLING
        // Check room grid AND other furniture for support below
        let furn_count = self.furniture.len();
        for i in 0..furn_count {
            if self.furniture[i].falling || self.furniture[i].shattered { continue; }
            // Object on floor level = always supported
            if self.furniture[i].pos.y <= FLOOR_Y as f32 + 2.0 { continue; }

            let mut supported = false;
            let gs = self.furniture[i].grid_size;

            'outer: for z in 0..gs {
                for x in 0..gs {
                    for y in 0..gs {
                        if self.furniture[i].get(x, y, z).is_solid() {
                            let wx = self.furniture[i].pos.x + x as f32;
                            let wy = self.furniture[i].pos.y + y as f32 - 1.0;
                            let wz = self.furniture[i].pos.z + z as f32;

                            // Check room grid
                            let rwx = wx as usize; let rwy = wy as usize; let rwz = wz as usize;
                            if rwx < GRID && rwy < GRID && rwz < GRID && self.room.get(rwx, rwy, rwz).is_solid() {
                                supported = true;
                                break 'outer;
                            }

                            // Check OTHER furniture (not self, not falling/shattered)
                            for j in 0..furn_count {
                                if j == i { continue; }
                                if self.furniture[j].falling || self.furniture[j].shattered { continue; }
                                if self.furniture[j].contains_world(wx, wy, wz) {
                                    supported = true;
                                    break 'outer;
                                }
                            }

                            break; // only check lowest solid per column
                        }
                    }
                }
            }

            if !supported {
                self.furniture[i].falling = true;
                self.furniture[i].vel = Vec3::new(0.0, -5.0, 0.0);
                println!("  {} lost support — FALLING!", self.furniture[i].name);
            }
        }

        // Update falling furniture
        let floor = FLOOR_Y as f32 + 1.0;
        for f in &mut self.furniture {
            let shattered = f.update(dt, floor);
            if shattered {
                // Object shattered on impact — spawn particles
                self.screen_shake = 0.2;
                for z in 0..f.grid_size { for y in 0..f.grid_size { for x in 0..f.grid_size {
                    if f.get(x, y, z).is_solid() {
                        let v = f.get(x, y, z);
                        self.particles.particles.push(Particle {
                            x: f.pos.x + x as f32, y: f.pos.y + y as f32, z: f.pos.z + z as f32,
                            vx: (x as f32 - f.grid_size as f32 * 0.5) * 8.0,
                            vy: 15.0,
                            vz: (z as f32 - f.grid_size as f32 * 0.5) * 8.0,
                            color: [(v.packed >> 16) as u8, (v.packed >> 8) as u8, v.packed as u8],
                            life: 2.0,
                        });
                    }
                }}}
                println!("  {} SHATTERED! ${:.0} damage", f.name, f.value);
                self.bill.record(&f.name, "shattered", f.value, 1.0);
            }
            if f.falling { f.mesh_dirty = true; }
        }

        // Meters & Timer
        if let Some(game_over) = self.meters.update(dt) {
            let reason = match game_over {
                GameOver::BoredToDeath => "BORED TO DEATH!",
                GameOver::ThrownOut => "THROWN OUT!",
                GameOver::NoLivesLeft => "NO LIVES LEFT!",
            };
            self.state = GameState::Over(reason.to_string());
        }
        if self.timer.update(dt) {
            self.state = GameState::Over("OWNER CAME HOME!".to_string());
        }

        // HUD in title
        if self.frame_count % 30 == 0 {
            if let Some(w) = &self.win {
                let dog_s = if self.dog.is_sleeping() { "Zzz" } else if self.dog.is_blocking() { "!!" } else { "..." };
                w.set_title(&format!("PURRGE Build 12 Leopold | {} | Boredom:{:.0}% | Annoy:{:.0}% | Lives:{} | ${:.0} | Dog:{}",
                    self.timer.clock_display(), self.meters.boredom, self.meters.annoyance,
                    self.meters.lives, self.bill.total(), dog_s));
            }
        }
    }

    fn render(&mut self) {
        let dt = 1.0 / 60.0;
        self.time += dt;
        self.frame_count += 1;
        if self.state.is_playing() { self.update(dt); }
        self.particles.update(dt);

        // Render particles + target marker as debris mesh
        let has_marker = self.target_marker.is_some() && !self.cat.scratching;
        if (!self.particles.particles.is_empty() || has_marker) && self.frame_count % 2 == 0 {
            // Build small mesh from particles using simple quads
            let mut verts: Vec<f32> = Vec::new();
            let mut indices: Vec<u32> = Vec::new();
            let mut vi = 0u32;
            for p in &self.particles.particles {
                let alpha = (p.life / 1.5).clamp(0.2, 1.0);
                let r = p.color[0] as f32 / 255.0 * alpha;
                let g = p.color[1] as f32 / 255.0 * alpha;
                let b = p.color[2] as f32 / 255.0 * alpha;
                let s = 1.5; // particle size
                // 6 faces of a cube
                for &(nx,ny,nz, dx0,dy0,dz0, dx1,dy1,dz1, dx2,dy2,dz2, dx3,dy3,dz3) in &[
                    (0.0,0.0,1.0, -s,-s,s, s,-s,s, s,s,s, -s,s,s),
                    (0.0,0.0,-1.0, s,-s,-s, -s,-s,-s, -s,s,-s, s,s,-s),
                    (0.0,1.0,0.0, -s,s,s, s,s,s, s,s,-s, -s,s,-s),
                    (0.0,-1.0,0.0, -s,-s,-s, s,-s,-s, s,-s,s, -s,-s,s),
                    (1.0,0.0,0.0, s,-s,s, s,-s,-s, s,s,-s, s,s,s),
                    (-1.0,0.0,0.0, -s,-s,-s, -s,-s,s, -s,s,s, -s,s,-s),
                ] {
                    for &(dx,dy,dz) in &[(dx0,dy0,dz0),(dx1,dy1,dz1),(dx2,dy2,dz2),(dx3,dy3,dz3)] {
                        verts.extend_from_slice(&[p.x+dx, p.y+dy, p.z+dz, nx, ny, nz, r, g, b, 1.0]);
                    }
                    indices.extend_from_slice(&[vi,vi+1,vi+2, vi,vi+2,vi+3]);
                    vi += 4;
                }
            }
            // Add target marker (pulsing cross)
            if let Some(marker) = self.target_marker {
                if !self.cat.scratching {
                    let pulse = 0.5 + 0.5 * (self.time * 8.0).sin();
                    let mr = 0.9 * pulse + 0.1;
                    let mg = 0.3 + 0.5 * pulse;
                    let mb = 0.1;
                    let ms = 1.8; // marker cube size
                    // Cross: 7 cubes
                    for &(dx, dy, dz) in &[(0.0,0.0,0.0),(ms*2.0,0.0,0.0),(-ms*2.0,0.0,0.0),
                                           (0.0,ms*2.0,0.0),(0.0,-ms*2.0,0.0),(0.0,0.0,ms*2.0),(0.0,0.0,-ms*2.0)] {
                        let cx = marker.x + dx;
                        let cy = marker.y + dy;
                        let cz = marker.z + dz;
                        for &(nx,ny,nz, dx0,dy0,dz0, dx1,dy1,dz1, dx2,dy2,dz2, dx3,dy3,dz3) in &[
                            (0.0,0.0,1.0, -ms,-ms,ms, ms,-ms,ms, ms,ms,ms, -ms,ms,ms),
                            (0.0,0.0,-1.0, ms,-ms,-ms, -ms,-ms,-ms, -ms,ms,-ms, ms,ms,-ms),
                            (0.0,1.0,0.0, -ms,ms,ms, ms,ms,ms, ms,ms,-ms, -ms,ms,-ms),
                            (0.0,-1.0,0.0, -ms,-ms,-ms, ms,-ms,-ms, ms,-ms,ms, -ms,-ms,ms),
                            (1.0,0.0,0.0, ms,-ms,ms, ms,-ms,-ms, ms,ms,-ms, ms,ms,ms),
                            (-1.0,0.0,0.0, -ms,-ms,-ms, -ms,-ms,ms, -ms,ms,ms, -ms,ms,-ms),
                        ] {
                            for &(ddx,ddy,ddz) in &[(dx0,dy0,dz0),(dx1,dy1,dz1),(dx2,dy2,dz2),(dx3,dy3,dz3)] {
                                verts.extend_from_slice(&[cx+ddx, cy+ddy, cz+ddz, nx, ny, nz, mr, mg, mb, 1.0]);
                            }
                            indices.extend_from_slice(&[vi,vi+1,vi+2, vi,vi+2,vi+3]);
                            vi += 4;
                        }
                    }
                }
            }

            if let Some(renderer) = self.renderer.as_mut() {
                let vb = renderer.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Particles"), contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let ib = renderer.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Particle Idx"), contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
                renderer.debris_mesh = Some(GpuMesh { vertex_buffer: vb, index_buffer: ib, index_count: indices.len() as u32 });
            }
        } else if self.particles.particles.is_empty() && !has_marker {
            if let Some(r) = self.renderer.as_mut() { r.debris_mesh = None; }
        }

        let renderer = self.renderer.as_mut().unwrap();

        // Build room chunks (only on init)
        if self.room_dirty {
            renderer.upload_room(&self.room, GRID);
            self.room_dirty = false;
            println!("  Room chunks built.");
        }

        // Rebuild cat mesh every 3 frames — Surface Nets + world_scale
        if self.frame_count % 3 == 0 {
            let cat_voxels = rasterize_cat_to_grid(&self.cat, self.time);
            let ws = CAT_WORLD_SCALE;
            let mesh = crate::core::meshing::generate_mesh_smooth_with_ao(
                &cat_voxels, CAT_GRID,
                Vec3::new(
                    self.cat.x - CAT_GRID as f32 * 0.5 * ws,
                    self.cat.y - CAT_GRID as f32 * 0.275 * ws,
                    self.cat.z - CAT_GRID as f32 * 0.5 * ws,
                ),
                ws,
            );
            renderer.cat_mesh = crate::core::render_mesh::GpuMesh::from_chunk_mesh(&renderer.device, &mesh);
        }

        // Rebuild dog mesh every 6 frames — Surface Nets + world_scale
        if self.frame_count % 6 == 0 {
            let dog_voxels = rasterize_dog_to_grid(&self.dog, self.time);
            let ws = DOG_WORLD_SCALE;
            let mesh = crate::core::meshing::generate_mesh_smooth_with_ao(
                &dog_voxels, DOG_GRID,
                Vec3::new(
                    self.dog.x - DOG_GRID as f32 * 0.5 * ws,
                    FLOOR_Y as f32,
                    self.dog.z - DOG_GRID as f32 * 0.5 * ws,
                ),
                ws,
            );
            renderer.dog_mesh = crate::core::render_mesh::GpuMesh::from_chunk_mesh(&renderer.device, &mesh);
        }

        // Rebuild furniture meshes (only when dirty — position changed, damaged, etc.)
        if renderer.furniture_meshes.len() != self.furniture.len() {
            renderer.furniture_meshes = (0..self.furniture.len()).map(|_| None).collect();
        }
        for (i, f) in self.furniture.iter_mut().enumerate() {
            if f.shattered { renderer.furniture_meshes[i] = None; continue; }
            if f.mesh_dirty || renderer.furniture_meshes[i].is_none() {
                let mesh = f.build_mesh();
                renderer.furniture_meshes[i] = crate::core::render_mesh::GpuMesh::from_chunk_mesh(&renderer.device, &mesh);
                f.mesh_dirty = false;
            }
        }

        // ── MENU MODE: cinematic camera + title overlay ──
        if self.state.is_menu() {
            let center = Vec3::new(GRID as f32 * 0.5, 40.0, GRID as f32 * 0.5);
            let menu_angle = self.time * 0.15;
            let eye = center + Vec3::new(menu_angle.sin() * 120.0, 100.0, menu_angle.cos() * 120.0);

            // Menu HUD overlay
            let mut hud = Vec::new();
            // Dark backdrop
            hud_bar(&mut hud, -1.0, -0.15, 2.0, 0.3, 0.0, 0.0, 0.0, 0.7);
            // Title bar (orange)
            hud_bar(&mut hud, -0.6, 0.0, 1.2, 0.08, 0.9, 0.5, 0.1, 1.0);
            // "Press SPACE" hint bar (dim)
            hud_bar(&mut hud, -0.35, -0.1, 0.7, 0.04, 0.5, 0.5, 0.5, 0.8);

            renderer.upload_hud(&hud);
            renderer.render(eye, center, 0.0, self.time);
            if let Some(w) = &self.win {
                w.set_title("PURRGE — Press SPACE / Enter / Click to play!");
            }
            self.win.as_ref().unwrap().request_redraw();
            return;
        }

        // Camera — free orbit around cat (mouse controls camera, AD controls cat)
        let cat_pos = Vec3::new(self.cat.x, self.cat.y, self.cat.z);
        let cam_dist = self.cam_dist;
        let cam_dir = Vec3::new(
            self.cam_yaw.sin() * self.cam_pitch.cos(),
            self.cam_pitch.sin(),
            self.cam_yaw.cos() * self.cam_pitch.cos(),
        );
        let mut eye = cat_pos + cam_dir * cam_dist;
        let target = cat_pos + Vec3::Y * 5.0;

        // Camera collision
        let cam_dir = (eye - cat_pos).normalize();
        let cam_len = (eye - cat_pos).length();
        if let Some(wall) = self.room.raycast(cat_pos + Vec3::Y * 5.0, cam_dir, cam_len) {
            let safe = (wall - cat_pos).length() - 4.0;
            if safe > 8.0 && safe < cam_len { eye = cat_pos + cam_dir * safe; }
            else if safe <= 8.0 { eye = cat_pos + cam_dir * 8.0 + Vec3::Y * 10.0; }
        }

        // Attack target marker — raycast from camera through mouse position
        let view_mat = Mat4::look_at_rh(eye, target, Vec3::Y);
        let proj_mat = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, renderer.config.width as f32 / renderer.config.height as f32, 0.1, 1000.0);
        let inv_vp = (proj_mat * view_mat).inverse();
        let ndc_x = (2.0 * self.mouse_screen.0 / renderer.config.width as f32) - 1.0;
        let ndc_y = 1.0 - (2.0 * self.mouse_screen.1 / renderer.config.height as f32);
        let near = inv_vp * glam::Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let far = inv_vp * glam::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        let ray_origin = near.truncate() / near.w;
        let ray_target = far.truncate() / far.w;
        let ray_dir = (ray_target - ray_origin).normalize();

        // Find hit point in room
        let max_range = 50.0; // max attack range from cat
        if let Some(hit) = self.room.raycast(ray_origin, ray_dir, 500.0) {
            let dist_to_cat = (hit - cat_pos).length();
            if dist_to_cat < max_range {
                self.target_marker = Some(hit);
            } else {
                // Clamp to max range along ray direction from cat
                let dir_to_hit = (hit - cat_pos).normalize();
                self.target_marker = Some(cat_pos + dir_to_hit * max_range);
            }
        } else {
            self.target_marker = None;
        }

        // Screen shake
        if self.screen_shake > 0.0 {
            let amt = self.screen_shake * 4.0;
            eye.x += (self.time * 47.0).sin() * amt;
            eye.y += (self.time * 53.0).cos() * amt;
        }

        // ── HUD BARS ──
        let mut hud = Vec::new();
        let bar_h = 0.025;
        let bar_y_start = 0.92;
        let bar_w = 0.4;
        let bar_x = -0.95;

        // Background bars (dark)
        hud_bar(&mut hud, bar_x, bar_y_start, bar_w, bar_h, 0.1, 0.1, 0.1, 0.6);
        hud_bar(&mut hud, bar_x, bar_y_start - bar_h * 1.5, bar_w, bar_h, 0.1, 0.1, 0.1, 0.6);
        hud_bar(&mut hud, bar_x, bar_y_start - bar_h * 3.0, bar_w, bar_h, 0.1, 0.1, 0.1, 0.6);

        // Boredom bar (yellow → green as it decreases — high = bad)
        let bored = self.meters.boredom / 100.0;
        hud_bar(&mut hud, bar_x, bar_y_start, bar_w * bored, bar_h, 0.9, 0.8, 0.1, 0.9);

        // Annoyance bar (red — high = thrown out)
        let annoy = self.meters.annoyance / 100.0;
        hud_bar(&mut hud, bar_x, bar_y_start - bar_h * 1.5, bar_w * annoy, bar_h, 0.9, 0.2, 0.15, 0.9);

        // Damage $ bar (gold — grows with destruction)
        let dmg = (self.bill.total() / 5000.0).min(1.0);
        hud_bar(&mut hud, bar_x, bar_y_start - bar_h * 3.0, bar_w * dmg, bar_h, 0.95, 0.75, 0.2, 0.9);

        // Lives dots (green circles → red)
        for i in 0..9 {
            let alive = (i as f32) < self.meters.lives as f32;
            let dot_x = 0.55 + i as f32 * 0.045;
            let (r, g, b) = if alive { (0.2, 0.85, 0.3) } else { (0.3, 0.15, 0.1) };
            hud_bar(&mut hud, dot_x, bar_y_start, 0.03, bar_h, r, g, b, 0.9);
        }

        renderer.upload_hud(&hud);
        renderer.render(eye, target, self.screen_shake, self.time);
        self.win.as_ref().unwrap().request_redraw();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &winit::event_loop::ActiveEventLoop) {
        if self.win.is_some() { return; }
        let w = Arc::new(el.create_window(Window::default_attributes()
            .with_title("PURRGE — Build 12 Leopold")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))).unwrap());

        // Don't grab cursor in menu — grab on game start
        self.renderer = Some(Renderer::new(w.clone()));
        self.room_dirty = true;
        self.win = Some(w);
    }

    fn device_event(&mut self, _: &winit::event_loop::ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        if !self.focused || !self.state.is_playing() { return; }
        if let DeviceEvent::MouseMotion { delta } = event {
            // Mouse controls CAMERA, not cat. AD controls cat turning.
            self.cam_yaw -= delta.0 as f32 * MOUSE_SENS;
            self.cam_pitch = (self.cam_pitch + delta.1 as f32 * MOUSE_SENS).clamp(0.15, 1.1);
        }
    }

    fn window_event(&mut self, el: &winit::event_loop::ActiveEventLoop, _: winit::window::WindowId, ev: WindowEvent) {
        match ev {
            WindowEvent::Focused(f) => {
                self.focused = f;
            }
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::RedrawRequested => self.render(),
            WindowEvent::Resized(s) => {
                if let Some(r) = self.renderer.as_mut() { r.resize(s.width, s.height); }
            }
            WindowEvent::MouseInput { button: MouseButton::Left, state, .. } => {
                if state.is_pressed() {
                    if self.state.is_menu() {
                        self.state = GameState::Playing;
                        if let Some(w) = &self.win {
                            w.set_cursor_grab(CursorGrabMode::Confined).or_else(|_| w.set_cursor_grab(CursorGrabMode::Locked)).ok();
                            w.set_cursor_visible(false);
                        }
                    } else if self.state.is_playing() {
                        if !self.cat.scratching { self.input.scratch_pressed = true; }
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_screen = (position.x as f32, position.y as f32);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 / 50.0,
                };
                self.cam_dist = (self.cam_dist - scroll * 20.0).clamp(30.0, 800.0);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let p = event.state.is_pressed();
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::KeyW) => self.input.forward = p,
                    PhysicalKey::Code(KeyCode::KeyS) => self.input.back = p,
                    PhysicalKey::Code(KeyCode::KeyA) => self.input.left = p,
                    PhysicalKey::Code(KeyCode::KeyD) => self.input.right = p,
                    PhysicalKey::Code(KeyCode::Space) => {
                        if p && self.state.is_menu() {
                            self.state = GameState::Playing;
                            if let Some(w) = &self.win {
                                w.set_cursor_grab(CursorGrabMode::Confined).or_else(|_| w.set_cursor_grab(CursorGrabMode::Locked)).ok();
                                w.set_cursor_visible(false);
                            }
                        } else {
                            self.input.jump = p;
                        }
                    }
                    PhysicalKey::Code(KeyCode::ShiftLeft) | PhysicalKey::Code(KeyCode::ShiftRight) => self.input.sprint = p,
                    PhysicalKey::Code(KeyCode::KeyE) => { if p && !self.cat.scratching { self.input.scratch_pressed = true; } }
                    PhysicalKey::Code(KeyCode::Enter) if p && self.state.is_menu() => {
                        self.state = GameState::Playing;
                        if let Some(w) = &self.win {
                            w.set_cursor_grab(CursorGrabMode::Confined).or_else(|_| w.set_cursor_grab(CursorGrabMode::Locked)).ok();
                            w.set_cursor_visible(false);
                        }
                    }
                    PhysicalKey::Code(KeyCode::Escape) if p => {
                        match &self.state {
                            GameState::Menu => { el.exit(); }
                            GameState::Playing => { self.state = GameState::Paused; }
                            GameState::Paused => { self.state = GameState::Playing; }
                            GameState::Over(_) => {
                                self.state = GameState::Bill;
                                println!("\n{}", self.bill.format_bill());
                                println!("  Rating: {}\n", self.bill.rating());
                            }
                            GameState::Bill => { el.exit(); }
                        }
                    }
                    PhysicalKey::Code(KeyCode::KeyQ) if p && self.state == GameState::Paused => {
                        el.exit();
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();
    println!();
    println!("  ═══════════════════════════════════════");
    println!("  \u{1F431} PURRGE — Build 12 \"Leopold\"");
    println!("     Mesh pipeline. Surface Nets. PBR.");
    println!("     No more cubes.");
    println!("  ═══════════════════════════════════════");
    println!();
    println!("  Mouse  = look       WASD   = move");
    println!("  Space  = jump       Shift  = sprint");
    println!("  LMB/E  = scratch!   Esc    = pause");
    println!();
    let el = EventLoop::new().unwrap();
    let mut app = App::new();
    el.run_app(&mut app).unwrap();
}
