// ═══════════════════════════════════════════════════════════════
// PURRGE — Build 2 "Whiskers"
// Mesh pipeline (Surface Nets + PBR). No more cubes.
// Procedural apartment, dog AI, parrot, meters, scoring.
// ═══════════════════════════════════════════════════════════════

mod core;
mod game;
mod render;
mod apartment;

use crate::apartment::{VoxelGrid, generate_apartment, GRID, FLOOR_Y};
use crate::core::body::BodyDefinition;
use crate::core::skeleton::Skeleton;
use crate::core::svo::Voxel;
use crate::game::dog::{Dog, DogEvent};
use crate::game::meters::{Meters, GameOver};
use crate::game::scoring::{RepairBill, OwnerType};
use crate::game::timer::GameTimer;
use crate::game::items;
use crate::render::Renderer;
use glam::{Mat4, Quat, Vec3};
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

#[derive(PartialEq)]
enum GameState { Playing, Paused, Over(String), Bill }
// Need custom PartialEq for String variant
impl GameState {
    fn is_playing(&self) -> bool { matches!(self, GameState::Playing) }
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
    // Scratch
    if cat.scratching {
        let p = cat.scratch_phase;
        if p < 0.4 {
            let s = { let l = p/0.4; l*l*(3.0-2.0*l) };
            sk.set_rotation("upper_arm_r", Quat::from_axis_angle(Vec3::X, -1.6*s)*Quat::from_axis_angle(Vec3::Z, 0.6*s));
            sk.set_rotation("forearm_r", Quat::from_axis_angle(Vec3::X, 2.0*s));
            sk.set_rotation("spine2", Quat::from_axis_angle(Vec3::X, -0.2*s));
            sk.set_rotation("ear_l", Quat::from_axis_angle(Vec3::X, 0.4*s));
            sk.set_rotation("ear_r", Quat::from_axis_angle(Vec3::X, 0.4*s));
        } else if p < 0.55 {
            let s = { let l = (p-0.4)/0.15; l*l };
            sk.set_rotation("upper_arm_r", Quat::from_axis_angle(Vec3::X, -1.6+3.2*s)*Quat::from_axis_angle(Vec3::Z, 0.6*(1.0-s)));
            sk.set_rotation("forearm_r", Quat::from_axis_angle(Vec3::X, 2.0*(1.0-s)));
            sk.set_rotation("spine2", Quat::from_axis_angle(Vec3::X, -0.2+0.35*s));
        } else {
            let s = { let l = ((p-0.55)/0.45).min(1.0); l*l*(3.0-2.0*l) };
            sk.set_rotation("upper_arm_r", Quat::from_axis_angle(Vec3::X, 1.6*(1.0-s)));
            sk.set_rotation("forearm_r", Quat::from_axis_angle(Vec3::X, 0.05));
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
    sk.root_rotation = Quat::IDENTITY;
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
        // Standing / alert
        let breath = 0.03 * (time * 1.0).sin();
        sk.set_rotation("spine2", Quat::from_axis_angle(Vec3::X, breath));
        sk.set_rotation("head", Quat::from_axis_angle(Vec3::X, 0.05 * (time * 1.5).sin()));
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
    time: f32,
    frame_count: u32,
    screen_shake: f32,
    hitstop: f32,
    focused: bool,
    room_dirty: bool, // need to rebuild room mesh
}

impl App {
    fn new() -> Self {
        let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs()).unwrap_or(42);
        let room = generate_apartment(seed);
        let owners = [OwnerType::Normal, OwnerType::Gamer, OwnerType::Bookworm, OwnerType::Minimalist];
        let owner = owners[(seed % owners.len() as u64) as usize].clone();
        println!("  Owner: {:?}, Seed: {}", owner, seed);

        Self {
            win: None, renderer: None,
            room, cat: CatState::new(), input: Input::default(),
            dog: Dog::new(148.0, 22.0),
            meters: Meters::new(), timer: GameTimer::new(),
            bill: RepairBill::new(owner),
            state: GameState::Playing,
            cam_pitch: 0.55, time: 0.0, frame_count: 0,
            screen_shake: 0.0, hitstop: 0.0,
            focused: true, room_dirty: true,
        }
    }

    fn update(&mut self, dt: f32) {
        if !self.state.is_playing() { return; }
        if self.hitstop > 0.0 { self.hitstop -= dt; return; }
        if self.screen_shake > 0.0 { self.screen_shake -= dt; }

        let cat = &mut self.cat;
        let fwd = cat.forward();
        let rgt = cat.right();
        let spd = if self.input.sprint { MOVE_SPEED * 2.2 } else { MOVE_SPEED };
        let (mut nx, mut nz) = (cat.x, cat.z);
        let mut moved = false;
        if self.input.forward { nx += fwd.x*spd*dt; nz += fwd.z*spd*dt; moved = true; }
        if self.input.back    { nx -= fwd.x*spd*dt*0.6; nz -= fwd.z*spd*dt*0.6; moved = true; }
        if self.input.left    { nx -= rgt.x*spd*dt*0.7; nz -= rgt.z*spd*dt*0.7; moved = true; }
        if self.input.right   { nx += rgt.x*spd*dt*0.7; nz += rgt.z*spd*dt*0.7; moved = true; }
        nx = nx.clamp(14.0, GRID as f32 - 14.0);
        nz = nz.clamp(14.0, GRID as f32 - 14.0);

        let cat_r = 5.0 * CAT_WORLD_SCALE;
        let can_x = !self.room.collides(nx, cat.z, cat.y, cat_r);
        let can_z = !self.room.collides(cat.x, nz, cat.y, cat_r);
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
                let fwd = self.cat.forward(); let rgt = self.cat.right();
                let hit = Vec3::new(self.cat.x, self.cat.y, self.cat.z) + fwd * SCRATCH_RANGE;
                let hit_low = Vec3::new(self.cat.x, self.cat.y - LEG_HEIGHT*0.5, self.cat.z) + fwd * SCRATCH_RANGE * 0.9;
                let d1 = self.room.scratch_at(hit, fwd, rgt);
                let d2 = self.room.scratch_at(hit_low, fwd, rgt);
                self.screen_shake = 0.12;
                self.hitstop = 0.04;
                self.room_dirty = true; // need mesh rebuild!

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
                w.set_title(&format!("PURRGE | {} | Boredom:{:.0}% | Annoy:{:.0}% | Lives:{} | ${:.0} | Dog:{}",
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

        let renderer = self.renderer.as_mut().unwrap();

        // Rebuild room mesh when damaged
        if self.room_dirty {
            renderer.upload_room(&self.room.data, GRID);
            self.room_dirty = false;
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

        // Camera
        let fwd = self.cat.forward();
        let cat_pos = Vec3::new(self.cat.x, self.cat.y, self.cat.z);
        let cam_dist = 70.0;
        let mut eye = cat_pos - fwd * cam_dist * self.cam_pitch.cos() + Vec3::Y * cam_dist * self.cam_pitch.sin();
        let target = cat_pos + fwd * 15.0 - Vec3::Y * 5.0;

        // Camera collision
        let cam_dir = (eye - cat_pos).normalize();
        let cam_len = (eye - cat_pos).length();
        if let Some(wall) = self.room.raycast(cat_pos + Vec3::Y * 5.0, cam_dir, cam_len) {
            let safe = (wall - cat_pos).length() - 4.0;
            if safe > 8.0 && safe < cam_len { eye = cat_pos + cam_dir * safe; }
            else if safe <= 8.0 { eye = cat_pos + cam_dir * 8.0 + Vec3::Y * 10.0; }
        }

        // Screen shake
        if self.screen_shake > 0.0 {
            let amt = self.screen_shake * 4.0;
            eye.x += (self.time * 47.0).sin() * amt;
            eye.y += (self.time * 53.0).cos() * amt;
        }

        renderer.render(eye, target, self.screen_shake, self.time);
        self.win.as_ref().unwrap().request_redraw();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &winit::event_loop::ActiveEventLoop) {
        if self.win.is_some() { return; }
        let w = Arc::new(el.create_window(Window::default_attributes()
            .with_title("PURRGE — Build 2 Whiskers (Mesh Pipeline)")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))).unwrap());

        w.set_cursor_grab(CursorGrabMode::Confined).or_else(|_| w.set_cursor_grab(CursorGrabMode::Locked)).ok();
        w.set_cursor_visible(false);

        self.renderer = Some(Renderer::new(w.clone()));
        self.room_dirty = true; // trigger initial room mesh build
        self.win = Some(w);
    }

    fn device_event(&mut self, _: &winit::event_loop::ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        if !self.focused || !self.state.is_playing() { return; }
        if let DeviceEvent::MouseMotion { delta } = event {
            self.cat.facing -= delta.0 as f32 * MOUSE_SENS;
            self.cam_pitch = (self.cam_pitch + delta.1 as f32 * MOUSE_SENS).clamp(0.15, 1.1);
        }
    }

    fn window_event(&mut self, el: &winit::event_loop::ActiveEventLoop, _: winit::window::WindowId, ev: WindowEvent) {
        match ev {
            WindowEvent::Focused(f) => {
                self.focused = f;
                if !f { if let Some(w) = &self.win { w.set_cursor_grab(CursorGrabMode::None).ok(); w.set_cursor_visible(true); } }
            }
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::RedrawRequested => self.render(),
            WindowEvent::Resized(s) => {
                if let Some(r) = self.renderer.as_mut() { r.resize(s.width, s.height); }
            }
            WindowEvent::MouseInput { button: MouseButton::Left, state, .. } => {
                if state.is_pressed() {
                    if self.focused && self.state.is_playing() {
                        if let Some(w) = &self.win {
                            w.set_cursor_grab(CursorGrabMode::Confined).or_else(|_| w.set_cursor_grab(CursorGrabMode::Locked)).ok();
                            w.set_cursor_visible(false);
                        }
                    }
                    if !self.cat.scratching { self.input.scratch_pressed = true; }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let p = event.state.is_pressed();
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::KeyW) => self.input.forward = p,
                    PhysicalKey::Code(KeyCode::KeyS) => self.input.back = p,
                    PhysicalKey::Code(KeyCode::KeyA) => self.input.left = p,
                    PhysicalKey::Code(KeyCode::KeyD) => self.input.right = p,
                    PhysicalKey::Code(KeyCode::Space) => self.input.jump = p,
                    PhysicalKey::Code(KeyCode::ShiftLeft) | PhysicalKey::Code(KeyCode::ShiftRight) => self.input.sprint = p,
                    PhysicalKey::Code(KeyCode::KeyE) => { if p && !self.cat.scratching { self.input.scratch_pressed = true; } }
                    PhysicalKey::Code(KeyCode::Escape) if p => {
                        match &self.state {
                            GameState::Playing => {
                                self.state = GameState::Paused;
                                if let Some(w) = &self.win { w.set_cursor_grab(CursorGrabMode::None).ok(); w.set_cursor_visible(true); }
                            }
                            GameState::Paused => {
                                self.state = GameState::Playing;
                                if let Some(w) = &self.win { w.set_cursor_grab(CursorGrabMode::Confined).or_else(|_| w.set_cursor_grab(CursorGrabMode::Locked)).ok(); w.set_cursor_visible(false); }
                            }
                            GameState::Over(_) => {
                                self.state = GameState::Bill;
                                println!("\n{}", self.bill.format_bill());
                                println!("  Rating: {}\n", self.bill.rating());
                            }
                            GameState::Bill => {
                                if let Some(w) = &self.win { w.set_cursor_grab(CursorGrabMode::None).ok(); w.set_cursor_visible(true); }
                                el.exit();
                            }
                        }
                    }
                    PhysicalKey::Code(KeyCode::KeyQ) if p && self.state == GameState::Paused => {
                        if let Some(w) = &self.win { w.set_cursor_grab(CursorGrabMode::None).ok(); w.set_cursor_visible(true); }
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
    println!("  \u{1F431} PURRGE — Build 2 \"Whiskers\"");
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
