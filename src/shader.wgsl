// ═══════════════════════════════════════════════════════════════
// PROMETHEUS ENGINE — Voxel Raymarcher (Phase 0)
// GPU raymarching through a flat voxel grid.
// Each pixel casts a ray, steps through the grid, finds the
// first non-empty voxel, and shades it.
// ═══════════════════════════════════════════════════════════════

struct Uniforms {
    inv_view_proj: mat4x4<f32>,
    eye_pos: vec4<f32>,
    grid_info: vec4<f32>,  // x = grid_size, y = time
}

struct Voxel {
    material: u32,  // packed: material(8) | r(8) | g(8) | b(8)
    flags: u32,     // packed: flags(8) | pad(24)
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> voxels: array<Voxel>;
@group(0) @binding(2) var fur_tex: texture_2d<f32>;
@group(0) @binding(3) var fur_samp: sampler;

// --- Fullscreen quad vertex shader ---
struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VSOut {
    // Two triangles covering the screen
    var positions = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
        vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0),
    );
    var out: VSOut;
    out.pos = vec4(positions[idx], 0.0, 1.0);
    out.uv = positions[idx] * 0.5 + 0.5;
    return out;
}

// --- Voxel lookup ---
fn grid_size() -> f32 { return u.grid_info.x; }
fn time() -> f32 { return u.grid_info.y; }

fn get_voxel(x: i32, y: i32, z: i32) -> Voxel {
    let gs = i32(grid_size());
    if x < 0 || y < 0 || z < 0 || x >= gs || y >= gs || z >= gs {
        return Voxel(0u, 0u);
    }
    let idx = z * gs * gs + y * gs + x;
    return voxels[idx];
}

fn voxel_color(v: Voxel) -> vec3<f32> {
    let packed = v.material;
    let r = f32((packed >> 8u) & 0xFFu) / 255.0;
    let g = f32((packed >> 16u) & 0xFFu) / 255.0;
    let b = f32((packed >> 24u) & 0xFFu) / 255.0;
    return vec3(r, g, b);
}

fn voxel_mat(v: Voxel) -> u32 {
    return v.material & 0xFFu;
}

// --- DDA Raymarching through voxel grid ---
// Standard 3D DDA algorithm (Amanatides & Woo)
struct RayHit {
    hit: bool,
    pos: vec3<i32>,
    normal: vec3<f32>,
    t: f32,
    voxel: Voxel,
}

fn raymarch(origin: vec3<f32>, dir: vec3<f32>) -> RayHit {
    var result: RayHit;
    result.hit = false;

    let gs = grid_size();

    // Find entry point into grid [0, gs]³
    var t_min = 0.0;
    var t_max = 1000.0;

    // AABB intersection with grid bounds
    for (var i = 0; i < 3; i++) {
        let inv_d = 1.0 / dir[i];
        var t0 = (0.0 - origin[i]) * inv_d;
        var t1 = (gs - origin[i]) * inv_d;
        if inv_d < 0.0 { let tmp = t0; t0 = t1; t1 = tmp; }
        t_min = max(t_min, t0);
        t_max = min(t_max, t1);
    }

    if t_max < t_min || t_max < 0.0 {
        return result;
    }

    // Start position
    let start = origin + dir * (t_min + 0.001);
    var map_pos = vec3<i32>(floor(start));

    let delta_dist = abs(vec3(1.0) / dir);
    let step = vec3<i32>(sign(dir));

    var side_dist: vec3<f32>;
    for (var i = 0; i < 3; i++) {
        if dir[i] < 0.0 {
            side_dist[i] = (start[i] - f32(map_pos[i])) * delta_dist[i];
        } else {
            side_dist[i] = (f32(map_pos[i]) + 1.0 - start[i]) * delta_dist[i];
        }
    }

    // March
    var side = 0;
    let max_steps = i32(gs) * 3;
    for (var i = 0; i < max_steps; i++) {
        // Check bounds
        let gsi = i32(gs);
        if map_pos.x < 0 || map_pos.y < 0 || map_pos.z < 0 ||
           map_pos.x >= gsi || map_pos.y >= gsi || map_pos.z >= gsi {
            break;
        }

        // Check voxel
        let v = get_voxel(map_pos.x, map_pos.y, map_pos.z);
        if voxel_mat(v) != 0u {
            result.hit = true;
            result.pos = map_pos;
            result.voxel = v;

            // Normal from last step
            result.normal = vec3(0.0);
            if side == 0 { result.normal.x = -f32(step.x); }
            else if side == 1 { result.normal.y = -f32(step.y); }
            else { result.normal.z = -f32(step.z); }

            return result;
        }

        // Step to next voxel (DDA)
        if side_dist.x < side_dist.y {
            if side_dist.x < side_dist.z {
                side_dist.x += delta_dist.x;
                map_pos.x += step.x;
                side = 0;
            } else {
                side_dist.z += delta_dist.z;
                map_pos.z += step.z;
                side = 2;
            }
        } else {
            if side_dist.y < side_dist.z {
                side_dist.y += delta_dist.y;
                map_pos.y += step.y;
                side = 1;
            } else {
                side_dist.z += delta_dist.z;
                map_pos.z += step.z;
                side = 2;
            }
        }
    }

    return result;
}

// --- Fragment shader ---
@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    // Reconstruct ray from screen UV
    let ndc = vec4(in.uv * 2.0 - 1.0, 0.0, 1.0);
    let ndc_far = vec4(in.uv * 2.0 - 1.0, 1.0, 1.0);

    let world_near = u.inv_view_proj * ndc;
    let world_far = u.inv_view_proj * ndc_far;

    let ray_origin = world_near.xyz / world_near.w;
    let ray_end = world_far.xyz / world_far.w;
    let ray_dir = normalize(ray_end - ray_origin);

    // Raymarch
    let hit = raymarch(ray_origin, ray_dir);

    if !hit.hit {
        // Background — lighter to frame the dollhouse
        let t = in.uv.y;
        return vec4(mix(vec3(0.15, 0.16, 0.22), vec3(0.08, 0.09, 0.14), t), 1.0);
    }

    // Shading
    var base_color = voxel_color(hit.voxel);

    // Fur texture: full texture on EACH FACE of each voxel (voxel ≠ pixel!)
    let mat = voxel_mat(hit.voxel);
    if mat == 13u {
        // Compute exact hit point on voxel face
        let n = hit.normal;
        var t_hit: f32 = 0.0;
        if abs(n.x) > 0.5 {
            let face_x = f32(hit.pos.x) + select(0.0, 1.0, n.x > 0.0);
            t_hit = (face_x - ray_origin.x) / ray_dir.x;
        } else if abs(n.y) > 0.5 {
            let face_y = f32(hit.pos.y) + select(0.0, 1.0, n.y > 0.0);
            t_hit = (face_y - ray_origin.y) / ray_dir.y;
        } else {
            let face_z = f32(hit.pos.z) + select(0.0, 1.0, n.z > 0.0);
            t_hit = (face_z - ray_origin.z) / ray_dir.z;
        }
        let hp = ray_origin + ray_dir * t_hit;

        // UV = fractional position on face (0-1 per voxel = full texture per face)
        var uv: vec2<f32>;
        if abs(n.x) > 0.5 {
            uv = fract(vec2(hp.z, hp.y));
        } else if abs(n.y) > 0.5 {
            uv = fract(vec2(hp.x, hp.z));
        } else {
            uv = fract(vec2(hp.x, hp.y));
        }

        let fur = textureSample(fur_tex, fur_samp, uv).rgb;
        base_color = base_color * fur * 1.5; // multiply blend + brighten
    }

    // Interior lighting: high ambient + omnidirectional fill
    let light1 = normalize(vec3(0.0, -0.8, -1.0));  // from camera direction (front)
    let light2 = normalize(vec3(0.5, 0.6, 0.3));    // top-right fill
    let light3 = normalize(vec3(-0.4, 0.3, -0.5));  // left fill
    let ndl1 = max(dot(hit.normal, light1), 0.0);
    let ndl2 = max(dot(hit.normal, light2), 0.0);
    let ndl3 = max(dot(hit.normal, light3), 0.0);

    // High ambient for interior visibility
    let ambient = 0.55;
    let diffuse = ndl1 * 0.3 + ndl2 * 0.2 + ndl3 * 0.15;
    let lit = base_color * (ambient + diffuse);

    // Minimal fog (almost none — interior scene)
    let dist = length(vec3<f32>(hit.pos) - ray_origin);
    let fog = exp(-dist * 0.003);
    let fog_color = vec3(0.12, 0.13, 0.18);
    let final_color = mix(fog_color, lit, fog);

    return vec4(final_color, 1.0);
}
