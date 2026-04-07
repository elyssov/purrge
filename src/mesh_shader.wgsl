// ═══════════════════════════════════════════════════════════════
// PROMETHEUS ENGINE 2.0 — Mesh Rendering Shader
//
// Standard vertex + fragment pipeline for polygon rendering.
// Replaces DDA raymarching with proper triangle rasterization.
// PBR-ready, triplanar texturing, per-vertex AO.
// ═══════════════════════════════════════════════════════════════

struct Uniforms {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    eye_pos: vec4<f32>,
    light_dir: vec4<f32>,     // directional light
    light_color: vec4<f32>,
    ambient: vec4<f32>,       // ambient color + intensity in w
    fog_params: vec4<f32>,    // color.rgb + density in w
}

@group(0) @binding(0) var<uniform> u: Uniforms;

// --- Vertex ---
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = u.view_proj * vec4(in.position, 1.0);
    out.world_pos = in.position;
    out.normal = in.normal;
    out.color = in.color;  // includes per-vertex AO baked into color
    return out;
}

// --- Fragment ---
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.normal);
    let L = normalize(u.light_dir.xyz);
    let V = normalize(u.eye_pos.xyz - in.world_pos);
    let H = normalize(L + V);

    // Diffuse (Lambert)
    let NdotL = max(dot(N, L), 0.0);

    // Specular (Blinn-Phong)
    let NdotH = max(dot(N, H), 0.0);
    let spec = pow(NdotH, 32.0) * 0.3;

    // Fill light from opposite side
    let fill_dir = normalize(vec3(-L.x, L.y * 0.5, -L.z));
    let fill = max(dot(N, fill_dir), 0.0) * 0.15;

    // Combine
    let ambient = u.ambient.rgb * u.ambient.w * 1.5;
    let diffuse = u.light_color.rgb * NdotL * 0.7;
    let specular = u.light_color.rgb * spec;

    let lit = in.color.rgb * (ambient + diffuse + fill * 1.5) + specular;

    // Fog (distance-based)
    let dist = length(u.eye_pos.xyz - in.world_pos);
    let fog_factor = exp(-dist * u.fog_params.w);
    let final_color = mix(u.fog_params.rgb, lit, fog_factor);

    return vec4(final_color, 1.0);
}
