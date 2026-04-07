// ═══════════════════════════════════════════════════════════════
// PROMETHEUS ENGINE 2.0 — Mesh Rendering Pipeline
//
// Standard wgpu triangle rasterization pipeline.
// Takes ChunkMesh from meshing.rs, uploads to GPU, renders.
// Replaces the DDA raymarcher for Phase 2.
// ═══════════════════════════════════════════════════════════════

use glam::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;
use super::meshing::{ChunkMesh, MeshVertex};

/// Uniform buffer layout (must match mesh_shader.wgsl)
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub eye_pos: [f32; 4],
    pub light_dir: [f32; 4],
    pub light_color: [f32; 4],
    pub ambient: [f32; 4],
    pub fog_params: [f32; 4],
}

impl MeshUniforms {
    pub fn new(
        view: Mat4,
        proj: Mat4,
        eye: Vec3,
        light_dir: Vec3,
    ) -> Self {
        Self {
            view_proj: (proj * view).to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            eye_pos: [eye.x, eye.y, eye.z, 1.0],
            light_dir: [light_dir.x, light_dir.y, light_dir.z, 0.0],
            light_color: [1.0, 0.98, 0.95, 1.0],  // warm white
            ambient: [0.15, 0.17, 0.22, 0.4],       // blue-ish ambient, intensity 0.4
            fog_params: [0.12, 0.13, 0.18, 0.003],  // fog color + density
        }
    }
}

/// Vertex layout for wgpu
impl MeshVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position: vec3<f32>
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                // normal: vec3<f32>
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,
                },
                // color: vec4<f32>
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 24,
                    shader_location: 2,
                },
            ],
        }
    }
}

/// GPU-side mesh ready for rendering
pub struct GpuMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

impl GpuMesh {
    /// Upload a ChunkMesh to GPU
    pub fn from_chunk_mesh(device: &wgpu::Device, mesh: &ChunkMesh) -> Option<Self> {
        if mesh.is_empty() { return None; }

        // Convert MeshVertex to raw bytes (position + normal + color, skip material for now)
        let vertex_data: Vec<f32> = mesh.vertices.iter().flat_map(|v| {
            vec![
                v.position[0], v.position[1], v.position[2],
                v.normal[0], v.normal[1], v.normal[2],
                v.color[0], v.color[1], v.color[2], v.color[3],
            ]
        }).collect();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Vertices"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Some(Self {
            vertex_buffer,
            index_buffer,
            index_count: mesh.indices.len() as u32,
        })
    }
}

/// Create the mesh rendering pipeline
pub fn create_mesh_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Mesh Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../mesh_shader.wgsl").into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Mesh BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Mesh Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Vertex buffer layout: 3 floats pos + 3 floats normal + 4 floats color = 10 floats = 40 bytes
    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: 40, // 10 × f32
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 24, shader_location: 2 },
        ],
    };

    let depth_format = wgpu::TextureFormat::Depth32Float;

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Mesh Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[vertex_layout],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back), // backface culling = 50% less triangles to shade
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    (pipeline, bind_group_layout)
}

/// Create depth texture for z-buffering
pub fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniforms_size() {
        // Ensure uniforms struct is correct size for GPU
        let size = std::mem::size_of::<MeshUniforms>();
        println!("MeshUniforms size: {} bytes", size);
        // Must be multiple of 16 for GPU alignment
        assert_eq!(size % 16, 0);
    }

    #[test]
    fn test_uniform_creation() {
        let u = MeshUniforms::new(
            Mat4::IDENTITY,
            Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 16.0/9.0, 0.1, 1000.0),
            Vec3::new(0.0, 10.0, -30.0),
            Vec3::new(0.3, -0.8, 0.5).normalize(),
        );
        // Eye position should be stored
        assert_eq!(u.eye_pos[1], 10.0);
    }
}
