// ═══════════════════════════════════════════════════════════════
// PURRGE — Rendering (Mesh Pipeline)
//
// Uses Prometheus Engine mesh pipeline: voxels → Surface Nets → GPU.
// Separate meshes for room (static) and entities (dynamic).
// PBR shader with depth buffer.
// ═══════════════════════════════════════════════════════════════

use glam::{Mat4, Vec3};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::core::svo::Voxel;
use crate::core::meshing::{generate_mesh, generate_mesh_smooth, generate_mesh_smooth_with_ao, ChunkMesh};
use crate::core::render_mesh::{create_mesh_pipeline, create_depth_texture, GpuMesh, MeshUniforms};

/// Everything GPU-related
pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    ubuf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    depth_view: wgpu::TextureView,

    // Meshes — room split into chunks for fast partial rebuild
    pub room_chunks: Vec<Option<GpuMesh>>,
    pub chunks_per_axis: usize,
    pub chunk_size: usize,
    pub cat_mesh: Option<GpuMesh>,
    pub dog_mesh: Option<GpuMesh>,
    pub debris_mesh: Option<GpuMesh>,
    pub furniture_meshes: Vec<Option<GpuMesh>>,

    // HUD overlay
    hud_pipeline: wgpu::RenderPipeline,
    pub hud_verts: Option<wgpu::Buffer>,
    pub hud_vert_count: u32,
}

impl Renderer {
    pub fn new(w: Arc<Window>) -> Self {
        let inst = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(), ..Default::default()
        });
        let surf = inst.create_surface(w.clone()).unwrap();
        let adap = pollster::block_on(inst.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surf),
            force_fallback_adapter: false,
        })).expect("No GPU found");
        println!("  GPU: {}", adap.get_info().name);

        let (dev, q) = pollster::block_on(adap.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("PURRGE"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            }, None,
        )).unwrap();

        let sz = w.inner_size();
        let cfg = surf.get_default_config(&adap, sz.width.max(1), sz.height.max(1)).unwrap();
        surf.configure(&dev, &cfg);

        // Mesh pipeline (PBR shader, depth buffer, backface culling)
        let (pipeline, bgl) = create_mesh_pipeline(&dev, cfg.format);

        // Uniform buffer
        let ubuf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniforms"),
            contents: bytemuck::bytes_of(&MeshUniforms::new(
                Mat4::IDENTITY,
                Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 16.0/9.0, 1.0, 3000.0),
                Vec3::ZERO,
                Vec3::new(0.3, -0.8, 0.5).normalize(),
            )),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() }],
        });

        let (_, depth_view) = create_depth_texture(&dev, cfg.width, cfg.height);

        // HUD overlay pipeline (2D, no depth)
        let hud_shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("HUD"), source: wgpu::ShaderSource::Wgsl(include_str!("hud.wgsl").into()),
        });
        let hud_pipeline = dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("HUD Pipeline"),
            layout: Some(&dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None, bind_group_layouts: &[], push_constant_ranges: &[],
            })),
            vertex: wgpu::VertexState {
                module: &hud_shader, entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 24, step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 8, shader_location: 1 },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &hud_shader, entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: cfg.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None, multisample: Default::default(), multiview: None, cache: None,
        });

        Self {
            device: dev, queue: q, surface: surf, config: cfg,
            pipeline, bgl, ubuf, bind_group, depth_view,
            room_chunks: Vec::new(), chunks_per_axis: 0, chunk_size: 128,
            cat_mesh: None, dog_mesh: None, debris_mesh: None,
            furniture_meshes: Vec::new(),
            hud_pipeline, hud_verts: None, hud_vert_count: 0,
        }
    }

    /// Resize (recreate depth texture)
    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);
        let (_, dv) = create_depth_texture(&self.device, self.config.width, self.config.height);
        self.depth_view = dv;
    }

    /// Build all room chunks from sparse grid
    /// One-pass bucketing: O(voxels) instead of O(voxels × chunks)
    pub fn upload_room(&mut self, room: &crate::apartment::VoxelGrid, grid_size: usize) {
        let cs = self.chunk_size;
        let cpa = (grid_size + cs - 1) / cs;
        self.chunks_per_axis = cpa;
        let total_chunks = cpa * cpa * cpa;
        self.room_chunks = (0..total_chunks).map(|_| None).collect();
        let padded = cs + 2;

        let t0 = std::time::Instant::now();

        // ── PASS 1: Bucket voxels by chunk (single pass over HashMap) ──
        // Each voxel goes into its own chunk + adjacent chunks (for border padding)
        let mut buckets: Vec<Vec<(u16, u16, u16, Voxel)>> = (0..total_chunks).map(|_| Vec::new()).collect();

        for (&(vx, vy, vz), &v) in &room.data {
            // This voxel's home chunk
            let home_cx = vx as usize / cs;
            let home_cy = vy as usize / cs;
            let home_cz = vz as usize / cs;

            // Place into home chunk + neighbor chunks that need it as border
            // A chunk at (cx,cy,cz) covers world [cx*cs .. (cx+1)*cs-1]
            // Its padded region needs [cx*cs-1 .. (cx+1)*cs] (one extra on each side)
            // So this voxel (vx,vy,vz) is needed by chunk (cx,cy,cz) if:
            //   cx*cs - 1 <= vx <= (cx+1)*cs  →  cx = (vx+1)/cs .. vx/cs (if on boundary)
            for dcz in -1i32..=1 {
                for dcy in -1i32..=1 {
                    for dcx in -1i32..=1 {
                        let cx = home_cx as i32 + dcx;
                        let cy = home_cy as i32 + dcy;
                        let cz = home_cz as i32 + dcz;
                        if cx < 0 || cy < 0 || cz < 0 { continue; }
                        let (cx, cy, cz) = (cx as usize, cy as usize, cz as usize);
                        if cx >= cpa || cy >= cpa || cz >= cpa { continue; }

                        let x0 = cx * cs;
                        let y0 = cy * cs;
                        let z0 = cz * cs;
                        let lx = vx as i32 - x0 as i32 + 1;
                        let ly = vy as i32 - y0 as i32 + 1;
                        let lz = vz as i32 - z0 as i32 + 1;
                        if lx >= 0 && ly >= 0 && lz >= 0
                            && (lx as usize) < padded && (ly as usize) < padded && (lz as usize) < padded {
                            let idx = cz * cpa * cpa + cy * cpa + cx;
                            buckets[idx].push((lx as u16, ly as u16, lz as u16, v));
                        }
                    }
                }
            }
        }

        // ── PASS 2: Mesh each non-empty bucket ──
        let mut total_tris = 0;
        let mut built_chunks = 0;
        for idx in 0..total_chunks {
            if buckets[idx].is_empty() { continue; }
            let cz = idx / (cpa * cpa);
            let cy = (idx / cpa) % cpa;
            let cx = idx % cpa;
            let x0 = cx * cs;
            let y0 = cy * cs;
            let z0 = cz * cs;

            let mut flat = vec![Voxel::empty(); padded * padded * padded];
            for &(lx, ly, lz, v) in &buckets[idx] {
                flat[lz as usize * padded * padded + ly as usize * padded + lx as usize] = v;
            }

            let offset = Vec3::new(x0 as f32 - 1.0, y0 as f32 - 1.0, z0 as f32 - 1.0);
            let mesh = generate_mesh(&flat, padded, offset, 1.0);
            total_tris += mesh.triangle_count;
            self.room_chunks[idx] = GpuMesh::from_chunk_mesh(&self.device, &mesh);
            built_chunks += 1;
        }

        let elapsed = t0.elapsed();
        println!("  Room: {}/{} chunks ({}³), {} tris, {} voxels — built in {:.1}s",
            built_chunks, total_chunks, cs, total_tris, room.voxel_count(), elapsed.as_secs_f32());
    }

    /// Rebuild a single chunk (after scratch damage)
    pub fn rebuild_chunk_at(&mut self, room: &crate::apartment::VoxelGrid, grid_size: usize, world_x: f32, world_y: f32, world_z: f32) {
        let cs = self.chunk_size;
        let cpa = self.chunks_per_axis;
        if cpa == 0 { return; }
        let cx = (world_x as usize / cs).min(cpa - 1);
        let cy = (world_y as usize / cs).min(cpa - 1);
        let cz = (world_z as usize / cs).min(cpa - 1);
        // Export single chunk via per-voxel iteration (ok for rebuild — small area)
        let padded = cs + 2;
        let mut flat = vec![Voxel::empty(); padded * padded * padded];
        let x0 = cx * cs; let y0 = cy * cs; let z0 = cz * cs;
        for (&(vx, vy, vz), &v) in &room.data {
            let (gx, gy, gz) = (vx as usize, vy as usize, vz as usize);
            if gx + 1 >= x0 && gx < x0 + padded && gy + 1 >= y0 && gy < y0 + padded && gz + 1 >= z0 && gz < z0 + padded {
                let lx = gx + 1 - x0; let ly = gy + 1 - y0; let lz = gz + 1 - z0;
                if lx < padded && ly < padded && lz < padded {
                    flat[lz * padded * padded + ly * padded + lx] = v;
                }
            }
        }
        let offset = Vec3::new(x0 as f32 - 1.0, y0 as f32 - 1.0, z0 as f32 - 1.0);
        let mesh = generate_mesh(&flat, padded, offset, 1.0);
        let idx = cz * cpa * cpa + cy * cpa + cx;
        self.room_chunks[idx] = GpuMesh::from_chunk_mesh(&self.device, &mesh);
    }

    /// Upload entity mesh (smooth — organic shapes like cat/dog)
    pub fn upload_entity_smooth(&mut self, voxels: &[Voxel], grid_size: usize, target: &str) {
        let mesh = generate_mesh_smooth_with_ao(voxels, grid_size, Vec3::ZERO, 1.0);
        match target {
            "cat" => self.cat_mesh = GpuMesh::from_chunk_mesh(&self.device, &mesh),
            "dog" => self.dog_mesh = GpuMesh::from_chunk_mesh(&self.device, &mesh),
            _ => {}
        }
    }

    /// Upload entity mesh (sharp — for blocky entities or debris)
    pub fn upload_entity_sharp(&mut self, voxels: &[Voxel], grid_size: usize, target: &str) {
        let mesh = generate_mesh(voxels, grid_size, Vec3::ZERO, 1.0);
        match target {
            "cat" => self.cat_mesh = GpuMesh::from_chunk_mesh(&self.device, &mesh),
            "dog" => self.dog_mesh = GpuMesh::from_chunk_mesh(&self.device, &mesh),
            "debris" => self.debris_mesh = GpuMesh::from_chunk_mesh(&self.device, &mesh),
            _ => {}
        }
    }

    /// Render frame
    pub fn render(&mut self, eye: Vec3, target: Vec3, screen_shake: f32, time: f32) {
        let aspect = self.config.width as f32 / self.config.height as f32;
        let view = Mat4::look_at_rh(eye, target, Vec3::Y);
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 1.0, 3000.0);
        let light_dir = Vec3::new(0.3, -0.8, 0.5).normalize();

        let uniforms = MeshUniforms::new(view, proj, eye, light_dir);
        self.queue.write_buffer(&self.ubuf, 0, bytemuck::bytes_of(&uniforms));

        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => { self.surface.configure(&self.device, &self.config); return; }
        };
        let color_view = frame.texture.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&Default::default());

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.45, g: 0.55, b: 0.7, a: 1.0 }), // bright sky blue
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);

            // Draw room chunks
            for chunk in &self.room_chunks {
                if let Some(mesh) = chunk {
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }
            }

            // Draw cat
            if let Some(mesh) = &self.cat_mesh {
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }

            // Draw dog
            if let Some(mesh) = &self.dog_mesh {
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }

            // Draw furniture
            for fmesh in &self.furniture_meshes {
                if let Some(mesh) = fmesh {
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }
            }

            // Draw debris
            if let Some(mesh) = &self.debris_mesh {
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }

        // HUD overlay pass (no depth, on top of everything)
        if let Some(hud_vb) = &self.hud_verts {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("HUD"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.hud_pipeline);
            pass.set_vertex_buffer(0, hud_vb.slice(..));
            pass.draw(0..self.hud_vert_count, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
    }

    /// Upload HUD geometry (screen-space quads in NDC)
    pub fn upload_hud(&mut self, verts: &[f32]) {
        if verts.is_empty() {
            self.hud_verts = None;
            self.hud_vert_count = 0;
            return;
        }
        self.hud_verts = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("HUD"),
            contents: bytemuck::cast_slice(verts),
            usage: wgpu::BufferUsages::VERTEX,
        }));
        self.hud_vert_count = (verts.len() / 6) as u32; // 6 floats per vertex (xy + rgba)
    }
}
