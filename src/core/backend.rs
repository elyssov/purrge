// ═══════════════════════════════════════════════════════════════
// PROMETHEUS ENGINE — Adaptive Rendering Backend
//
// Like Quake in '96: one game, multiple renderers.
// Detects hardware, picks the best path:
//
//   Tier 0: Software (CPU only) — old laptops, phones
//   Tier 1: Basic GPU (Iris Xe, Mali) — integrated graphics
//   Tier 2: Discrete GPU (GTX/RX) — gaming laptops
//   Tier 3: RTX/RDNA3 — tensor cores, hardware RT
//
// Same formulas, same world, same game. Different visual quality.
// ═══════════════════════════════════════════════════════════════

/// Hardware capability tier
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HardwareTier {
    /// CPU-only software rendering. Lowest quality, runs everywhere.
    /// Targets: old smartphones, Raspberry Pi, office PCs without GPU
    /// Resolution: 320×240 or 640×480, low voxel density
    Software = 0,

    /// Basic integrated GPU (Intel Iris, AMD APU, Mali, Adreno)
    /// Targets: office laptops, budget phones
    /// Resolution: 720p-1080p, medium voxel density
    IntegratedGPU = 1,

    /// Discrete GPU without RT cores (GTX 1060+, RX 580+)
    /// Targets: gaming PCs, gaming laptops
    /// Resolution: 1080p-1440p, high voxel density, full effects
    DiscreteGPU = 2,

    /// RTX/RDNA3 with tensor cores and hardware raytracing
    /// Targets: RTX 2060+, RX 7600+
    /// Resolution: 1440p-4K with DLSS/FSR, max quality
    RTXClass = 3,
}

/// Rendering settings adapted to hardware
#[derive(Debug, Clone)]
pub struct RenderConfig {
    pub tier: HardwareTier,

    // Resolution
    pub render_width: u32,
    pub render_height: u32,
    pub display_width: u32,
    pub display_height: u32,
    pub upscale: bool,          // render low, upscale to display

    // Voxel world
    pub view_distance: i32,     // chunks
    pub chunk_size: usize,      // voxels per chunk edge
    pub max_chunks: usize,      // in memory
    pub voxel_budget: usize,    // max voxels to generate per frame

    // Visual quality
    pub ao_enabled: bool,
    pub ao_samples: u32,
    pub shadow_enabled: bool,
    pub reflection_enabled: bool,
    pub fog_enabled: bool,
    pub bloom_enabled: bool,
    pub fov: f32,               // field of view degrees

    // Generation
    pub hollow_models: bool,
    pub lod_levels: u32,        // 1 = no LOD, 4 = full LOD chain
    pub prefetch_frames: u32,   // how many frames ahead to predict

    // AI acceleration
    pub tensor_denoise: bool,   // use tensor cores for denoising
    pub tensor_upscale: bool,   // use tensor cores for upscaling
}

impl RenderConfig {
    /// Auto-detect hardware and create optimal config
    pub fn auto_detect() -> Self {
        // TODO: actually detect via wgpu adapter info
        // For now, default to Tier 2 (discrete GPU)
        Self::for_tier(HardwareTier::DiscreteGPU, 1920, 1080)
    }

    /// Create config for specific tier and display resolution
    pub fn for_tier(tier: HardwareTier, display_w: u32, display_h: u32) -> Self {
        match tier {
            HardwareTier::Software => Self {
                tier,
                render_width: display_w / 4,    // render at quarter res
                render_height: display_h / 4,
                display_width: display_w,
                display_height: display_h,
                upscale: true,
                view_distance: 1,               // 3×3×3 = 27 chunks
                chunk_size: 32,                  // smaller chunks
                max_chunks: 27,
                voxel_budget: 50_000,
                ao_enabled: false,
                ao_samples: 0,
                shadow_enabled: false,
                reflection_enabled: false,
                fog_enabled: true,              // fog hides pop-in
                bloom_enabled: false,
                fov: 120.0,
                hollow_models: true,
                lod_levels: 1,
                prefetch_frames: 5,
                tensor_denoise: false,
                tensor_upscale: false,
            },
            HardwareTier::IntegratedGPU => Self {
                tier,
                render_width: display_w / 2,
                render_height: display_h / 2,
                display_width: display_w,
                display_height: display_h,
                upscale: true,
                view_distance: 2,               // 5×5×5 = 125 chunks
                chunk_size: 64,
                max_chunks: 125,
                voxel_budget: 200_000,
                ao_enabled: true,
                ao_samples: 4,
                shadow_enabled: false,
                reflection_enabled: false,
                fog_enabled: true,
                bloom_enabled: false,
                fov: 120.0,
                hollow_models: true,
                lod_levels: 2,
                prefetch_frames: 10,
                tensor_denoise: false,
                tensor_upscale: false,
            },
            HardwareTier::DiscreteGPU => Self {
                tier,
                render_width: display_w,
                render_height: display_h,
                display_width: display_w,
                display_height: display_h,
                upscale: false,
                view_distance: 3,               // 7×7×7 = 343 chunks
                chunk_size: 64,
                max_chunks: 343,
                voxel_budget: 1_000_000,
                ao_enabled: true,
                ao_samples: 8,
                shadow_enabled: true,
                reflection_enabled: false,
                fog_enabled: true,
                bloom_enabled: true,
                fov: 120.0,
                hollow_models: true,
                lod_levels: 3,
                prefetch_frames: 15,
                tensor_denoise: false,
                tensor_upscale: false,
            },
            HardwareTier::RTXClass => Self {
                tier,
                render_width: display_w * 2 / 3, // render at 2/3, tensor upscale
                render_height: display_h * 2 / 3,
                display_width: display_w,
                display_height: display_h,
                upscale: true,
                view_distance: 4,               // 9×9×9 = 729 chunks
                chunk_size: 64,
                max_chunks: 729,
                voxel_budget: 5_000_000,
                ao_enabled: true,
                ao_samples: 16,
                shadow_enabled: true,
                reflection_enabled: true,
                fog_enabled: true,
                bloom_enabled: true,
                fov: 120.0,
                hollow_models: true,
                lod_levels: 4,
                prefetch_frames: 20,
                tensor_denoise: true,            // denoise sparse rays
                tensor_upscale: true,            // 720p → 1080p via tensor
            },
        }
    }

    /// Effective world size visible at once (in voxels)
    pub fn visible_world_size(&self) -> usize {
        (self.view_distance as usize * 2 + 1) * self.chunk_size
    }

    /// Estimated GPU memory usage (MB)
    pub fn estimated_vram_mb(&self) -> f32 {
        let total_voxels = self.visible_world_size().pow(3);
        let bytes = total_voxels * 8; // 8 bytes per voxel
        bytes as f32 / 1_048_576.0
    }

    /// Print config summary
    pub fn print_summary(&self) {
        println!("  ═══ Render Config ═══");
        println!("  Tier: {:?}", self.tier);
        println!("  Render: {}×{} → {}×{} {}",
            self.render_width, self.render_height,
            self.display_width, self.display_height,
            if self.upscale { "(upscaled)" } else { "" });
        println!("  FOV: {}°", self.fov);
        println!("  View: {} chunks ({} voxels³)",
            self.view_distance, self.visible_world_size());
        println!("  VRAM: ~{:.0} MB", self.estimated_vram_mb());
        println!("  AO: {} ({}×)", self.ao_enabled, self.ao_samples);
        println!("  Shadows: {}, Bloom: {}", self.shadow_enabled, self.bloom_enabled);
        println!("  Tensor denoise: {}, upscale: {}",
            self.tensor_denoise, self.tensor_upscale);
        println!("  LOD levels: {}", self.lod_levels);
        println!("  Prefetch: {} frames ahead", self.prefetch_frames);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_configs() {
        for tier in [HardwareTier::Software, HardwareTier::IntegratedGPU,
                     HardwareTier::DiscreteGPU, HardwareTier::RTXClass] {
            let cfg = RenderConfig::for_tier(tier, 1920, 1080);
            cfg.print_summary();
            println!();
            // VRAM should be reasonable
            assert!(cfg.estimated_vram_mb() < 4096.0);
        }
    }

    #[test]
    fn test_software_tier_minimal() {
        let cfg = RenderConfig::for_tier(HardwareTier::Software, 1280, 720);
        // Software should be very low memory
        assert!(cfg.estimated_vram_mb() < 50.0);
        assert!(!cfg.ao_enabled);
        assert!(!cfg.shadow_enabled);
        assert_eq!(cfg.chunk_size, 32);
    }

    #[test]
    fn test_fov_120() {
        let cfg = RenderConfig::auto_detect();
        assert_eq!(cfg.fov, 120.0); // always 120, better safe than sorry
    }
}
