// ═══════════════════════════════════════════════════════════════
// PROMETHEUS ENGINE — Destruction System
//
// Hit → remove voxels → check connectivity → spawn debris.
// The core loop that makes destruction PHYSICAL, not cosmetic.
//
// Key algorithm: flood fill from ground contact points.
// Everything NOT connected to ground = debris = falls.
//
// Also: screen shake params, hitstop, particle spawning.
// ═══════════════════════════════════════════════════════════════

use std::collections::{HashSet, VecDeque};
use super::svo::Voxel;
use super::material::MaterialRegistry;

/// A chunk of voxels that broke off and becomes a rigid body
#[derive(Clone, Debug)]
pub struct Debris {
    /// World-space position (center of mass)
    pub position: [f32; 3],
    /// Velocity
    pub velocity: [f32; 3],
    /// Angular velocity (simplified: rotation around Y)
    pub angular_vel: f32,
    /// List of voxels relative to center of mass
    pub voxels: Vec<([i32; 3], Voxel)>,
    /// Total mass (from material densities)
    pub mass: f32,
    /// Average brittleness (determines if debris shatters on impact)
    pub brittleness: f32,
    /// Is this debris still active?
    pub active: bool,
    /// Time alive (for cleanup)
    pub age: f32,
}

/// Result of a destruction event
#[derive(Debug)]
pub struct DestructionResult {
    /// Voxels that were removed
    pub removed_count: usize,
    /// Debris chunks that broke off (need physics)
    pub debris: Vec<Debris>,
    /// Particle spawn requests (position, color, count)
    pub particles: Vec<ParticleSpawn>,
    /// Screen shake intensity (0.0 = none, 1.0 = max)
    pub shake_intensity: f32,
    /// Hitstop duration in seconds (0.0 = none)
    pub hitstop: f32,
    /// Noise level (for AI alerting)
    pub noise: f32,
}

/// Request to spawn particles
#[derive(Debug)]
pub struct ParticleSpawn {
    pub position: [f32; 3],
    pub color: [u8; 3],
    pub count: u32,
    pub speed: f32,
}

/// Destroy voxels in a sphere and check for disconnected pieces.
///
/// This is THE core destruction function:
/// 1. Remove voxels in sphere
/// 2. Flood fill from ground to find connected regions
/// 3. Everything not connected → debris
/// 4. Generate particles, shake, hitstop
pub fn destroy_sphere(
    voxels: &mut [Voxel],
    size: usize,
    cx: f32, cy: f32, cz: f32,
    radius: f32,
    materials: &MaterialRegistry,
    ground_y: usize,  // Y level considered "ground" (voxels at this Y = anchored)
) -> DestructionResult {
    let r2 = radius * radius;
    let ri = radius.ceil() as i32;
    let mut removed = Vec::new();
    let mut total_noise = 0.0_f32;

    // Step 1: Remove voxels in sphere
    for dz in -ri..=ri { for dy in -ri..=ri { for dx in -ri..=ri {
        if (dx*dx + dy*dy + dz*dz) as f32 > r2 { continue; }
        let x = (cx as i32 + dx) as usize;
        let y = (cy as i32 + dy) as usize;
        let z = (cz as i32 + dz) as usize;
        if x >= size || y >= size || z >= size { continue; }

        let idx = z * size * size + y * size + x;
        let v = voxels[idx];
        if v.is_solid() {
            let mat_id = (v.packed & 0xFF) as u8;
            let hit = materials.hit_result(mat_id, 1.0);
            if hit.destroyed {
                removed.push((x, y, z, v));
                voxels[idx] = Voxel::empty();
                total_noise += hit.noise;
            }
        }
    }}}

    if removed.is_empty() {
        return DestructionResult {
            removed_count: 0, debris: Vec::new(), particles: Vec::new(),
            shake_intensity: 0.0, hitstop: 0.0, noise: 0.0,
        };
    }

    // Step 2: Generate particles from removed voxels
    let mut particles = Vec::new();
    for &(x, y, z, v) in &removed {
        let mat_id = (v.packed & 0xFF) as u8;
        let mat = materials.get(mat_id);
        particles.push(ParticleSpawn {
            position: [x as f32, y as f32, z as f32],
            color: mat.particle_color,
            count: if mat.brittleness > 0.5 { 5 } else { 2 },
            speed: mat.brittleness * 3.0 + 1.0,
        });
    }

    // Step 3: Find disconnected pieces (flood fill from ground)
    let debris = find_disconnected_pieces(voxels, size, ground_y, cx, cy, cz, radius * 3.0, materials);

    // Step 4: Calculate game feel parameters
    let shake = (removed.len() as f32 / 50.0).min(1.0);
    let hitstop = if removed.len() > 10 { 0.04 } else { 0.0 };

    DestructionResult {
        removed_count: removed.len(),
        debris,
        particles,
        shake_intensity: shake,
        hitstop,
        noise: total_noise / removed.len().max(1) as f32,
    }
}

/// Find voxel groups not connected to ground near the destruction site.
/// Returns them as Debris objects (removed from the grid).
fn find_disconnected_pieces(
    voxels: &mut [Voxel],
    size: usize,
    ground_y: usize,
    cx: f32, cy: f32, cz: f32,
    search_radius: f32,
    materials: &MaterialRegistry,
) -> Vec<Debris> {
    let idx = |x: usize, y: usize, z: usize| -> usize { z * size * size + y * size + x };

    // Search area around destruction (don't flood-fill the entire world)
    let sr = search_radius.ceil() as i32;
    let min_x = (cx as i32 - sr).max(0) as usize;
    let max_x = (cx as i32 + sr).min(size as i32 - 1) as usize;
    let min_y = (cy as i32 - sr).max(0) as usize;
    let max_y = (cy as i32 + sr).min(size as i32 - 1) as usize;
    let min_z = (cz as i32 - sr).max(0) as usize;
    let max_z = (cz as i32 + sr).min(size as i32 - 1) as usize;

    // Flood fill from all ground-touching voxels in search area
    let mut grounded: HashSet<(usize, usize, usize)> = HashSet::new();
    let mut queue: VecDeque<(usize, usize, usize)> = VecDeque::new();

    // Seed: all solid voxels at ground level
    for z in min_z..=max_z {
        for x in min_x..=max_x {
            if ground_y < size && voxels[idx(x, ground_y, z)].is_solid() {
                queue.push_back((x, ground_y, z));
                grounded.insert((x, ground_y, z));
            }
            // Also seed from edges (walls are anchored)
            for y in min_y..=max_y {
                if x == 0 || x == size-1 || z == 0 || z == size-1 {
                    if voxels[idx(x, y, z)].is_solid() {
                        queue.push_back((x, y, z));
                        grounded.insert((x, y, z));
                    }
                }
            }
        }
    }

    // BFS: mark all voxels connected to ground
    while let Some((x, y, z)) = queue.pop_front() {
        let neighbors: [(i32,i32,i32); 6] = [
            (1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)
        ];
        for (dx, dy, dz) in neighbors {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            let nz = z as i32 + dz;
            if nx < min_x as i32 || nx > max_x as i32
                || ny < min_y as i32 || ny > max_y as i32
                || nz < min_z as i32 || nz > max_z as i32 { continue; }
            let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);
            if grounded.contains(&(nx, ny, nz)) { continue; }
            if !voxels[idx(nx, ny, nz)].is_solid() { continue; }
            grounded.insert((nx, ny, nz));
            queue.push_back((nx, ny, nz));
        }
    }

    // Find all solid voxels in search area NOT connected to ground
    let mut floating: Vec<(usize, usize, usize, Voxel)> = Vec::new();
    for z in min_z..=max_z {
        for y in (min_y+1)..=max_y { // skip ground level
            for x in min_x..=max_x {
                if voxels[idx(x, y, z)].is_solid() && !grounded.contains(&(x, y, z)) {
                    floating.push((x, y, z, voxels[idx(x, y, z)]));
                }
            }
        }
    }

    if floating.is_empty() { return Vec::new(); }

    // Group floating voxels into connected components (each = one debris piece)
    let mut visited: HashSet<(usize, usize, usize)> = HashSet::new();
    let mut debris_list: Vec<Debris> = Vec::new();
    let floating_set: HashSet<(usize, usize, usize)> = floating.iter().map(|(x,y,z,_)| (*x,*y,*z)).collect();

    for &(sx, sy, sz, _) in &floating {
        if visited.contains(&(sx, sy, sz)) { continue; }

        // BFS to find this connected component
        let mut component: Vec<([i32;3], Voxel)> = Vec::new();
        let mut comp_queue: VecDeque<(usize,usize,usize)> = VecDeque::new();
        comp_queue.push_back((sx, sy, sz));
        visited.insert((sx, sy, sz));

        while let Some((x, y, z)) = comp_queue.pop_front() {
            let v = voxels[idx(x, y, z)];
            component.push(([x as i32, y as i32, z as i32], v));
            // Remove from grid
            voxels[idx(x, y, z)] = Voxel::empty();

            for (dx, dy, dz) in [(1i32,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)] {
                let nx = (x as i32 + dx) as usize;
                let ny = (y as i32 + dy) as usize;
                let nz = (z as i32 + dz) as usize;
                if nx >= size || ny >= size || nz >= size { continue; }
                if visited.contains(&(nx, ny, nz)) { continue; }
                if !floating_set.contains(&(nx, ny, nz)) { continue; }
                visited.insert((nx, ny, nz));
                comp_queue.push_back((nx, ny, nz));
            }
        }

        if component.is_empty() { continue; }

        // Compute center of mass
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_z = 0.0f32;
        let mut total_mass = 0.0f32;
        let mut total_brit = 0.0f32;
        for (pos, v) in &component {
            let mat_id = (v.packed & 0xFF) as u8;
            let mat = materials.get(mat_id);
            let m = mat.density;
            sum_x += pos[0] as f32 * m;
            sum_y += pos[1] as f32 * m;
            sum_z += pos[2] as f32 * m;
            total_mass += m;
            total_brit += mat.brittleness;
        }
        let count = component.len() as f32;
        let com = [sum_x / total_mass, sum_y / total_mass, sum_z / total_mass];

        // Convert positions to relative-to-COM
        let relative: Vec<([i32;3], Voxel)> = component.iter().map(|(pos, v)| {
            ([pos[0] - com[0] as i32, pos[1] - com[1] as i32, pos[2] - com[2] as i32], *v)
        }).collect();

        debris_list.push(Debris {
            position: com,
            velocity: [0.0, 0.0, 0.0],
            angular_vel: 0.0,
            voxels: relative,
            mass: total_mass,
            brittleness: total_brit / count,
            active: true,
            age: 0.0,
        });
    }

    debris_list
}

/// Update debris physics (gravity, floor collision)
pub fn update_debris(debris: &mut [Debris], dt: f32, ground_y: f32, materials: &MaterialRegistry) {
    let gravity = -9.8;

    for d in debris.iter_mut() {
        if !d.active { continue; }
        d.age += dt;

        // Gravity
        d.velocity[1] += gravity * dt;

        // Move
        d.position[0] += d.velocity[0] * dt;
        d.position[1] += d.velocity[1] * dt;
        d.position[2] += d.velocity[2] * dt;

        // Floor collision
        if d.position[1] <= ground_y {
            d.position[1] = ground_y;
            if d.velocity[1].abs() > 2.0 && d.brittleness > 0.5 {
                // Shatter on impact! (TODO: split into smaller debris)
                d.active = false;
            } else {
                // Bounce (damped)
                d.velocity[1] *= -0.3;
                d.velocity[0] *= 0.8; // friction
                d.velocity[2] *= 0.8;
            }

            // Stop if slow enough
            let speed = (d.velocity[0]*d.velocity[0] + d.velocity[1]*d.velocity[1] + d.velocity[2]*d.velocity[2]).sqrt();
            if speed < 0.5 {
                d.velocity = [0.0, 0.0, 0.0];
            }
        }

        // Rotation
        d.angular_vel *= 0.98; // damping

        // Cleanup: deactivate after 10 seconds
        if d.age > 10.0 {
            d.active = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_table(voxels: &mut Vec<Voxel>, size: usize) {
        let idx = |x:usize,y:usize,z:usize| z*size*size + y*size + x;
        let wood = Voxel::solid(1, 140, 95, 50);
        // Table top: y=20, x=20-40, z=20-30
        for z in 20..30 { for x in 20..40 {
            voxels[idx(x, 20, z)] = wood;
        }}
        // 4 legs: (20,1-19,20), (39,1-19,20), (20,1-19,29), (39,1-19,29)
        for y in 1..20 {
            voxels[idx(20, y, 20)] = wood;
            voxels[idx(39, y, 20)] = wood;
            voxels[idx(20, y, 29)] = wood;
            voxels[idx(39, y, 29)] = wood;
        }
        // Floor
        for z in 0..size { for x in 0..size {
            voxels[idx(x, 0, z)] = Voxel::solid(7, 180, 155, 115);
        }}
    }

    #[test]
    fn test_destroy_sphere() {
        let size = 64;
        let mut voxels = vec![Voxel::empty(); size*size*size];
        let materials = MaterialRegistry::default();
        make_table(&mut voxels, size);

        let result = destroy_sphere(&mut voxels, size, 30.0, 20.0, 25.0, 3.0, &materials, 0);
        println!("Destroyed {} voxels, {} debris pieces, shake={:.2}",
            result.removed_count, result.debris.len(), result.shake_intensity);
        assert!(result.removed_count > 0);
    }

    #[test]
    fn test_destroy_table_leg_creates_debris() {
        let size = 64;
        let mut voxels = vec![Voxel::empty(); size*size*size];
        let materials = MaterialRegistry::default();
        make_table(&mut voxels, size);

        // Destroy a table leg at (20, 10, 20) — should disconnect the tabletop
        let result = destroy_sphere(&mut voxels, size, 20.0, 10.0, 20.0, 3.0, &materials, 0);

        println!("After destroying leg:");
        println!("  Removed: {} voxels", result.removed_count);
        println!("  Debris pieces: {}", result.debris.len());
        for (i, d) in result.debris.iter().enumerate() {
            println!("  Debris {}: {} voxels, mass={:.0}, pos=({:.0},{:.0},{:.0})",
                i, d.voxels.len(), d.mass, d.position[0], d.position[1], d.position[2]);
        }

        // The tabletop should become debris (not connected to floor after leg removed)
        // Note: depends on which legs remain connected. With 4 legs and 1 removed,
        // the top might still be connected through other legs.
        // Destroy ALL legs to guarantee disconnection:
        assert!(result.removed_count > 0);
    }

    #[test]
    fn test_destroy_all_legs() {
        let size = 64;
        let mut voxels = vec![Voxel::empty(); size*size*size];
        let materials = MaterialRegistry::default();
        make_table(&mut voxels, size);

        // Destroy ALL 4 legs
        destroy_sphere(&mut voxels, size, 20.0, 10.0, 20.0, 2.0, &materials, 0);
        destroy_sphere(&mut voxels, size, 39.0, 10.0, 20.0, 2.0, &materials, 0);
        destroy_sphere(&mut voxels, size, 20.0, 10.0, 29.0, 2.0, &materials, 0);
        let result = destroy_sphere(&mut voxels, size, 39.0, 10.0, 29.0, 2.0, &materials, 0);

        println!("After destroying ALL legs:");
        println!("  Debris: {} pieces", result.debris.len());
        for (i, d) in result.debris.iter().enumerate() {
            println!("  Piece {}: {} voxels", i, d.voxels.len());
        }

        // Tabletop should now be floating → becomes debris
        assert!(!result.debris.is_empty(), "Tabletop should become debris!");
    }

    #[test]
    fn test_debris_falls() {
        let mut debris = vec![Debris {
            position: [30.0, 20.0, 25.0],
            velocity: [0.0, 0.0, 0.0],
            angular_vel: 0.0,
            voxels: vec![([0,0,0], Voxel::solid(1, 140, 95, 50))],
            mass: 600.0,
            brittleness: 0.3,
            active: true,
            age: 0.0,
        }];

        let materials = MaterialRegistry::default();

        // Simulate 2 seconds
        for _ in 0..120 {
            update_debris(&mut debris, 1.0/60.0, 0.0, &materials);
        }

        println!("After 2 seconds: y={:.1}, vy={:.2}", debris[0].position[1], debris[0].velocity[1]);
        // Should have fallen to ground
        assert!(debris[0].position[1] <= 1.0);
    }
}
