// ═══════════════════════════════════════════════════════════════
// PURRGE — Dog AI
//
// Small kawaii dog. Sleeps 70% of the time.
// Wakes on loud noises (3-second grace period).
// Patrols 30-60 sec, blocks cat from objects.
// Gets stronger each roguelike cycle.
// ═══════════════════════════════════════════════════════════════

/// Dog behavioral state
#[derive(Debug, Clone, PartialEq)]
pub enum DogState {
    /// Zzzzz. Dream bubbles. Peaceful.
    Sleeping {
        /// How long until natural wake-up (seconds)
        sleep_remaining: f32,
    },
    /// Heard a noise! Grace period before full wake.
    Stirring {
        /// Seconds until fully awake (starts at 3.0)
        grace_timer: f32,
    },
    /// Awake and patrolling. Blocks cat access to furniture.
    Patrolling {
        /// Seconds until goes back to sleep
        patrol_remaining: f32,
        /// Current patrol target (world position index or waypoint)
        target_x: f32,
        target_z: f32,
    },
    /// Spotted the cat! Chasing / blocking.
    Blocking {
        /// How long dog will block this spot
        block_timer: f32,
    },
}

/// The dog entity — game logic only, no rendering
pub struct Dog {
    pub state: DogState,
    pub x: f32,
    pub z: f32,

    // ── Tuning ──
    /// Base sleep duration range (seconds)
    pub sleep_min: f32,
    pub sleep_max: f32,
    /// Patrol duration range
    pub patrol_min: f32,
    pub patrol_max: f32,
    /// Grace period when hearing noise (seconds)
    pub grace_duration: f32,
    /// How close cat must be for dog to notice
    pub detection_radius: f32,
    /// Dog movement speed (voxels/sec)
    pub move_speed: f32,
    /// Roguelike cycle — dog gets stronger each round
    pub cycle: u32,

    /// Pseudo-random state for deterministic behavior
    rng_state: u32,
}

impl Dog {
    pub fn new(x: f32, z: f32) -> Self {
        Self {
            state: DogState::Sleeping { sleep_remaining: 20.0 },
            x, z,
            sleep_min: 15.0,
            sleep_max: 45.0,
            patrol_min: 20.0,
            patrol_max: 45.0,
            grace_duration: 3.0,
            detection_radius: 25.0,
            move_speed: 15.0,
            cycle: 0,
            rng_state: 42,
        }
    }

    /// Advance one cycle — dog gets tougher
    pub fn next_cycle(&mut self) {
        self.cycle += 1;
        // Sleeps less, patrols longer, detects farther
        self.sleep_max = (self.sleep_max - 3.0).max(10.0);
        self.patrol_max += 5.0;
        self.detection_radius += 3.0;
        self.move_speed += 2.0;
    }

    /// Simple deterministic "random" for dog behavior
    fn rand(&mut self) -> f32 {
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.rng_state >> 16) & 0x7FFF) as f32 / 32767.0
    }

    fn rand_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.rand() * (max - min)
    }

    /// Main update. Returns events that happened this frame.
    /// cat_x, cat_z: cat position (for detection)
    /// noise: current frame's noise level (0.0-1.0, from destruction)
    pub fn update(&mut self, dt: f32, cat_x: f32, cat_z: f32, noise: f32) -> Vec<DogEvent> {
        let mut events = Vec::new();

        match &mut self.state {
            DogState::Sleeping { sleep_remaining } => {
                *sleep_remaining -= dt;

                // Loud noise wakes the dog
                if noise > 0.3 {
                    self.state = DogState::Stirring {
                        grace_timer: self.grace_duration,
                    };
                    events.push(DogEvent::Stirred);
                }
                // Natural wake-up
                else if *sleep_remaining <= 0.0 {
                    let patrol_time = self.rand_range(self.patrol_min, self.patrol_max);
                    let (tx, tz) = self.random_patrol_target(cat_x, cat_z);
                    self.state = DogState::Patrolling {
                        patrol_remaining: patrol_time,
                        target_x: tx,
                        target_z: tz,
                    };
                    events.push(DogEvent::WokeUp);
                }
            }

            DogState::Stirring { grace_timer } => {
                *grace_timer -= dt;

                if *grace_timer <= 0.0 {
                    // Fully awake — start patrolling
                    let patrol_time = self.rand_range(self.patrol_min, self.patrol_max);
                    let (tx, tz) = self.random_patrol_target(cat_x, cat_z);
                    self.state = DogState::Patrolling {
                        patrol_remaining: patrol_time,
                        target_x: tx,
                        target_z: tz,
                    };
                    events.push(DogEvent::WokeUp);
                }
                // If silence during grace period — go back to sleep
                else if noise < 0.05 && *grace_timer < self.grace_duration - 1.0 {
                    let sleep_time = self.rand_range(self.sleep_min, self.sleep_max);
                    self.state = DogState::Sleeping {
                        sleep_remaining: sleep_time,
                    };
                    events.push(DogEvent::FellBackAsleep);
                }
            }

            DogState::Patrolling { patrol_remaining, target_x, target_z } => {
                *patrol_remaining -= dt;
                let pr = *patrol_remaining;
                let tx_cur = *target_x;
                let tz_cur = *target_z;

                // Move toward target
                let dx = tx_cur - self.x;
                let dz = tz_cur - self.z;
                let dist = (dx * dx + dz * dz).sqrt();
                if dist > 2.0 {
                    let speed = self.move_speed * dt;
                    self.x += (dx / dist) * speed;
                    self.z += (dz / dist) * speed;
                } else {
                    let (ntx, ntz) = self.random_patrol_target(cat_x, cat_z);
                    if let DogState::Patrolling { target_x, target_z, .. } = &mut self.state {
                        *target_x = ntx;
                        *target_z = ntz;
                    }
                }

                // Detect cat nearby
                let cat_dx = cat_x - self.x;
                let cat_dz = cat_z - self.z;
                let cat_dist = (cat_dx * cat_dx + cat_dz * cat_dz).sqrt();
                if cat_dist < self.detection_radius {
                    self.state = DogState::Blocking { block_timer: 8.0 };
                    events.push(DogEvent::SpottedCat);
                } else if pr <= 0.0 {
                    let sleep_time = self.rand_range(self.sleep_min, self.sleep_max);
                    self.state = DogState::Sleeping { sleep_remaining: sleep_time };
                    events.push(DogEvent::FellAsleep);
                }
            }

            DogState::Blocking { block_timer } => {
                *block_timer -= dt;

                // Move toward cat (slowly)
                let dx = cat_x - self.x;
                let dz = cat_z - self.z;
                let dist = (dx * dx + dz * dz).sqrt();
                if dist > 5.0 {
                    let speed = self.move_speed * 0.7 * dt;
                    self.x += (dx / dist) * speed;
                    self.z += (dz / dist) * speed;
                }

                // Cat ran away
                if dist > self.detection_radius * 1.5 || *block_timer <= 0.0 {
                    let patrol_time = self.rand_range(self.patrol_min * 0.5, self.patrol_max * 0.5);
                    let (tx, tz) = self.random_patrol_target(cat_x, cat_z);
                    self.state = DogState::Patrolling {
                        patrol_remaining: patrol_time,
                        target_x: tx, target_z: tz,
                    };
                    events.push(DogEvent::LostCat);
                }
            }
        }

        events
    }

    /// Pick a random patrol waypoint (biased toward cat's area)
    fn random_patrol_target(&mut self, cat_x: f32, cat_z: f32) -> (f32, f32) {
        // 40% chance to patrol near cat, 60% random
        if self.rand() < 0.4 {
            let offset_x = self.rand_range(-30.0, 30.0);
            let offset_z = self.rand_range(-30.0, 30.0);
            (cat_x + offset_x, cat_z + offset_z)
        } else {
            (self.rand_range(20.0, 236.0), self.rand_range(20.0, 236.0))
        }
    }

    /// Is the dog currently an obstacle for the cat?
    pub fn is_blocking(&self) -> bool {
        matches!(self.state, DogState::Blocking { .. })
    }

    /// Is the dog sleeping? (cat can be louder)
    pub fn is_sleeping(&self) -> bool {
        matches!(self.state, DogState::Sleeping { .. })
    }

    /// Distance from dog to a point
    pub fn distance_to(&self, x: f32, z: f32) -> f32 {
        let dx = self.x - x;
        let dz = self.z - z;
        (dx * dx + dz * dz).sqrt()
    }
}

/// Events emitted by dog AI (for sound, UI, particles)
#[derive(Debug, Clone, PartialEq)]
pub enum DogEvent {
    /// Dog heard a noise, starting to stir
    Stirred,
    /// Dog fully woke up
    WokeUp,
    /// Went back to sleep after stirring (noise stopped)
    FellBackAsleep,
    /// Spotted the cat!
    SpottedCat,
    /// Lost sight of cat, resuming patrol
    LostCat,
    /// Patrol ended, going to sleep naturally
    FellAsleep,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_starts_sleeping() {
        let dog = Dog::new(100.0, 100.0);
        assert!(dog.is_sleeping());
    }

    #[test]
    fn test_noise_wakes_dog() {
        let mut dog = Dog::new(100.0, 100.0);
        let events = dog.update(0.1, 50.0, 50.0, 0.5); // loud noise
        assert!(events.contains(&DogEvent::Stirred));
        assert!(!dog.is_sleeping());
    }

    #[test]
    fn test_grace_period() {
        let mut dog = Dog::new(100.0, 100.0);
        dog.update(0.1, 50.0, 50.0, 0.5); // stir
        assert!(matches!(dog.state, DogState::Stirring { .. }));
        // Keep some noise during grace so dog doesn't fall back asleep
        for _ in 0..35 {
            dog.update(0.1, 50.0, 50.0, 0.1);
        }
        // Grace (3s) expired — should be patrolling
        assert!(matches!(dog.state, DogState::Patrolling { .. }));
    }

    #[test]
    fn test_natural_wakeup() {
        let mut dog = Dog::new(100.0, 100.0);
        dog.state = DogState::Sleeping { sleep_remaining: 1.0 };
        dog.update(1.5, 50.0, 50.0, 0.0); // sleep expires
        assert!(matches!(dog.state, DogState::Patrolling { .. }));
    }

    #[test]
    fn test_spots_cat() {
        let mut dog = Dog::new(100.0, 100.0);
        dog.state = DogState::Patrolling {
            patrol_remaining: 30.0,
            target_x: 100.0, target_z: 100.0,
        };
        let events = dog.update(0.1, 105.0, 100.0, 0.0); // cat nearby!
        assert!(events.contains(&DogEvent::SpottedCat));
        assert!(dog.is_blocking());
    }

    #[test]
    fn test_cycle_makes_dog_stronger() {
        let mut dog = Dog::new(100.0, 100.0);
        let old_detection = dog.detection_radius;
        dog.next_cycle();
        assert!(dog.detection_radius > old_detection);
        assert_eq!(dog.cycle, 1);
    }

    #[test]
    fn test_falls_back_asleep_if_quiet() {
        let mut dog = Dog::new(100.0, 100.0);
        dog.update(0.1, 50.0, 50.0, 0.5); // stir
        // Wait 1.5s in silence (past the 1s threshold)
        for _ in 0..20 {
            dog.update(0.1, 200.0, 200.0, 0.0);
        }
        // Should fall back asleep
        assert!(dog.is_sleeping());
    }
}
