// ═══════════════════════════════════════════════════════════════
// PURRGE — Game Meters
//
// Boredom:    100→0 over time. Destruction restores it. 0 = death.
// Annoyance:  0→100 from noise. 100 = thrown out.
// Lives:      9→0. Fall damage, electricity, heavy objects.
// ═══════════════════════════════════════════════════════════════

/// All three game meters in one place
pub struct Meters {
    /// 0.0-100.0, starts at 100. Decreases over time. 0 = game over (boredom death).
    pub boredom: f32,
    /// 0.0-100.0, starts at 0. Increases from destruction noise. 100 = thrown out.
    pub annoyance: f32,
    /// 9 down to 0. 0 = game over (all lives lost).
    pub lives: u8,

    // ── Tuning parameters ──
    /// Boredom decay per second (default: 1.5 = ~67 seconds to die of boredom)
    pub boredom_decay_rate: f32,
    /// Annoyance decay per second when NOT destroying (neighbors calm down)
    pub annoyance_decay_rate: f32,
    /// Annoyance cap — can't exceed this
    pub annoyance_max: f32,
}

/// What ended the game
#[derive(Debug, Clone, PartialEq)]
pub enum GameOver {
    /// Boredom reached 0 — cat died of ennui
    BoredToDeath,
    /// Annoyance reached 100 — owner came home, cat thrown out
    ThrownOut,
    /// All 9 lives used up
    NoLivesLeft,
}

impl Meters {
    pub fn new() -> Self {
        Self {
            boredom: 100.0,
            annoyance: 0.0,
            lives: 9,
            boredom_decay_rate: 1.5,
            annoyance_decay_rate: 0.3,
            annoyance_max: 100.0,
        }
    }

    /// Call every frame. Returns GameOver if any meter triggers end.
    pub fn update(&mut self, dt: f32) -> Option<GameOver> {
        // Boredom always decays
        self.boredom = (self.boredom - self.boredom_decay_rate * dt).max(0.0);

        // Annoyance slowly decays when quiet
        self.annoyance = (self.annoyance - self.annoyance_decay_rate * dt).max(0.0);

        // Check game over conditions
        if self.boredom <= 0.0 {
            return Some(GameOver::BoredToDeath);
        }
        if self.annoyance >= self.annoyance_max {
            return Some(GameOver::ThrownOut);
        }
        if self.lives == 0 {
            return Some(GameOver::NoLivesLeft);
        }
        None
    }

    /// Cat destroyed something! Restores boredom, adds annoyance.
    /// value: monetary value of destroyed item ($)
    /// noise: how loud the destruction was (0.0-1.0)
    pub fn on_destroy(&mut self, value: f32, noise: f32) {
        // Boredom restored proportional to value
        // $15 cup = +1.5 boredom, $800 sofa = +80 boredom (capped at 100)
        let boredom_restore = (value * 0.1).min(50.0);
        self.boredom = (self.boredom + boredom_restore).min(100.0);

        // Annoyance from noise
        // Glass (noise=1.0) = +30 annoyance
        // Fabric (noise=0.05) = +1.5 annoyance
        let annoyance_add = noise * 30.0;
        self.annoyance = (self.annoyance + annoyance_add).min(self.annoyance_max);
    }

    /// Cat took damage (fall, electricity, heavy object)
    pub fn lose_life(&mut self) {
        if self.lives > 0 {
            self.lives -= 1;
        }
    }

    /// Cat is doing something mildly entertaining (exploring, sitting in box)
    pub fn mild_entertainment(&mut self, amount: f32) {
        self.boredom = (self.boredom + amount).min(100.0);
    }

    /// Boredom as 0.0-1.0 (for UI display)
    pub fn boredom_frac(&self) -> f32 { self.boredom / 100.0 }

    /// Annoyance as 0.0-1.0 (for UI display)
    pub fn annoyance_frac(&self) -> f32 { self.annoyance / self.annoyance_max }

    /// Is cat in danger zone? (boredom < 20 or annoyance > 80)
    pub fn is_critical(&self) -> bool {
        self.boredom < 20.0 || self.annoyance > 80.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let m = Meters::new();
        assert_eq!(m.boredom, 100.0);
        assert_eq!(m.annoyance, 0.0);
        assert_eq!(m.lives, 9);
    }

    #[test]
    fn test_boredom_decays() {
        let mut m = Meters::new();
        m.update(10.0); // 10 seconds
        assert!(m.boredom < 100.0);
        assert!(m.boredom > 0.0);
    }

    #[test]
    fn test_boredom_death() {
        let mut m = Meters::new();
        m.boredom = 1.0;
        let result = m.update(1.0); // 1 second, decay 1.5 → boredom goes to 0
        assert_eq!(result, Some(GameOver::BoredToDeath));
    }

    #[test]
    fn test_destroy_restores_boredom() {
        let mut m = Meters::new();
        m.boredom = 20.0;
        m.on_destroy(200.0, 0.5); // $200 vase
        assert!(m.boredom > 20.0); // restored!
        assert!(m.annoyance > 0.0); // but made noise!
    }

    #[test]
    fn test_annoyance_triggers_thrown_out() {
        let mut m = Meters::new();
        m.annoyance = 99.0;
        m.on_destroy(100.0, 1.0); // loud destruction
        let result = m.update(0.0);
        assert_eq!(result, Some(GameOver::ThrownOut));
    }

    #[test]
    fn test_lives() {
        let mut m = Meters::new();
        for _ in 0..9 {
            m.lose_life();
        }
        assert_eq!(m.lives, 0);
        assert_eq!(m.update(0.0), Some(GameOver::NoLivesLeft));
    }

    #[test]
    fn test_annoyance_decays_when_quiet() {
        let mut m = Meters::new();
        m.annoyance = 50.0;
        m.update(10.0); // 10 quiet seconds
        assert!(m.annoyance < 50.0);
    }

    #[test]
    fn test_critical_zone() {
        let mut m = Meters::new();
        assert!(!m.is_critical());
        m.boredom = 15.0;
        assert!(m.is_critical());
    }
}
