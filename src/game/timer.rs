// ═══════════════════════════════════════════════════════════════
// PURRGE — Game Timer
//
// 9 game hours (owner is at work). Maps to ~30-45 min real time.
// Time affects lighting, dog behavior, urgency.
// ═══════════════════════════════════════════════════════════════

/// Game clock: 9:00 AM → 6:00 PM (9 hours)
pub struct GameTimer {
    /// Elapsed game-hours (0.0 = 9:00 AM, 9.0 = 6:00 PM)
    pub hours_elapsed: f32,
    /// Total game hours (default 9)
    pub total_hours: f32,
    /// Real seconds per game hour (default: 200s = 3.3 min/hour → 30 min total)
    pub seconds_per_hour: f32,
}

impl GameTimer {
    pub fn new() -> Self {
        Self {
            hours_elapsed: 0.0,
            total_hours: 9.0,
            seconds_per_hour: 200.0, // 9 hours × 200s = 1800s = 30 min
        }
    }

    /// Update timer. Returns true when time is up (owner comes home).
    pub fn update(&mut self, dt: f32) -> bool {
        self.hours_elapsed += dt / self.seconds_per_hour;
        self.hours_elapsed >= self.total_hours
    }

    /// Current time as "HH:MM" string (starts at 9:00)
    pub fn clock_display(&self) -> String {
        let total_minutes = (9.0 + self.hours_elapsed) * 60.0;
        let hours = (total_minutes / 60.0) as u32;
        let minutes = (total_minutes % 60.0) as u32;
        format!("{:02}:{:02}", hours, minutes)
    }

    /// Progress 0.0-1.0 (how much time has passed)
    pub fn progress(&self) -> f32 {
        (self.hours_elapsed / self.total_hours).min(1.0)
    }

    /// Time remaining as fraction (1.0 = full time, 0.0 = owner arrives)
    pub fn remaining_frac(&self) -> f32 {
        1.0 - self.progress()
    }

    /// Is it getting late? (last 2 game hours — urgency increases)
    pub fn is_late(&self) -> bool {
        self.hours_elapsed > self.total_hours - 2.0
    }

    /// Lighting factor: morning=bright, afternoon=warm, evening=dim
    pub fn lighting(&self) -> f32 {
        let hour = 9.0 + self.hours_elapsed;
        if hour < 12.0 { 0.9 + (hour - 9.0) * 0.03 }      // morning: 0.9→1.0
        else if hour < 15.0 { 1.0 }                          // noon: peak
        else { 1.0 - (hour - 15.0) * 0.1 }                   // afternoon: 1.0→0.7
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_clock() {
        let t = GameTimer::new();
        assert_eq!(t.clock_display(), "09:00");
        assert_eq!(t.progress(), 0.0);
    }

    #[test]
    fn test_clock_advances() {
        let mut t = GameTimer::new();
        t.update(t.seconds_per_hour); // 1 game hour
        assert_eq!(t.clock_display(), "10:00");
        assert!((t.progress() - 1.0/9.0).abs() < 0.01);
    }

    #[test]
    fn test_time_up() {
        let mut t = GameTimer::new();
        let result = t.update(t.seconds_per_hour * 9.0); // 9 hours
        assert!(result); // owner came home!
        assert_eq!(t.clock_display(), "18:00");
    }

    #[test]
    fn test_is_late() {
        let mut t = GameTimer::new();
        assert!(!t.is_late());
        t.hours_elapsed = 8.0; // 5 PM
        assert!(t.is_late());
    }

    #[test]
    fn test_lighting() {
        let mut t = GameTimer::new();
        let morning = t.lighting();
        t.hours_elapsed = 6.0; // 3 PM
        let afternoon = t.lighting();
        t.hours_elapsed = 8.0; // 5 PM
        let evening = t.lighting();
        assert!(morning < afternoon);
        assert!(evening < afternoon);
    }
}
