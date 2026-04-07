// ═══════════════════════════════════════════════════════════════
// PURRGE — Scoring / Repair Bill
//
// Track everything the cat destroyed. Generate itemized bill.
// Owner personality multipliers. Shareable screenshot.
// ═══════════════════════════════════════════════════════════════

/// One line on the repair bill
#[derive(Clone, Debug)]
pub struct DamageEntry {
    pub item_name: String,
    pub material_name: String,
    pub base_value: f32,
    pub damage_percent: f32,  // 0.0-1.0
    pub cost: f32,            // base_value × damage_percent × personality_mult
}

/// Owner personality — affects item values (from design doc)
#[derive(Clone, Debug)]
pub enum OwnerType {
    Gamer,       // TV ×3, console ×5
    Artist,      // paintings ×3
    Bookworm,    // books ×2, shelf ×2
    Fitness,     // yoga mat ×2, weights ×2
    Minimalist,  // everything ×2, fewer items
    Hoarder,     // everything ×0.5, many items
    Nostalgic,   // photos ×5, max chaos
    PlantLover,  // plants ×3
    CatLover,    // cat items ×0 (no bonus)
    Normal,      // no multipliers
}

impl OwnerType {
    /// Value multiplier for an item based on owner personality
    pub fn multiplier(&self, item_name: &str) -> f32 {
        let name = item_name.to_lowercase();
        match self {
            OwnerType::Gamer => {
                if name.contains("tv") || name.contains("monitor") { 3.0 }
                else if name.contains("console") { 5.0 }
                else { 1.0 }
            }
            OwnerType::Artist => {
                if name.contains("paint") || name.contains("canvas") { 3.0 }
                else { 1.0 }
            }
            OwnerType::Bookworm => {
                if name.contains("book") || name.contains("shelf") { 2.0 }
                else { 1.0 }
            }
            OwnerType::Minimalist => 2.0, // everything is precious
            OwnerType::Hoarder => 0.5,    // nothing is precious
            OwnerType::Nostalgic => {
                if name.contains("photo") || name.contains("frame") { 5.0 }
                else { 1.0 }
            }
            OwnerType::PlantLover => {
                if name.contains("plant") || name.contains("pot") { 3.0 }
                else { 1.0 }
            }
            OwnerType::CatLover => {
                if name.contains("cat") || name.contains("scratch") { 0.0 }
                else { 1.0 }
            }
            _ => 1.0,
        }
    }
}

/// Tracks all destruction for the repair bill
pub struct RepairBill {
    pub entries: Vec<DamageEntry>,
    pub owner: OwnerType,
    pub chain_bonus_total: f32,
}

impl RepairBill {
    pub fn new(owner: OwnerType) -> Self {
        Self { entries: Vec::new(), owner, chain_bonus_total: 0.0 }
    }

    /// Record damage to an item
    pub fn record(&mut self, item_name: &str, material_name: &str, base_value: f32, damage_percent: f32) {
        let mult = self.owner.multiplier(item_name);
        let cost = base_value * damage_percent * mult;
        self.entries.push(DamageEntry {
            item_name: item_name.to_string(),
            material_name: material_name.to_string(),
            base_value, damage_percent, cost,
        });
    }

    /// Record chain reaction bonus
    pub fn add_chain_bonus(&mut self, bonus: f32) {
        self.chain_bonus_total += bonus;
    }

    /// Total damage cost
    pub fn total(&self) -> f32 {
        self.entries.iter().map(|e| e.cost).sum::<f32>() + self.chain_bonus_total
    }

    /// Number of items damaged
    pub fn items_damaged(&self) -> usize {
        self.entries.len()
    }

    /// Format as text for display / screenshot
    pub fn format_bill(&self) -> String {
        let mut s = String::new();
        s.push_str("╔══════════════════════════════════╗\n");
        s.push_str("║       REPAIR BILL               ║\n");
        s.push_str("╠══════════════════════════════════╣\n");
        for entry in &self.entries {
            s.push_str(&format!("║ {:<20} ${:>8.2} ║\n", entry.item_name, entry.cost));
        }
        if self.chain_bonus_total > 0.0 {
            s.push_str("╠──────────────────────────────────╣\n");
            s.push_str(&format!("║ Chain reactions      ${:>8.2} ║\n", self.chain_bonus_total));
        }
        s.push_str("╠══════════════════════════════════╣\n");
        s.push_str(&format!("║ TOTAL                ${:>8.2} ║\n", self.total()));
        s.push_str("╚══════════════════════════════════╝\n");
        s
    }

    /// Rating based on total damage
    pub fn rating(&self) -> &'static str {
        let t = self.total();
        if t < 50.0 { "Buddhist Cat — achieved nothing" }
        else if t < 500.0 { "Curious Kitten" }
        else if t < 2000.0 { "Domestic Terrorist" }
        else if t < 5000.0 { "Feline Wrecking Ball" }
        else if t < 10000.0 { "Category 5 Catastrophe" }
        else { "LEGENDARY DESTROYER" }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_bill() {
        let bill = RepairBill::new(OwnerType::Normal);
        assert_eq!(bill.total(), 0.0);
        assert_eq!(bill.rating(), "Buddhist Cat — achieved nothing");
    }

    #[test]
    fn test_record_damage() {
        let mut bill = RepairBill::new(OwnerType::Normal);
        bill.record("Vase", "Ceramic", 200.0, 1.0);
        bill.record("Sofa", "Fabric", 800.0, 0.5);
        assert_eq!(bill.items_damaged(), 2);
        assert!((bill.total() - 600.0).abs() < 0.01); // 200 + 400
    }

    #[test]
    fn test_gamer_multiplier() {
        let mut bill = RepairBill::new(OwnerType::Gamer);
        bill.record("TV", "Glass", 1200.0, 1.0);
        assert!((bill.total() - 3600.0).abs() < 0.01); // 1200 × 3.0
    }

    #[test]
    fn test_hoarder_discount() {
        let mut bill = RepairBill::new(OwnerType::Hoarder);
        bill.record("Vase", "Ceramic", 200.0, 1.0);
        assert!((bill.total() - 100.0).abs() < 0.01); // 200 × 0.5
    }

    #[test]
    fn test_chain_bonus() {
        let mut bill = RepairBill::new(OwnerType::Normal);
        bill.record("Table leg", "Wood", 50.0, 1.0);
        bill.add_chain_bonus(75.0); // table fell → vase broke
        assert!((bill.total() - 125.0).abs() < 0.01);
    }

    #[test]
    fn test_rating_tiers() {
        let mut bill = RepairBill::new(OwnerType::Normal);
        assert!(bill.rating().contains("Buddhist"));
        bill.record("Everything", "Glass", 6000.0, 1.0);
        assert!(bill.rating().contains("Catastrophe"));
    }

    #[test]
    fn test_format_bill() {
        let mut bill = RepairBill::new(OwnerType::Normal);
        bill.record("Vase", "Ceramic", 200.0, 1.0);
        let text = bill.format_bill();
        assert!(text.contains("Vase"));
        assert!(text.contains("200.00"));
        assert!(text.contains("TOTAL"));
    }
}
