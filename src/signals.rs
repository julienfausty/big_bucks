use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatArbSignal {
    pub assets: (String, String),
    pub z_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Signal {
    StatArb(StatArbSignal),
    MarketUncertain,
}
