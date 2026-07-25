use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatArbSignal {
    pub assets: (String, String),
    pub prices: (f64, f64),
    pub z_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Signal {
    StatArb(StatArbSignal),
    MarketUncertain,
}
