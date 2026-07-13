use serde::{Deserialize, Serialize};

use std::collections::HashMap;

use big_bucks::fetch::{MarketChartQuery, query_market_chart};
use big_bucks::interp::rebase;
use big_bucks::orders::{Confirmation, Order};
use big_bucks::signals::{Signal, StatArbSignal};
use big_bucks::strategy::{StatArbModel, StatArbPolicy};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Account {}

impl Account {
    pub async fn state(&self) -> Result<(f64, HashMap<String, f64>)> {
        Ok((100.0, HashMap::new()))
    }
    pub async fn execute(&self, order: Order) -> Result<Confirmation, String> {
        Ok(Confirmation(order))
    }
}

#[tokio::main]
async fn main() -> Result<(), String> {
    Ok(())
}
