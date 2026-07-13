use serde::Deserialize;

use tokio::sync::watch;

use std::collections::HashMap;

#[derive(Debug)]
pub struct MarketChartQuery {
    pub coin: String,
    pub currency: String,
    pub since: String,
    pub interval: String,
    pub precision: String,
}

#[derive(Debug, Deserialize)]
pub struct MarketChart {
    pub prices: Vec<(u64, f64)>,
    pub total_volumes: Vec<(u64, f64)>,
    pub market_caps: Vec<(u64, f64)>,
}

pub async fn query_market_chart(query: MarketChartQuery) -> Result<MarketChart, String> {
    let (coin, params) = (
        query.coin,
        [
            ("vs_currency", query.currency),
            ("days", query.since),
            ("interval", query.interval),
            ("precision", query.precision),
        ],
    );

    let url = reqwest::Url::parse_with_params(
        &format!("https://api.coingecko.com/api/v3/coins/{coin}/market_chart"),
        &params,
    )
    .expect("Could not parse url correctly.");
    match reqwest::Client::new()
        .get(url)
        .header("User-Agent", "big_bucks 0.1")
        .send()
        .await
    {
        Ok(response) => Ok(
            match serde_json::from_str(&match response.text().await {
                Ok(body) => body,
                Err(message) => return Err(format!("{}", message)),
            }) {
                Ok(chart) => chart,
                Err(message) => return Err(format!("{}", message)),
            },
        ),
        Err(message) => Err(format!("{}", message)),
    }
}

pub type StampedPricePair = (u64, f64, f64);

pub struct PricePairPipe {
    pipe: watch::Receiver<Result<StampedPricePair, String>>,
}

impl PricePairPipe {
    pub async fn new(assets: (String, String)) -> Result<PricePairPipe, String> {
        let asset_mapping = HashMap::from([
            ("BTC".to_string(), "BTC/USD".to_string()),
            ("ETH".to_string(), "ETH/USD".to_string()),
            ("SOL".to_string(), "SOL/USD".to_string()),
        ]);

        let krak_assets = match (asset_mapping.get(&assets.0), asset_mapping.get(&assets.1)) {
            (Some(krak0), Some(krak1)) => (krak0.clone(), krak1.clone()),
            _ => return Err("Provided unsupported asset name to price pipeline creation.".into()),
        };

        Err("Not finished implementing".to_string())
    }

    pub async fn newest_change(&mut self) -> Result<StampedPricePair, String> {
        match self.pipe.changed().await {
            Ok(()) => self.pipe.borrow_and_update().clone(),
            Err(message) => Err(format!("{}", message)),
        }
    }
}
