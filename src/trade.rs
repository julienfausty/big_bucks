use log;
use tracing::Level;
use tracing_log;
use tracing_subscriber;

use serde::{Deserialize, Serialize};

use tokio::sync::watch;
use tokio::time::{Duration, sleep};

use std::collections::HashMap;

use big_bucks::fetch::{MarketChartQuery, query_market_chart};
use big_bucks::interp::{RebasedSeries, rebase};
use big_bucks::orders::{Confirmation, Order};
use big_bucks::signals::{Signal, StatArbSignal};
use big_bucks::strategy::{StatArbModel, StatArbPolicy};

const ADF_THRESHOLD: f64 = -2.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Account {}

impl Account {
    pub async fn state(&self) -> Result<(f64, HashMap<String, f64>), String> {
        Ok((100.0, HashMap::new()))
    }
    pub async fn execute(&self, order: Order) -> Result<Confirmation, String> {
        Ok(Confirmation(order))
    }
}

async fn fetch_model_data(assets: (String, String)) -> Result<RebasedSeries, String> {
    let chart0 = query_market_chart(MarketChartQuery {
        coin: assets.0,
        currency: "usd".into(),
        since: "max".into(),
        interval: "hourly".into(),
        precision: "full".into(),
    })
    .await?;

    let chart1 = query_market_chart(MarketChartQuery {
        coin: assets.1,
        currency: "usd".into(),
        since: "max".into(),
        interval: "hourly".into(),
        precision: "full".into(),
    })
    .await?;

    let wrapped = vec![chart0.prices.clone(), chart1.prices.clone()];

    rebase(wrapped)
}

async fn fit_model(assets: (String, String)) -> Result<StatArbModel, String> {
    let ticker_map: HashMap<String, String> = HashMap::from([
        ("BTC".into(), "bitcoin".into()),
        ("ETH".into(), "ethereum".into()),
        ("SOL".into(), "solana".into()),
    ]);

    let fetch_assets = match (ticker_map.get(&assets.0), ticker_map.get(&assets.1)) {
        (Some(zero), Some(one)) => (zero.clone(), one.clone()),
        _ => {
            return Err(
                "Failed to translate provided asset tickers into ticker for fetch.".to_string(),
            );
        }
    };
    let model_data = fetch_model_data(fetch_assets).await?;

    StatArbModel::new(ADF_THRESHOLD, assets, model_data)
}

pub struct ModelPipe {
    pipe: watch::Receiver<Result<StatArbModel, String>>,
}

impl ModelPipe {
    pub async fn new(assets: (String, String)) -> Result<ModelPipe, String> {
        let (tx_model, rx_model) = watch::channel(fit_model(assets.clone()).await);

        tokio::spawn(async move {
            loop {
                sleep(Duration::from_hours(1)).await;
                match tx_model.send(fit_model(assets.clone()).await) {
                    Ok(()) => (),
                    Err(message) => {
                        log::error!("{message}");
                        break;
                    }
                };
            }
        });

        Ok(ModelPipe { pipe: rx_model })
    }

    pub async fn newest(&mut self) -> Result<StatArbModel, String> {
        self.pipe.borrow_and_update().clone()
    }
}

#[tokio::main]
async fn main() -> Result<(), String> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stdout)
        .with_max_level(Level::INFO)
        .init();

    let account = Account {};
    log::info!(
        "Account initialized {}",
        serde_json::to_string(&account).unwrap_or_else(|e| format!("{:?}", e))
    );

    let trading_pair = ("BTC".to_string(), "ETH".to_string());
    log::info!("Trading pair {trading_pair:?}");

    let mut model_pipe = ModelPipe::new(trading_pair.clone()).await?;
    let model = model_pipe.newest().await?;

    log::info!(
        "Model computed {}",
        serde_json::to_string(&model).unwrap_or_else(|e| format!("{:?}", e))
    );

    Ok(())
}
