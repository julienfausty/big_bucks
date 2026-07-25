use log;
use tracing::Level;
use tracing_log;
use tracing_subscriber;

use dotenv;

use serde::{Deserialize, Serialize};

use tokio::sync::watch;
use tokio::time::{Duration, sleep};

use std::collections::HashMap;

use big_bucks::fetch::{
    MarketChartQuery, PricePairPipe, fetch_kraken_account_data, query_market_chart,
};
use big_bucks::interp::{RebasedSeries, rebase};
use big_bucks::orders::{Confirmation, Order};
use big_bucks::signals::Signal;
use big_bucks::strategy::{StatArbModel, StatArbPolicy};

const ADF_THRESHOLD: f64 = -2.0;
const ENTRY_SCORE: f64 = 1.25;
const EXIT_SCORE: f64 = 0.25;
const CUT_LOSS: f64 = 3.0;
const MAX_RISK: f64 = 0.05;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Account {}

impl Account {
    pub async fn state(&self) -> Result<(f64, HashMap<String, f64>), String> {
        fetch_kraken_account_data().await
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
    dotenv::dotenv().ok();

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
    let mut price_pipe = PricePairPipe::new(trading_pair.clone()).await?;

    loop {
        let (timestamp, p0, p1) = price_pipe.newest_change().await?;
        log::info!(
            "New price action (t, {}, {}): ({}, {}, {})",
            trading_pair.0,
            trading_pair.1,
            timestamp,
            p0,
            p1
        );

        let account_state = account.state().await?;
        log::info!(
            "Account state: Cash {}, Portfolio {:?}",
            account_state.0,
            account_state.1
        );

        let policy = StatArbPolicy::new(
            account_state.0,
            account_state.1,
            ENTRY_SCORE,
            EXIT_SCORE,
            CUT_LOSS,
            MAX_RISK,
        );
        log::info!(
            "Policy initialized {}",
            serde_json::to_string(&policy).unwrap_or_else(|e| format!("{:?}", e))
        );

        let signal = match model_pipe.newest().await {
            Ok(model) => {
                log::info!(
                    "Model computed {}",
                    serde_json::to_string(&model).unwrap_or_else(|e| format!("{:?}", e))
                );
                Signal::StatArb(model.signal(timestamp, (p0, p1)))
            }
            Err(_) => Signal::MarketUncertain,
        };

        log::info!(
            "Signal {}",
            serde_json::to_string(&signal).unwrap_or_else(|e| format!("{:?}", e))
        );

        let orders = policy.evaluate(signal);

        log::info!(
            "Orders generated {}",
            serde_json::to_string(&orders).unwrap_or_else(|e| format!("{:?}", e))
        );

        let mut order_fail = false;
        match orders {
            Some(orders) => {
                for order in orders.into_iter() {
                    match account.execute(order).await {
                        Ok(confirmation) => log::info!(
                            "Order completed {}",
                            serde_json::to_string(&confirmation)
                                .unwrap_or_else(|e| format!("{:?}", e))
                        ),
                        Err(message) => {
                            log::error!("Could not complete order: {}", message);
                            order_fail = true;
                        }
                    };
                }
            }
            None => (),
        };

        if order_fail {
            break;
        }
    }

    Err("Program exited loop".to_string())
}
