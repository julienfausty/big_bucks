use kraken_async_rs::wss::{
    ChannelMessage, KrakenMessageStream, KrakenWSSClient, Message, Ticker, TickerSubscription,
    WS_KRAKEN, WS_KRAKEN_AUTH, WssMessage,
};

use serde::Deserialize;

use num_traits::cast::ToPrimitive;

use log;

use tokio::sync::watch;
use tokio::time::{Duration, timeout};
use tokio_stream::StreamExt;

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

const TIMEOUT_DURATION: u64 = 60;

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
    receiver: watch::Receiver<Result<StampedPricePair, String>>,
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

        let mut client = KrakenWSSClient::new_with_urls(WS_KRAKEN, WS_KRAKEN_AUTH);
        let mut connection = match client.connect::<WssMessage>().await {
            Ok(connection) => connection,
            Err(message) => return Err(format!("Failed connection to Kraken API:\n{}", message)),
        };

        let subscription =
            TickerSubscription::new(vec![krak_assets.0.clone(), krak_assets.1.clone()]);
        let subscription_message = Message::new_subscription(subscription, 42);

        match connection.send(&subscription_message).await {
            Ok(_) => (),
            Err(message) => {
                return Err(format!(
                    "Failed to subscribe to ticker prices:\n{}",
                    message
                ));
            }
        }

        let (tx_price, rx_price) = watch::channel(Ok((0, 0.0, 0.0)));

        tokio::spawn(async move {
            let mut cache: StampedPricePair = (0, 0.0, 0.0);
            loop {
                match timeout(Duration::from_secs(TIMEOUT_DURATION), connection.next()).await {
                    Ok(Some(communication)) => match communication {
                        Ok(WssMessage::Channel(packet)) => match packet {
                            ChannelMessage::Heartbeat => (),
                            ChannelMessage::Ticker(tick) => {
                                cache.0 = match SystemTime::now().duration_since(UNIX_EPOCH) {
                                    Ok(duration) => duration.as_millis() as u64,
                                    Err(message) => {
                                        log::error!(
                                            "Could not determine current timestamp:\n{}",
                                            message
                                        );
                                        cache.0
                                    }
                                };
                                if tick.data.symbol == krak_assets.0 {
                                    if let Some(val) = tick.data.vwap.to_f64() {
                                        cache.1 = val;
                                    }
                                } else if tick.data.symbol == krak_assets.1 {
                                    if let Some(val) = tick.data.vwap.to_f64() {
                                        cache.2 = val;
                                    }
                                } else {
                                    log::info!("Received odd tick data: {:?}", tick)
                                }

                                match tx_price.send(Ok(cache)) {
                                    Ok(()) => (),
                                    Err(message) => log::error!(
                                        "Error sending price data on pipe:\n{}",
                                        message
                                    ),
                                };
                            }
                            _ => {
                                log::info!("Received channel message from price pipe: {:?}", packet)
                            }
                        },
                        Ok(WssMessage::Method(info)) => {
                            log::info!("{:?}", info);
                        }
                        Ok(WssMessage::Error(err)) => {
                            match tx_price
                                .send(Err(format!("Error from Kraken connection {:?}", err)))
                            {
                                Ok(()) => (),
                                Err(message) => {
                                    log::error!("Error sending price data on pipe:\n{}", message)
                                }
                            };
                        }
                        _ => log::info!("Received message from price pipe: {:?}", communication),
                    },
                    Ok(None) => {
                        match tx_price.send(Err("Connection to Kraken API closed.".to_string())) {
                            Ok(()) => (),
                            Err(message) => {
                                log::error!(
                                    "Error sending Kraken closed data on pipe:\n{}",
                                    message
                                );
                            }
                        };
                        break;
                    }
                    Err(message) => {
                        match tx_price.send(Err(format!(
                            "Failure in web socket communication with Kraken API:\n{}",
                            message
                        ))) {
                            Ok(()) => (),
                            Err(message) => {
                                log::error!("Error sending error data on pipe:\n{}", message);
                            }
                        };
                        break;
                    }
                }
            }
        });

        Ok(PricePairPipe { receiver: rx_price })
    }

    pub async fn newest_change(&mut self) -> Result<StampedPricePair, String> {
        match self.receiver.changed().await {
            Ok(()) => self.receiver.borrow_and_update().clone(),
            Err(message) => Err(format!("{}", message)),
        }
    }
}
