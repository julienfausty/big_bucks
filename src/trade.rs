use serde::{Deserialize, Serialize};

use ndarray::s;

use std::collections::HashMap;

use big_bucks::fetch::{MarketChartQuery, query_market_chart};
use big_bucks::interp::{RebasedSeries, rebase};
use big_bucks::regression::{check_augmented_dicky_fuller, check_johansen};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StatArbSignal {
    assets: (String, String),
    z_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StatArbModel {
    assets: (String, String),
    intercept: f64,
    time_slope: f64,
    relationship: (f64, f64),
    mean: f64,
    deviation: f64,
}

impl StatArbModel {
    pub fn new(
        adf_critical: f64,
        assets: (String, String),
        series: RebasedSeries,
    ) -> Result<StatArbModel, String> {
        if series.series.shape()[1] != 2 {
            return Err(
                "Found series with more or less than 2 prices. Only pair trading supported for now.".to_string(),
            );
        }

        let adf_checks: Vec<_> = vec![
            (
                series.time.map(|t| *t as f64),
                series.series.slice(s![.., 0]).to_owned(),
            ),
            (
                series.time.map(|t| *t as f64),
                series.series.slice(s![.., 1]).to_owned(),
            ),
        ]
        .into_iter()
        .map(|(time_view, val_view)| check_augmented_dicky_fuller(time_view, val_view).unwrap())
        .collect();

        if adf_checks[0].criterion < adf_critical || adf_checks[1].criterion < adf_critical {
            return Err("Failed integrated of order 1 checks for one of the series.".to_string());
        }

        let johansen_check = check_johansen(series.time.map(|t| *t as f64), series.series.clone())?;

        if johansen_check.estimated_rank != 1 {
            return Err("Could not find cointegration relationship in assets".to_string());
        }

        let (adjustment_vectors, cointegration_relationships) =
            johansen_check.factor_error_correction()?;

        let intercept = johansen_check
            .regression
            .solution
            .slice(s![0, ..])
            .flatten()
            .dot(&adjustment_vectors.flatten());
        let time_slope = johansen_check
            .regression
            .solution
            .slice(s![1, ..])
            .flatten()
            .dot(&adjustment_vectors.flatten());

        let spread = intercept
            + time_slope * series.time.map(|t| *t as f64)
            + series
                .series
                .dot(&cointegration_relationships.t())
                .flatten()
                .to_owned();

        let mean = spread.sum() / (spread.len() as f64);
        let deviation =
            ((spread.clone() - mean).map(|val| val.powf(2.0)).sum() / (spread.len() as f64)).sqrt();

        Ok(StatArbModel {
            assets,
            intercept,
            time_slope,
            relationship: (
                cointegration_relationships[[0, 0]],
                cointegration_relationships[[1, 0]],
            ),
            mean,
            deviation,
        })
    }

    pub fn signal(&self, t: u64, prices: (f64, f64)) -> StatArbSignal {
        StatArbSignal {
            assets: self.assets.clone(),
            z_score: ((self.intercept - self.mean)
                + self.time_slope * (t as f64)
                + self.relationship.0 * prices.0
                + self.relationship.1 * prices.1)
                / self.deviation,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Signal {
    StatArb(StatArbSignal),
    MarketUncertain,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Order {
    Open((String, f64)),
    Close(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StatArbPolicy {
    cash: f64,
    portfolio: HashMap<String, f64>,
    entry_score: f64,
    exit_score: f64,
    cut_loss: f64,
    max_risk: f64,
}

impl StatArbPolicy {
    pub fn new(
        cash: f64,
        portfolio: HashMap<String, f64>,
        entry_score: f64,
        exit_score: f64,
        cut_loss: f64,
        max_risk: f64,
    ) -> StatArbPolicy {
        StatArbPolicy {
            cash,
            portfolio,
            entry_score,
            exit_score,
            cut_loss,
            max_risk,
        }
    }

    pub fn evaluate(&self, signal: Signal) -> Option<Vec<Order>> {
        match signal {
            Signal::StatArb(score) => {
                if score.z_score.abs() >= self.entry_score {
                    if self.portfolio.get(&score.assets.0).is_some()
                        || self.portfolio.get(&score.assets.1).is_some()
                    {
                        None
                    } else {
                        let volume = self.cash * self.max_risk;
                        let long = volume / 2.0;
                        let short = -volume / 2.0;
                        if score.z_score < 0.0 {
                            Some(vec![
                                Order::Open((score.assets.0, long)),
                                Order::Open((score.assets.1, short)),
                            ])
                        } else {
                            Some(vec![
                                Order::Open((score.assets.0, short)),
                                Order::Open((score.assets.1, long)),
                            ])
                        }
                    }
                } else if score.z_score.abs() <= self.exit_score
                    || score.z_score.abs() >= self.cut_loss
                {
                    let mut orders = Vec::new();
                    if self.portfolio.get(&score.assets.0).is_some() {
                        orders.push(Order::Close(score.assets.0));
                    }
                    if self.portfolio.get(&score.assets.1).is_some() {
                        orders.push(Order::Close(score.assets.1));
                    }

                    Some(orders)
                } else {
                    None
                }
            }
            Signal::MarketUncertain => {
                if !self.portfolio.is_empty() {
                    Some(
                        self.portfolio
                            .iter()
                            .map(|(asset, _)| Order::Close(asset.clone()))
                            .collect(),
                    )
                } else {
                    None
                }
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Confirmation(Order);

struct Executor {}

impl Executor {
    pub async fn execute(order: Order) -> Result<Confirmation, String> {
        Ok(Confirmation(order))
    }
}

#[tokio::main]
async fn main() -> Result<(), String> {
    Ok(())
}
