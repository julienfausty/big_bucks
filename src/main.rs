use core::f64;

use plotters::prelude::*;

mod fetch;
use fetch::{MarketChartQuery, query_market_chart};

mod interp;

fn plot_simple_normalized_prices(prices: Vec<Vec<(u64, f64)>>) {
    let prices: Vec<_> = prices
        .into_iter()
        .map(|mut series| {
            series.sort_by(|l, r| l.0.cmp(&r.0));
            series
        })
        .collect();

    let time_range = (
        prices
            .iter()
            .fold(u64::MAX, |acc, series| series[0].0.min(acc)),
        prices
            .iter()
            .fold(u64::MIN, |acc, series| series.last().unwrap().0.max(acc)),
    );
    let time_width = (time_range.1 - time_range.0) as f64;

    let normalize = |prices: Vec<(u64, f64)>| -> Vec<(f64, f64)> {
        let last_price = prices.last().unwrap().1;
        prices
            .into_iter()
            .map(|(t, p)| ((t - time_range.0) as f64 / time_width, p / last_price))
            .collect()
    };

    let normalized: Vec<_> = prices.into_iter().map(normalize).collect();

    let root_area = BitMapBackend::new("/tmp/simple_prices.png", (1200, 800)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let price_range = (
        0.9 * normalized.iter().fold(f64::INFINITY, |acc, series| {
            series
                .iter()
                .fold(acc, |acc_inner, (_, p)| p.min(acc_inner))
        }),
        1.1 * normalized.iter().fold(0.0, |acc, series| {
            series
                .iter()
                .fold(acc, |acc_inner, (_, p)| p.max(acc_inner))
        }),
    );

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Crypto Prices", ("monospace", 40))
        .build_cartesian_2d(0.0..1.0, price_range.0..price_range.1)
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    for series in normalized.into_iter() {
        ctx.draw_series(LineSeries::new(series, &GREEN)).unwrap();
    }
}

#[tokio::main]
async fn main() -> Result<(), String> {
    let bitcoin_chart = query_market_chart(MarketChartQuery {
        coin: "bitcoin".into(),
        currency: "usd".into(),
        since: "max".into(),
        interval: "hourly".into(),
        precision: "full".into(),
    })
    .await
    .unwrap();

    let ethereum_chart = query_market_chart(MarketChartQuery {
        coin: "ethereum".into(),
        currency: "usd".into(),
        since: "max".into(),
        interval: "hourly".into(),
        precision: "full".into(),
    })
    .await
    .unwrap();

    let wrapped = vec![bitcoin_chart.prices.clone(), ethereum_chart.prices.clone()];

    plot_simple_normalized_prices(wrapped);

    Ok(())
}
