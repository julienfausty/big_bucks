use core::f64;

use ndarray::{Array2, s};

use plotters::prelude::*;

mod fetch;
use fetch::{MarketChartQuery, query_market_chart};

mod interp;
use interp::rebase;

mod regression;

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

    root_area
        .present()
        .expect("Failed to write simple price plot to file.");
}

fn plot_scatter_w_histograms(series: Array2<f64>) {
    if series.shape()[1] != 2 {
        panic!("Cannot scatter plot with histograms for more than 2 series.");
    }

    let x_range = series
        .slice(s![.., 0])
        .fold((f64::INFINITY, 0.0), |acc, val| {
            (val.min(acc.0), val.max(acc.1))
        });

    let y_range = series
        .slice(s![.., 1])
        .fold((f64::INFINITY, 0.0), |acc, val| {
            (val.min(acc.0), val.max(acc.1))
        });

    let root_area = BitMapBackend::new("/tmp/scatter_prices.png", (1200, 800)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let areas = root_area.split_by_breakpoints([1000], [200]);

    let mut x_hist_ctx = ChartBuilder::on(&areas[0])
        .y_label_area_size(40)
        .build_cartesian_2d(
            (x_range.0..x_range.1)
                .step((x_range.1 - x_range.0) / 50.0)
                .use_round()
                .into_segmented(),
            0..(series.shape()[0] / 10),
        )
        .unwrap();

    let mut y_hist_ctx = ChartBuilder::on(&areas[3])
        .x_label_area_size(40)
        .build_cartesian_2d(
            0..(series.shape()[0] / 10),
            (y_range.0..y_range.1)
                .step((y_range.1 - y_range.0) / 50.0)
                .use_round(),
        )
        .unwrap();

    let mut scatter_ctx = ChartBuilder::on(&areas[2])
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)
        .unwrap();

    scatter_ctx.configure_mesh().draw().unwrap();

    scatter_ctx
        .draw_series(
            series
                .axis_iter(ndarray::Axis(0))
                .map(|view| Circle::new((view[0], view[1]), 2, &GREEN)),
        )
        .unwrap();

    let x_hist = Histogram::vertical(&x_hist_ctx)
        .style(GREEN.filled())
        .margin(0)
        .data(series.slice(s![.., 0]).iter().map(|val| (*val, 1)));

    let y_hist = Histogram::horizontal(&y_hist_ctx)
        .style(GREEN.filled())
        .margin(0)
        .data(series.slice(s![.., 1]).iter().map(|val| (*val, 1)));

    x_hist_ctx.draw_series(x_hist).unwrap();
    y_hist_ctx.draw_series(y_hist).unwrap();

    root_area
        .present()
        .expect("Failed to write scatter plot to file.");
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

    plot_simple_normalized_prices(wrapped.clone());

    let rebased = rebase(wrapped).expect("Failed to rebase charts onto single timeline.");

    plot_scatter_w_histograms(rebased.series);

    Ok(())
}
