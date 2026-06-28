use core::f64;

use ndarray::{Array1, Array2, s};

use plotters::prelude::*;

mod fetch;
use fetch::{MarketChartQuery, query_market_chart};

mod interp;
use interp::rebase;

mod regression;
use regression::check_augmented_dicky_fuller;

const ADF_THRESHOLD: f64 = -0.5;

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

fn plot_histograms(values: Vec<Array1<f64>>) {
    let x_range = values
        .iter()
        .fold((f64::INFINITY, -f64::INFINITY), |range, array| {
            let minmax = (
                array.iter().fold(
                    f64::INFINITY,
                    |min, val| if *val < min { *val } else { min },
                ),
                array.iter().fold(
                    -f64::INFINITY,
                    |max, val| if *val > max { *val } else { max },
                ),
            );
            (
                if minmax.0 < range.0 {
                    minmax.0
                } else {
                    range.0
                },
                if minmax.1 > range.1 {
                    minmax.1
                } else {
                    range.1
                },
            )
        });

    let max_len = values.iter().fold(
        0,
        |len, array| if array.len() > len { array.len() } else { len },
    );

    let root_area = BitMapBackend::new("/tmp/histograms.png", (1200, 800)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut hist_ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(
            (x_range.0..x_range.1)
                .step((x_range.1 - x_range.0) / 100.0)
                .use_round()
                .into_segmented(),
            0..max_len,
        )
        .unwrap();

    hist_ctx.configure_mesh().draw().unwrap();

    let colors = vec![GREEN, RED, BLUE, YELLOW];

    let mut i_color = 0;
    for series in values.into_iter() {
        let hist = Histogram::vertical(&hist_ctx)
            .style(colors[i_color].mix(0.5).filled())
            .margin(0)
            .data(series.iter().map(|val| (*val, 1)));
        i_color = (i_color + 1) % colors.len();

        hist_ctx.draw_series(hist).unwrap();
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

    plot_simple_normalized_prices(wrapped.clone());

    let rebased = rebase(wrapped).expect("Failed to rebase charts onto single timeline.");

    plot_scatter_w_histograms(rebased.series.clone());

    let adf_checks: Vec<_> = vec![
        (
            rebased.time.map(|t| *t as f64),
            rebased.series.slice(s![.., 0]).to_owned(),
        ),
        (
            rebased.time.map(|t| *t as f64),
            rebased.series.slice(s![.., 1]).to_owned(),
        ),
    ]
    .into_iter()
    .map(|(time_view, val_view)| check_augmented_dicky_fuller(time_view, val_view).unwrap())
    .collect();

    println!("Bitcoin ADF check: \n{:?}\n", adf_checks[0].criterion);
    println!("Ethereum ADF check: \n{:?}", adf_checks[1].criterion);

    if adf_checks[0].criterion < ADF_THRESHOLD || adf_checks[1].criterion < ADF_THRESHOLD {
        return Err("Failed integrated of order 1 checks for one of the series.".to_string());
    }

    plot_histograms(
        adf_checks
            .into_iter()
            .map(|check| {
                check.regression.residuals.flatten().to_owned() / check.regression.variance.sqrt()
            })
            .collect(),
    );

    Ok(())
}
