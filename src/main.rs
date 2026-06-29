use core::f64;

use ndarray::{Array1, Array2, s};
use ndarray_linalg::SVDInto;

use plotters::prelude::*;

use std::iter::zip;

mod fetch;
use fetch::{MarketChartQuery, query_market_chart};

mod interp;
use interp::rebase;

mod regression;
use regression::{autoregress, check_augmented_dicky_fuller, check_johansen};

const ADF_THRESHOLD: f64 = -2.0;

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

fn factor_error_correction(
    rank: usize,
    linear_map: Array2<f64>,
) -> Result<(Array2<f64>, Array2<f64>), String> {
    let base_dim = linear_map.shape()[1];

    if rank > base_dim {
        return Err(format!(
            "Provided rank was above base_dim: {} > {}",
            rank, base_dim
        ));
    }

    let (u, svd, vt) = match linear_map.svd_into(true, true) {
        Ok((Some(u), svd, Some(vt))) => (u, svd, vt),
        Ok((None, _, _)) => return Err("Did not get vt in SVD decomposition.".to_string()),
        Ok((_, _, None)) => return Err("Did not get vt in SVD decomposition.".to_string()),
        Err(message) => {
            return Err(format!(
                "Failed the SVD when looking for cointegration relationship:\n{}",
                message
            ));
        }
    };

    let mut potentials = zip(
        zip(u.columns().into_iter(), svd.into_iter()),
        vt.rows().into_iter(),
    )
    .map(|((col, val), row)| (col.to_owned() * row[0], val, row.to_owned() / row[0]))
    .collect::<Vec<_>>();

    potentials.sort_by(|l, r| l.1.total_cmp(&r.1));
    let potentials = potentials.into_iter().rev().collect::<Vec<_>>();

    let mut relationships = Array2::<f64>::zeros((rank, base_dim));
    let mut adjustments = Array2::<f64>::zeros((base_dim, rank));
    for i_rank in 0..rank {
        relationships
            .slice_mut(s![i_rank, ..])
            .assign(&(potentials[i_rank].2.clone()));
        adjustments
            .slice_mut(s![.., i_rank])
            .assign(&(potentials[i_rank].0.clone() * potentials[i_rank].1));
    }

    Ok((adjustments, relationships))
}

fn plot_spread(time: Array1<f64>, invariant: Array1<f64>) {
    let x_range = time.fold((f64::INFINITY, 0.0), |acc, val| {
        (val.min(acc.0), val.max(acc.1))
    });

    let y_range = invariant.fold((f64::INFINITY, 0.0), |acc, val| {
        (val.min(acc.0), val.max(acc.1))
    });

    let root_area = BitMapBackend::new("/tmp/invariant.png", (1200, 800)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let areas = root_area.split_vertically(400);

    let mut ctx = ChartBuilder::on(&areas.1)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Invariant", ("monspace", 40))
        .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    let mut hist_ctx = ChartBuilder::on(&areas.0)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(
            (y_range.0..y_range.1)
                .step((y_range.1 - y_range.0) / 100.0)
                .use_round()
                .into_segmented(),
            0..(invariant.len() / 20),
        )
        .unwrap();

    hist_ctx.configure_mesh().draw().unwrap();

    ctx.draw_series(
        AreaSeries::new(
            zip(time.into_iter(), invariant.clone().into_iter()), // The data iter
            0.0,                                                  // Baseline
            &BLUE.mix(0.2),                                       // Make the series opac
        )
        .border_style(&BLUE), // Make a brighter border
    )
    .unwrap();

    let hist = Histogram::vertical(&hist_ctx)
        .style(BLUE.mix(0.5).filled())
        .margin(0)
        .data(invariant.iter().map(|val| (*val, 1)));

    hist_ctx.draw_series(hist).unwrap();
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

    println!(
        "Bitcoin ADF check: \n{}, (gamma = {}, var = {})\n",
        adf_checks[0].criterion, adf_checks[0].gamma, adf_checks[0].regression.variance
    );
    println!(
        "Ethereum ADF check: \n{}, (gamma = {}, var = {})\n",
        adf_checks[1].criterion, adf_checks[1].gamma, adf_checks[1].regression.variance
    );

    if adf_checks[0].criterion < ADF_THRESHOLD || adf_checks[1].criterion < ADF_THRESHOLD {
        return Err("Failed integrated of order 1 checks for one of the series.".to_string());
    }

    let johansen_check = check_johansen(rebased.time.map(|t| *t as f64), rebased.series.clone())
        .expect("Failed to run Johansen test on series.");

    println!(
        "Number of estimated cointegration relationships: {}",
        johansen_check.estimated_rank
    );
    println!("Optimized lag: {}", johansen_check.lag);

    let (adjustment_vectors, cointegration_relationships) =
        factor_error_correction(johansen_check.estimated_rank, johansen_check.pi.to_owned())
            .unwrap();

    println!("Cointegration vector {:?}", cointegration_relationships);

    plot_histograms(vec![
        johansen_check.regression.residuals.flatten().to_owned()
            / johansen_check.regression.variance.sqrt(),
    ]);

    let spread = rebased
        .series
        .dot(&cointegration_relationships.t())
        .flatten()
        .to_owned()
        + johansen_check
            .regression
            .solution
            .slice(s![0, ..])
            .flatten()
            .dot(&adjustment_vectors.flatten())
        + johansen_check
            .regression
            .solution
            .slice(s![1, ..])
            .flatten()
            .dot(&adjustment_vectors.flatten())
            * rebased.time.map(|t| *t as f64);

    let mean_spread = spread.sum() / (spread.len() as f64);

    let spread_moments = (
        mean_spread,
        ((spread.clone() - mean_spread)
            .map(|val| val.powf(2.0))
            .sum()
            / (spread.len() as f64))
            .sqrt(),
    );

    println!("Spread AR mean and dev: {:?}", spread_moments);

    let z_score = (spread.clone() - spread_moments.0) / spread_moments.1;

    plot_spread(rebased.time.map(|t| *t as f64), z_score);

    Ok(())
}
