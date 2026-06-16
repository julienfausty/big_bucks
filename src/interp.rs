use ndarray::{Array, Array1, Array2};

use std::collections::HashSet;

pub struct RebasedSeries {
    pub time: Array1<u64>,
    pub series: Array2<f64>,
}

pub fn rebase(series: Vec<Vec<(u64, f64)>>) -> Result<RebasedSeries, String> {
    let t_range = series.iter().fold((u64::MIN, u64::MAX), |acc, chart| {
        let local_range = chart
            .iter()
            .fold((u64::MAX, u64::MIN), |acc_inner, (t, _)| {
                (*t.min(&acc_inner.0), *t.max(&acc_inner.1))
            });
        (acc.0.max(local_range.0), acc.1.min(local_range.1))
    });

    if t_range.0 == t_range.1 {
        return Err("Rebased time range is zero width.".to_string());
    }

    let timeline = series.iter().fold(HashSet::new(), |acc, chart| {
        let chart_set = HashSet::from_iter(
            chart
                .iter()
                .map(|(t, _)| (*t).clone())
                .filter(|t| *t >= t_range.0 && *t <= t_range.1),
        );
        acc.intersection(&chart_set).map(|t| t.clone()).collect()
    });

    let timeline = (|| {
        let mut timeline = Vec::from_iter(timeline.into_iter());
        timeline.sort();
        timeline
    })();

    match Array::from_shape_vec(
        (timeline.len(), series.len()),
        series
            .into_iter()
            .map(|mut chart| {
                chart.sort_by(|l, r| l.0.cmp(&r.0));
                let mut i_chart = 0;
                timeline
                    .iter()
                    .map(|t| {
                        for j_chart in i_chart..(chart.len() - 1) {
                            if (chart[j_chart].0 <= *t) && (chart[j_chart + 1].0 > *t) {
                                i_chart = j_chart;
                                let width = (chart[j_chart + 1].0 - chart[j_chart].0) as f64;
                                return chart[j_chart].1 * ((*t - chart[j_chart].0) as f64 / width)
                                    + chart[j_chart + 1].1
                                        * ((chart[j_chart + 1].0 - *t) as f64 / width);
                            }
                        }

                        0.0
                    })
                    .collect::<Vec<f64>>()
            })
            .fold(Vec::new(), |acc, rebased| [acc, rebased].concat()),
    ) {
        Ok(values) => Ok(RebasedSeries {
            time: Array::from_vec(timeline),
            series: values,
        }),
        Err(message) => Err(format!("{}", message)),
    }
}
