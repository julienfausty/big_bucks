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
        acc.union(&chart_set).map(|t| t.clone()).collect()
    });

    let timeline = (|| {
        let mut timeline = Vec::from_iter(timeline.into_iter());
        timeline.sort();
        timeline
    })();

    match Array::from_shape_vec(
        (series.len(), timeline.len()),
        series
            .into_iter()
            .map(|mut chart| {
                chart.sort_by(|l, r| l.0.cmp(&r.0));
                let mut i_chart = 0;
                let interp = timeline
                    .iter()
                    .map(|t| {
                        for j_chart in i_chart..(chart.len() - 1) {
                            if (chart[j_chart].0 <= *t) && (chart[j_chart + 1].0 >= *t) {
                                i_chart = j_chart;
                                let width = (chart[j_chart + 1].0 - chart[j_chart].0) as f64;
                                let weight = (*t - chart[j_chart].0) as f64 / width;

                                return chart[j_chart].1 * (1.0 - weight)
                                    + chart[j_chart + 1].1 * weight;
                            }
                        }

                        0.0
                    })
                    .collect::<Vec<f64>>();
                interp
            })
            .fold(Vec::new(), |acc, rebased| [acc, rebased].concat()),
    ) {
        Ok(values) => Ok(RebasedSeries {
            time: Array::from_vec(timeline),
            series: values.reversed_axes(),
        }),
        Err(message) => Err(format!("{}", message)),
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    use std::iter::zip;

    const EPS: f64 = 1e-12;

    #[test]
    fn test_rebase_simple() {
        let simple_data = vec![
            vec![(0, 1.0), (1, 2.0), (3, 4.0), (5, 6.0)],
            vec![(2, 3.0), (4, 5.0), (7, 8.0)],
        ];

        let rebased = rebase(simple_data);
        assert!(rebased.is_ok());

        let rebased = rebased.unwrap();

        assert!(zip(rebased.time.into_iter(), vec![2, 3, 4, 5].into_iter()).all(|(l, r)| l == r));

        assert!(
            (Array::from_shape_vec((4, 2), vec![3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0]).unwrap()
                - rebased.series)
                .map(|element: &f64| element.powf(2.0))
                .sum()
                < EPS
        );
    }

    #[test]
    fn test_rebase_single() {
        let single_data = vec![vec![(0, 1.0), (1, 2.0), (3, 4.0), (5, 6.0)]];

        let rebased = rebase(single_data);
        assert!(rebased.is_ok());

        let rebased = rebased.unwrap();

        assert!(zip(rebased.time.into_iter(), vec![0, 1, 3, 5].into_iter()).all(|(l, r)| l == r));

        assert!(
            (Array::from_shape_vec((4, 1), vec![1.0, 2.0, 4.0, 6.0]).unwrap() - rebased.series)
                .map(|element: &f64| element.powf(2.0))
                .sum()
                < EPS
        );
    }

    #[test]
    fn test_rebase_empty() {
        let empty_data = Vec::new();

        let rebased = rebase(empty_data);
        assert!(rebased.is_ok());

        let rebased = rebased.unwrap();
        assert!(rebased.time.is_empty());
        assert!(rebased.series.shape()[0] == 0);
        assert!(rebased.series.shape()[1] == 0);
    }

    #[test]
    fn test_rebase_zero_time_range() {
        let single_data = vec![vec![(1, 1.0), (1, 2.0)]];

        assert!(rebase(single_data).is_err());
    }
}
