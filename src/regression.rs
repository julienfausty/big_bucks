use ndarray::{Array1, Array2, ArrayView1, ArrayView2, par_azip, s};
use ndarray_linalg::SVDInto;
use statrs::distribution::{ChiSquared, ContinuousCDF};

use std::f64::consts::PI;
use std::iter::zip;

const MAX_LAG: usize = 20;
const EPS: f64 = 1e-16;

#[derive(Debug)]
pub struct RegressionSolution {
    pub solution: Array2<f64>,
    pub variance: f64,
    pub covariance_matrix: Array2<f64>,
    pub residuals: Array2<f64>,
}

impl RegressionSolution {
    pub fn log_likelihood(&self) -> f64 {
        let n = self.residuals.len() as f64;
        -(n / 2.0) * ((2.0 * PI).ln() + self.variance.ln())
            - (1.0 / (2.0 * self.variance)) * self.residuals.map(|r| r.powf(2.0)).sum()
    }

    pub fn bayesian_information_criterion(&self) -> f64 {
        ((self.solution.len() as f64) * (self.residuals.len() as f64).ln() as f64)
            - 2.0 * self.log_likelihood()
    }
}

pub fn autoregress(
    lag: usize,
    timeline: ArrayView1<f64>,
    series: ArrayView2<f64>,
) -> Result<RegressionSolution, String> {
    if series.shape()[0] != timeline.len() {
        return Err("Timeline and series do not have same length.".to_string());
    }

    if series.shape()[0] <= lag + 1 {
        return Err(format!(
            "Lag {lag} is too large for size of series {}",
            series.shape()[0]
        ));
    }

    let mut differences = Array2::<f64>::zeros((series.shape()[0] - 1, series.shape()[1]));
    par_azip!((d in differences.view_mut(), s0 in series.slice(s![..-1, ..]), s1 in series.slice(s![1.., ..])) *d = s1 - s0);

    let design_shape = (
        differences.shape()[0] - (lag + 1),
        2 + (1 + lag) * series.shape()[1],
    );
    let mut design_matrix = Array2::<f64>::zeros(design_shape.clone());

    *design_matrix.slice_mut(s![.., 0]) += 1.0;

    par_azip!((dm in design_matrix.slice_mut(s![.., 1]), t in timeline.slice(s![timeline.len()-design_shape.0..])) *dm += t);

    par_azip!((dm in design_matrix.slice_mut(s![.., 2..2+series.shape()[1]]), val in series.slice(s![(series.shape()[0] - design_shape.0 - 1)..(series.shape()[0]-1), ..])) *dm += val);

    for i_diff in 0..lag {
        let design_index_range = (
            2 + (1 + i_diff) * series.shape()[1],
            2 + (2 + i_diff) * series.shape()[1],
        );
        let diff_slice_range = (
            differences.shape()[0] - i_diff - 1 - design_shape.0,
            differences.shape()[0] - i_diff - 1,
        );
        par_azip!((dm in design_matrix.slice_mut(s![.., design_index_range.0..design_index_range.1]), diff in differences.slice(s![diff_slice_range.0..diff_slice_range.1, ..])) *dm += diff);
    }

    let observations = design_matrix
        .t()
        .dot(&differences.slice(s![(differences.shape()[0] - design_shape.0).., ..]));

    let inv_gram_matrix = match (design_matrix.t().dot(&design_matrix)).svd_into(true, true) {
        Ok((u, diag, vt)) => {
            let inv_diag = diag.map(|elem| if *elem > EPS { 1.0 / *elem } else { *elem });
            vt.unwrap()
                .t()
                .dot(&Array2::from_diag(&inv_diag))
                .dot(&u.unwrap().t())
        }
        Err(message) => {
            return Err(format!(
                "Failed SVD factorization in autoregression:\n{}",
                message
            ));
        }
    };

    let solution_vec = inv_gram_matrix.dot(&observations).to_owned();

    let residuals = design_matrix.dot(&solution_vec)
        - differences.slice(s![(differences.shape()[0] - design_shape.0).., ..]);

    let variance = residuals
        .rows()
        .into_iter()
        .map(|row| row.into_iter().map(|val| val.powf(2.0)).sum::<f64>())
        .sum::<f64>()
        / (residuals.len() as f64);

    Ok(RegressionSolution {
        solution: solution_vec,
        covariance_matrix: variance.clone() * inv_gram_matrix,
        variance,
        residuals,
    })
}

#[derive(Debug)]
pub struct ADFCheck {
    pub criterion: f64,
    pub gamma: f64,
    pub bic: f64,
    pub lag: usize,
    pub regression: RegressionSolution,
}

pub fn check_augmented_dicky_fuller<S: IntoIterator<Item = f64>>(
    timeline: S,
    series: S,
) -> Result<ADFCheck, String> {
    let timeline = Array1::from_iter(timeline.into_iter());

    let series = Array1::from_iter(series.into_iter());
    let n_samples = series.len();
    let series = match series.to_shape((n_samples, 1)) {
        Ok(series) => series.to_owned(),
        Err(message) => return Err(format!("{}", message)),
    };

    if timeline.len() != n_samples {
        return Err(format!(
            "Passed different length timeline ({}) and series data ({}) during Dicky-Fuller check.",
            timeline.len(),
            n_samples
        ));
    }

    match (0..MAX_LAG).fold(
        None,
        |acc: Option<(f64, RegressionSolution, usize)>, lag| {
            let solution = match autoregress(lag, timeline.view(), series.view()) {
                Ok(sol) => sol,
                Err(_) => return acc,
            };

            let bic = solution.bayesian_information_criterion();
            match acc {
                Some(tuple) => {
                    if bic < tuple.0 {
                        Some((bic, solution, lag))
                    } else {
                        Some(tuple)
                    }
                }
                None => Some((bic, solution, lag)),
            }
        },
    ) {
        Some(sol) => {
            let (bic, solution, lag) = sol;
            let gamma = solution.solution[[2, 0]];
            let criterion = gamma / solution.covariance_matrix[[2, 2]].sqrt();
            Ok(ADFCheck {
                criterion,
                gamma,
                bic,
                lag,
                regression: solution,
            })
        }
        None => Err("No lag value generated a regression solution.".to_string()),
    }
}

#[derive(Debug)]
pub struct JohansenCheck {
    pub trace_statistic: f64,
    pub max_eigenvalue_statistic: f64,
    pub estimated_rank: usize,
    pub pi: Array2<f64>,
    pub bic: f64,
    pub lag: usize,
    pub regression: RegressionSolution,
}

impl JohansenCheck {
    pub fn factor_error_correction(&self) -> Result<(Array2<f64>, Array2<f64>), String> {
        let rank = self.estimated_rank;
        let linear_map = self.pi.clone();

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
}

pub fn check_johansen(timeline: Array1<f64>, series: Array2<f64>) -> Result<JohansenCheck, String> {
    let n_samples = series.shape()[0];

    if timeline.len() != n_samples {
        return Err(format!(
            "Passed different length timeline ({}) and series data ({}) during Johansen check.",
            timeline.len(),
            n_samples
        ));
    }

    match (0..MAX_LAG).fold(
        None,
        |acc: Option<(f64, RegressionSolution, usize)>, lag| {
            let solution = match autoregress(lag, timeline.view(), series.view()) {
                Ok(sol) => sol,
                Err(_) => return acc,
            };

            let bic = solution.bayesian_information_criterion();
            match acc {
                Some(tuple) => {
                    if bic < tuple.0 {
                        Some((bic, solution, lag))
                    } else {
                        Some(tuple)
                    }
                }
                None => Some((bic, solution, lag)),
            }
        },
    ) {
        Some(sol) => {
            let (bic, solution, lag) = sol;
            let pi = solution
                .solution
                .slice(s![2..(2 + series.shape()[1]), ..])
                .t()
                .to_owned();
            let mut svd: Vec<f64> = match pi.clone().svd_into(false, false) {
                Ok((_, svd, _)) => svd.into_iter().collect::<Vec<_>>(),
                Err(message) => {
                    return Err(format!(
                        "Failed the singular value decomposition in the Johansen test result:\n {:?}",
                        message
                    ));
                }
            };
            svd.sort_by(|l, r| l.total_cmp(r));
            let svd: Vec<f64> = svd.into_iter().rev().collect();

            let n_samples = solution.residuals.shape()[0] as f64;
            let mut hypothesis = (svd.len(), 0.0, 0.0);
            for i_rank in 0..svd.len() {
                let max = -n_samples * (1.0 - svd[i_rank]).ln();
                let trace = -n_samples
                    * (&svd[i_rank..])
                        .iter()
                        .map(|val| (1.0 - *val).ln())
                        .sum::<f64>();
                let n_freedom: f64 = (svd.len() - i_rank) as f64;
                let critical_trace_val = match ChiSquared::new(2.0 * n_freedom.powf(2.0)) {
                    Ok(distribution) => {
                        (0.85 - 0.58 / (2.0 * n_freedom.powf(2.0))) * distribution.inverse_cdf(0.99)
                    }
                    Err(message) => {
                        return Err(format!(
                            "Failed to compute critical value in Johansen test:\n{:?}",
                            message
                        ));
                    }
                };

                if trace < critical_trace_val {
                    hypothesis = (i_rank, max, trace);
                    break;
                }
            }
            Ok(JohansenCheck {
                estimated_rank: hypothesis.0,
                max_eigenvalue_statistic: hypothesis.1,
                trace_statistic: hypothesis.2,
                pi,
                bic,
                lag,
                regression: solution,
            })
        }
        None => Err("No lag value generated a regression solution.".to_string()),
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    use ndarray::Array;
    use ndarray_linalg::SVDInto;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Normal;

    #[test]
    fn linear_autoregression() {
        let timeline = Array1::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4]);
        let series = Array2::from_shape_vec((5, 1), vec![1.0, 1.1, 1.2, 1.3, 1.4]).unwrap();

        let regression_0 = autoregress(0, timeline.view(), series.view());

        assert!(regression_0.is_ok());

        let regression_0 = regression_0.unwrap();
        assert!(regression_0.variance < EPS);

        assert!(regression_0.residuals.into_iter().sum::<f64>().powf(2.0) < EPS);

        let regression_1 = autoregress(1, timeline.view(), series.view());

        assert!(regression_1.is_ok());

        let regression_1 = regression_1.unwrap();
        assert!(regression_1.variance < EPS);

        assert!(regression_1.residuals.into_iter().sum::<f64>().powf(2.0) < EPS);
    }

    #[test]
    fn lag_too_large() {
        let timeline = Array1::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4]);
        let series = Array2::from_shape_vec((5, 1), vec![1.0, 1.1, 1.2, 1.3, 1.4]).unwrap();

        let regression = autoregress(3, timeline.view(), series.view());

        assert!(regression.is_ok());

        let regression = autoregress(4, timeline.view(), series.view());

        assert!(regression.is_err());
    }

    #[test]
    fn shape_mismatch() {
        let timeline = Array1::from_vec(vec![0.0, 0.1, 0.2, 0.3]);
        let series = Array2::from_shape_vec((5, 1), vec![1.0, 1.1, 1.2, 1.3, 1.4]).unwrap();

        let regression = autoregress(0, timeline.view(), series.view());

        assert!(regression.is_err());
    }

    #[test]
    fn test_random_autoregression() {
        let length = 100000;
        let timeline = Array1::<f64>::linspace(0.0, 1.0, length);
        let series = Array::random((length, 1), Normal::new(0.0, 1.0).unwrap());

        let autoregression = autoregress(0, timeline.view(), series.view());

        assert!(autoregression.is_ok());

        let autoregression = autoregression.unwrap();

        let expected_tolerance = 1.0 / (length as f64).sqrt();

        assert!((1.0 - autoregression.variance).powf(2.0) < expected_tolerance);
        assert!(autoregression.solution[[0, 0]].powf(2.0) < expected_tolerance);
        assert!(autoregression.solution[[1, 0]].powf(2.0) < expected_tolerance);
        assert!((1.0 + autoregression.solution[[2, 0]]).powf(2.0) < expected_tolerance);

        let autoregression = autoregress(1, timeline.view(), series.view());

        assert!(autoregression.is_ok());

        let autoregression = autoregression.unwrap();

        let expected_tolerance = 1.0 / (length as f64).sqrt();

        assert!((1.0 - autoregression.variance).powf(2.0) < expected_tolerance);
        assert!(autoregression.solution[[0, 0]].powf(2.0) < expected_tolerance);
        assert!(autoregression.solution[[1, 0]].powf(2.0) < expected_tolerance);
        assert!((1.0 + autoregression.solution[[2, 0]]).powf(2.0) < expected_tolerance);
        assert!(autoregression.solution[[3, 0]].powf(2.0) < expected_tolerance);

        let autoregression = autoregress(7, timeline.view(), series.view());

        assert!(autoregression.is_ok());

        let autoregression = autoregression.unwrap();

        let expected_tolerance = 1.0 / (length as f64).sqrt();

        assert!((1.0 - autoregression.variance).powf(2.0) < expected_tolerance);
        assert!(autoregression.solution[[0, 0]].powf(2.0) < expected_tolerance);
        assert!(autoregression.solution[[1, 0]].powf(2.0) < expected_tolerance);
        assert!((1.0 + autoregression.solution[[2, 0]]).powf(2.0) < expected_tolerance);
        assert!(autoregression.solution[[3, 0]].powf(2.0) < expected_tolerance);
        assert!(autoregression.solution[[4, 0]].powf(2.0) < expected_tolerance);
        assert!(autoregression.solution[[5, 0]].powf(2.0) < expected_tolerance);
        assert!(autoregression.solution[[6, 0]].powf(2.0) < expected_tolerance);
    }

    #[test]
    fn test_noisy_linear_autoregression() {
        let length = 100000;
        let timeline = Array1::<f64>::linspace(0.0, 1.0, length);
        let series: Array2<f64> = Array::random((length, 1), Normal::new(0.0, 1.0).unwrap())
            + 0.1 * timeline.clone().to_shape((length, 1)).unwrap()
            + 3.14;

        let autoregression = autoregress(0, timeline.view(), series.view());

        assert!(autoregression.is_ok());

        let autoregression = autoregression.unwrap();

        let expected_tolerance = 1.0 / (length as f64).sqrt();

        assert!((1.0 - autoregression.variance).powf(2.0) < expected_tolerance);
        assert!((3.14 - autoregression.solution[[0, 0]]).powf(2.0) < expected_tolerance);
        assert!((0.1 - autoregression.solution[[1, 0]]).powf(2.0) < expected_tolerance);
        assert!((1.0 + autoregression.solution[[2, 0]]).powf(2.0) < expected_tolerance);

        let autoregression = autoregress(1, timeline.view(), series.view());

        assert!(autoregression.is_ok());

        let autoregression = autoregression.unwrap();

        assert!((1.0 - autoregression.variance).powf(2.0) < expected_tolerance);
        assert!((3.14 - autoregression.solution[[0, 0]]).powf(2.0) < expected_tolerance);
        assert!((0.1 - autoregression.solution[[1, 0]]).powf(2.0) < expected_tolerance);
        assert!((1.0 + autoregression.solution[[2, 0]]).powf(2.0) < expected_tolerance);
        assert!(autoregression.solution[[3, 0]].powf(2.0) < expected_tolerance);
    }

    #[test]
    fn test_random_walk_autoregression() {
        let length = 100000;
        let timeline = Array1::<f64>::linspace(0.0, 1.0, length);
        let mut series: Array2<f64> = Array::random((length, 1), Normal::new(0.0, 1.0).unwrap())
            + 0.1 * timeline.clone().to_shape((length, 1)).unwrap()
            + 3.14;

        for i_series in 0..(length - 1) {
            series[[i_series + 1, 0]] += series[[i_series, 0]]
        }

        let autoregression = autoregress(0, timeline.view(), series.view());

        assert!(autoregression.is_ok());

        let autoregression = autoregression.unwrap();

        let expected_tolerance = 1.0 / (length as f64).sqrt();

        assert!((1.0 - autoregression.variance).powf(2.0) < expected_tolerance);
        assert!((3.14 - autoregression.solution[[0, 0]]).powf(2.0) < expected_tolerance);
        assert!(autoregression.solution[[2, 0]].powf(2.0) < expected_tolerance);

        let autoregression = autoregress(1, timeline.view(), series.view());

        assert!(autoregression.is_ok());

        let autoregression = autoregression.unwrap();

        assert!((1.0 - autoregression.variance).powf(2.0) < expected_tolerance);
        assert!((3.14 - autoregression.solution[[0, 0]]).powf(2.0) < expected_tolerance);
        assert!(autoregression.solution[[2, 0]].powf(2.0) < expected_tolerance);
        assert!(autoregression.solution[[3, 0]].powf(2.0) < expected_tolerance);
    }

    #[test]
    fn test_multi_dimensional_autoregression() {
        let length = 100000;
        let timeline = Array1::<f64>::linspace(0.0, 1.0, length);
        let mut series: Array2<f64> = Array::random((length, 3), Normal::new(0.0, 1.0).unwrap())
            + timeline
                .clone()
                .to_shape((length, 1))
                .unwrap()
                .dot(&Array2::from_shape_vec((1, 3), vec![0.1, 0.2, 0.3]).unwrap())
            + 3.14;

        for i_series in 0..(length - 1) {
            series[[i_series + 1, 0]] += series[[i_series, 0]];
            series[[i_series + 1, 1]] += 0.2 * series[[i_series, 0]];
            series[[i_series + 1, 2]] += 1.3 * series[[i_series, 0]];
        }

        let autoregression = autoregress(3, timeline.view(), series.view());

        assert!(autoregression.is_ok());

        let autoregression = autoregression.unwrap();

        let expected_tolerance = 1.0 / (length as f64).sqrt();

        assert!((1.0 - autoregression.variance).powf(2.0) < expected_tolerance);
        assert!(
            (3.14 - autoregression.solution.slice(s![0, ..]).sum() / 3.0).powf(2.0)
                < expected_tolerance
        );

        let (_, svd, _) = autoregression
            .solution
            .slice(s![2.., ..])
            .to_owned()
            .svd_into(false, false)
            .unwrap();

        assert_eq!(
            svd.fold(0, |rank: usize, val: &f64| {
                if val.powf(2.0) > expected_tolerance {
                    rank + 1
                } else {
                    rank
                }
            }),
            2
        );

        assert!(autoregression.solution.slice(s![5.., ..]).sum().powf(2.0) < expected_tolerance);
    }

    #[test]
    fn random_adf() {
        let length = 100000;
        let timeline = Array1::<f64>::linspace(0.0, 1.0, length);
        let series = Array::random(length, Normal::new(0.0, 1.0).unwrap());

        let checked = check_augmented_dicky_fuller(timeline, series);

        assert!(checked.is_ok());

        let checked = checked.unwrap();

        assert!(checked.criterion < -3.96);

        let expected_tolerance = 1.0 / (length as f64).sqrt();
        assert!((1.0 + checked.gamma).powf(2.0) < expected_tolerance);
        assert_eq!(checked.lag, 0);
    }

    #[test]
    fn test_noisy_linear_adf() {
        let length = 100000;
        let timeline = Array1::<f64>::linspace(0.0, 1.0, length);
        let series: Array1<f64> =
            Array::random(length, Normal::new(0.0, 1.0).unwrap()) + 0.1 * timeline.clone() + 3.14;

        let checked = check_augmented_dicky_fuller(timeline, series);

        assert!(checked.is_ok());

        let checked = checked.unwrap();

        assert!(checked.criterion < -3.96);

        let expected_tolerance = 1.0 / (length as f64).sqrt();
        assert!((1.0 + checked.gamma).powf(2.0) < expected_tolerance);
        assert_eq!(checked.lag, 0);
    }

    #[test]
    fn test_random_walk_adf() {
        let length = 100000;
        let timeline = Array1::<f64>::linspace(0.0, 1.0, length);
        let mut series: Array1<f64> =
            Array::random(length, Normal::new(0.0, 1.0).unwrap()) + 0.1 * timeline.clone() + 3.14;
        for i_series in 0..(length - 1) {
            series[[i_series + 1]] += series[[i_series]]
        }

        let checked = check_augmented_dicky_fuller(timeline, series);

        assert!(checked.is_ok());

        let checked = checked.unwrap();

        assert!(checked.criterion > -3.96);
        let expected_tolerance = 1.0 / (length as f64).sqrt();
        assert!(checked.gamma.powf(2.0) < expected_tolerance);
        assert_eq!(checked.lag, 0);
    }

    #[test]
    fn test_johansen_cointegrated_2() {
        let length = 100000;
        let timeline = Array1::<f64>::linspace(0.0, 1.0, length);
        let mut series: Array2<f64> = Array::random((length, 3), Normal::new(0.0, 1.0).unwrap())
            + timeline
                .clone()
                .to_shape((length, 1))
                .unwrap()
                .dot(&Array2::from_shape_vec((1, 3), vec![1.1, 0.2, 5.3]).unwrap())
            + 3.14;

        for i_series in 0..(length - 1) {
            series[[i_series + 1, 0]] += series[[i_series, 0]];
            series[[i_series + 1, 1]] += 1.3 * series[[i_series, 0]];
            series[[i_series + 1, 2]] += 0.2 * series[[i_series, 0]];
        }

        let checked = check_johansen(timeline, series);

        assert!(checked.is_ok());

        let checked = checked.unwrap();

        assert!(checked.estimated_rank == 2);
    }
}
