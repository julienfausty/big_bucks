use ndarray::{Array1, Array2, ArrayView1, ArrayView2, par_azip, s};
use ndarray_linalg::{FactorizeInto, Inverse};

use std::f64::consts::PI;

const MAX_LAG: usize = 20;

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
    par_azip!((d in differences.view_mut(), s0 in series.slice(s![1.., ..]), s1 in series.slice(s![..-1, ..])) *d = s1 - s0);

    let design_shape = (
        differences.shape()[0] - (lag + 1),
        (3 + lag) * series.shape()[1],
    );
    let mut design_matrix = Array2::<f64>::zeros(design_shape.clone());

    *design_matrix.slice_mut(s![.., 0]) += 1.0;

    par_azip!((dm in design_matrix.slice_mut(s![.., 1]), t in timeline.slice(s![..design_shape.0])) *dm += t);

    par_azip!((dm in design_matrix.slice_mut(s![.., 2..2+series.shape()[1]]), val in series.slice(s![1..design_shape.0 + 1, ..])) *dm += val);

    for i_diff in 0..lag {
        let design_index_range = (
            (3 + i_diff) * series.shape()[1],
            (4 + i_diff) * series.shape()[1],
        );
        par_azip!((dm in design_matrix.slice_mut(s![.., design_index_range.0..design_index_range.1]), diff in differences.slice(s![(1+i_diff)..(design_shape.0 + i_diff + 1), ..])) *dm += diff);
    }

    let observations = design_matrix
        .t()
        .dot(&differences.slice(s![..design_shape.0, ..]));

    let inv_gram_matrix = match (design_matrix.t().dot(&design_matrix)).factorize_into() {
        Ok(factorized) => match factorized.inv() {
            Ok(inverse) => inverse,
            Err(message) => {
                return Err(format!(
                    "Failed to inverse matrix after factorization in autoregression:\n{}",
                    message
                ));
            }
        },
        Err(message) => {
            return Err(format!(
                "Failed LU factorization in autoregression:\n{}",
                message
            ));
        }
    };

    let solution_vec = inv_gram_matrix.dot(&observations).to_owned();

    let residuals = design_matrix.dot(&solution_vec) - differences.slice(s![..design_shape.0, ..]);

    let variance = residuals
        .rows()
        .into_iter()
        .map(|row| row.into_iter().map(|val| val.powf(2.0)).sum::<f64>())
        .sum::<f64>()
        / (design_shape.0 as f64);

    Ok(RegressionSolution {
        solution: solution_vec,
        covariance_matrix: variance.clone() * inv_gram_matrix,
        variance,
        residuals,
    })
}

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
