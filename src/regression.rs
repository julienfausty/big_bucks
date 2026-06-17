use ndarray::{Array, Array1, Array2};

use std::f64::consts::PI;

const MAX_LAG: usize = 20;

pub struct RegressionSolution {
    pub slopes: Array1<f64>,
    pub intercept: f64,
    pub variance: f64,
    pub covariance_matrix: Array2<f64>,
    pub residuals: Array1<f64>,
}

impl RegressionSolution {
    pub fn log_likelihood(&self) -> f64 {
        let n = self.residuals.len() as f64;
        -(n / 2.0) * ((2.0 * PI).ln() + self.variance.ln())
            - (1.0 / (2.0 * self.variance)) * self.residuals.map(|r| r.powf(2.0)).sum()
    }

    pub fn bayesian_information_criterion(&self) -> f64 {
        ((self.slopes.len() as f64) * (self.residuals.len() as f64).ln() as f64)
            - 2.0 * self.log_likelihood()
    }
}

pub fn autoregress<D>(
    lag: usize,
    timeline: &Array1<f64>,
    series: &Array<f64, D>,
) -> Result<RegressionSolution, String> {
    Err("Not implemented yet".to_string())
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

    match (0..MAX_LAG).fold(
        None,
        |acc: Option<(f64, RegressionSolution, usize)>, lag| {
            let solution = match autoregress(lag, &timeline, &series) {
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
            let gamma = solution.slopes[2];
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
