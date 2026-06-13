use serde::Deserialize;

#[derive(Debug)]
pub struct MarketChartQuery {
    pub coin: String,
    pub currency: String,
    pub since: String,
    pub interval: String,
    pub precision: String,
}

#[derive(Debug, Deserialize)]
pub struct MarketChart {
    pub prices: Vec<(u64, f64)>,
    pub total_volumes: Vec<(u64, f64)>,
    pub market_caps: Vec<(u64, f64)>,
}

pub async fn query_market_chart(query: MarketChartQuery) -> Result<MarketChart, String> {
    let (coin, params) = (
        query.coin,
        [
            ("vs_currency", query.currency),
            ("days", query.since),
            ("interval", query.interval),
            ("precision", query.precision),
        ],
    );

    let url = reqwest::Url::parse_with_params(
        &format!("https://api.coingecko.com/api/v3/coins/{coin}/market_chart"),
        &params,
    )
    .expect("Could not parse url correctly.");
    match reqwest::Client::new()
        .get(url)
        .header("User-Agent", "big_bucks 0.1")
        .send()
        .await
    {
        Ok(response) => Ok(
            match serde_json::from_str(&match response.text().await {
                Ok(body) => body,
                Err(message) => return Err(format!("{}", message)),
            }) {
                Ok(chart) => chart,
                Err(message) => return Err(format!("{}", message)),
            },
        ),
        Err(message) => Err(format!("{}", message)),
    }
}
