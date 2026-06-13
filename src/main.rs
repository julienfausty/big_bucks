pub mod fetch;
use fetch::{MarketChart, MarketChartQuery, query_market_chart};

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

    println!("Bitcoin length {:?}", bitcoin_chart.prices.len());
    println!("Ethereum length {:?}", ethereum_chart.prices.len());

    Ok(())
}
