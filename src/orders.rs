use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Order {
    Open((String, f64)),
    Close(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Confirmation(pub Order);
