use std::fs;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Prediction {
    label: String,
    score: f32,
}

fn main() {
    let model_path = "models/exported/classifier.onnx";
    println!("Loading model from: {}", model_path);
}
