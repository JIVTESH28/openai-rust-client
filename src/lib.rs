#![allow(non_local_definitions)]
use pyo3::prelude::*;

mod openai_client;
mod anthropic_client;

use openai_client::OpenAIClient;
use anthropic_client::AnthropicClient;

#[pymodule]
fn rustc_ai_framework(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<OpenAIClient>()?;
    m.add_class::<AnthropicClient>()?;
    Ok(())
}