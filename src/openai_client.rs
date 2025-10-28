use pyo3::prelude::*;
use pyo3::exceptions::PyException;
use pyo3::types::{PyDict, PyList};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Semaphore;

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    id: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: Message,
    finish_reason: String,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[pyclass]
pub struct OpenAIClient {
    client: Arc<Client>,
    api_key: Arc<String>,
    base_url: Arc<String>,
    semaphore: Arc<Semaphore>,
}

#[pymethods]
impl OpenAIClient {
    #[new]
    #[pyo3(signature = (api_key, max_concurrent_requests=50, timeout_secs=30))]
    fn new(
        api_key: String,
        max_concurrent_requests: usize,
        timeout_secs: u64,
    ) -> PyResult<Self> {
        let client = Client::builder()
            // Increase connection pool size
            .pool_max_idle_per_host(200)
            .pool_idle_timeout(std::time::Duration::from_secs(90))
            // Enable HTTP/2 for multiplexing
            .http2_prior_knowledge()
            // Add timeouts to prevent hanging
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .connect_timeout(std::time::Duration::from_secs(10))
            // Enable TCP keepalive
            .tcp_keepalive(std::time::Duration::from_secs(60))
            // Enable response compression
            .gzip(true)
            .brotli(true)
            // Disable unnecessary redirects for API calls
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| PyException::new_err(format!("Failed to create client: {}", e)))?;

        Ok(OpenAIClient {
            client: Arc::new(client),
            api_key: Arc::new(api_key),
            base_url: Arc::new("https://api.openai.com/v1".to_string()),
            semaphore: Arc::new(Semaphore::new(max_concurrent_requests)),
        })
    }

    fn chat_completion<'py>(
        &self,
        py: Python<'py>,
        model: String,
        messages: Vec<(String, String)>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> PyResult<&'py PyAny> {
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let base_url = self.base_url.clone();
        let semaphore = self.semaphore.clone();

        let messages: Vec<Message> = messages
            .into_iter()
            .map(|(role, content)| Message { role, content })
            .collect();

        pyo3_asyncio::tokio::future_into_py(py, async move {
            let _permit = semaphore.acquire().await
                .map_err(|e| PyException::new_err(format!("Semaphore error: {}", e)))?;

            let request = ChatRequest {
                model,
                messages,
                temperature,
                max_tokens,
                stream: Some(false),
            };

            let response = client
                .post(format!("{}/chat/completions", base_url.as_ref()))
                .header("Authorization", format!("Bearer {}", api_key.as_ref()))
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await
                .map_err(|e| PyException::new_err(format!("Request failed: {}", e)))?;

            let status = response.status();
            if !status.is_success() {
                let error_text = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Unknown error".to_string());
                return Err(PyException::new_err(format!(
                    "API error {}: {}",
                    status, error_text
                )));
            }

            let chat_response: ChatResponse = response
                .json()
                .await
                .map_err(|e| PyException::new_err(format!("Failed to parse response: {}", e)))?;

            Python::with_gil(|py| {
                let dict = PyDict::new(py);
                dict.set_item("id", chat_response.id)?;
                dict.set_item(
                    "content",
                    chat_response.choices[0].message.content.clone(),
                )?;
                dict.set_item("finish_reason", &chat_response.choices[0].finish_reason)?;
                dict.set_item("prompt_tokens", chat_response.usage.prompt_tokens)?;
                dict.set_item("completion_tokens", chat_response.usage.completion_tokens)?;
                dict.set_item("total_tokens", chat_response.usage.total_tokens)?;
                Ok(dict.to_object(py))
            })
        })
    }

    fn batch_chat_completion<'py>(
        &self,
        py: Python<'py>,
        model: String,
        requests: Vec<Vec<(String, String)>>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> PyResult<&'py PyAny> {
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let base_url = self.base_url.clone();
        let semaphore = self.semaphore.clone();

        pyo3_asyncio::tokio::future_into_py(py, async move {
            // Pre-allocate results vector
            let request_count = requests.len();
            let mut tasks = Vec::with_capacity(request_count);

            for messages in requests {
                let client = client.clone();
                let api_key = api_key.clone();
                let base_url = base_url.clone();
                let model = model.clone();
                let semaphore = semaphore.clone();

                let messages: Vec<Message> = messages
                    .into_iter()
                    .map(|(role, content)| Message { role, content })
                    .collect();

                let task = tokio::spawn(async move {
                    // Acquire semaphore permit before making request
                    let _permit = semaphore.acquire().await?;

                    let request = ChatRequest {
                        model,
                        messages,
                        temperature,
                        max_tokens,
                        stream: Some(false),
                    };

                    let response = client
                        .post(format!("{}/chat/completions", base_url.as_ref()))
                        .header("Authorization", format!("Bearer {}", api_key.as_ref()))
                        .header("Content-Type", "application/json")
                        .json(&request)
                        .send()
                        .await?;

                    let status = response.status();
                    if !status.is_success() {
                        let error_text = response
                            .text()
                            .await
                            .unwrap_or_else(|_| "Unknown error".to_string());
                        return Err(anyhow::anyhow!("API error {}: {}", status, error_text));
                    }

                    let chat_response: ChatResponse = response.json().await?;
                    Ok::<ChatResponse, anyhow::Error>(chat_response)
                });

                tasks.push(task);
            }

            // Use join_all for better performance
            let results = futures::future::join_all(tasks).await;

            Python::with_gil(|py| {
                let list = PyList::empty(py);
                for result in results {
                    match result {
                        Ok(Ok(chat_response)) => {
                            let dict = PyDict::new(py);
                            dict.set_item("id", chat_response.id)?;
                            dict.set_item(
                                "content",
                                chat_response.choices[0].message.content.clone(),
                            )?;
                            dict.set_item("finish_reason", &chat_response.choices[0].finish_reason)?;
                            dict.set_item("prompt_tokens", chat_response.usage.prompt_tokens)?;
                            dict.set_item(
                                "completion_tokens",
                                chat_response.usage.completion_tokens,
                            )?;
                            dict.set_item("total_tokens", chat_response.usage.total_tokens)?;
                            list.append(dict)?;
                        }
                        Ok(Err(e)) => {
                            return Err(PyException::new_err(format!("Request failed: {}", e)));
                        }
                        Err(e) => {
                            return Err(PyException::new_err(format!("Task failed: {}", e)));
                        }
                    }
                }
                Ok(list.to_object(py))
            })
        })
    }

    /// Set custom base URL (useful for proxies or custom endpoints)
    fn set_base_url(&mut self, base_url: String) {
        self.base_url = Arc::new(base_url);
    }

    /// Get current configuration stats
    fn get_stats(&self) -> PyResult<Py<PyDict>> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("max_concurrent_requests", self.semaphore.available_permits())?;
            dict.set_item("base_url", self.base_url.as_ref())?;
            Ok(dict.into())
        })
    }
}

// Module initialization
#[pymodule]
fn openai_client_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<OpenAIClient>()?;
    Ok(())
}