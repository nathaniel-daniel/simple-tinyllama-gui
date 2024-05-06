mod token_output_stream;

use self::token_output_stream::TokenOutputStream;
use anyhow::ensure;
use anyhow::Context;
use candle_core::Device;
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::generation::Sampling;
use candle_transformers::models::quantized_llama::ModelWeights;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;
use tracing::info;

enum Message {
    LoadModel {
        model_path: Arc<PathBuf>,
        tokenizer_path: Arc<PathBuf>,
        tx: tokio::sync::oneshot::Sender<anyhow::Result<()>>,
    },
    RunModel {
        prompt: Box<str>,
        tx: tokio::sync::oneshot::Sender<anyhow::Result<()>>,
    },
}

#[derive(Debug, Clone)]
pub struct ModelRunner {
    tx: tokio::sync::mpsc::Sender<Message>,
}

impl ModelRunner {
    pub fn new() -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        std::thread::spawn(move || task(rx));

        Self { tx }
    }

    pub async fn load_model(
        &self,
        model_path: Arc<PathBuf>,
        tokenizer_path: Arc<PathBuf>,
    ) -> anyhow::Result<()> {
        let (tx, rx) = tokio::sync::oneshot::channel();

        self.tx
            .send(Message::LoadModel {
                model_path,
                tokenizer_path,
                tx,
            })
            .await?;

        rx.await?
    }

    pub async fn run_model(&self, prompt: Box<str>) -> anyhow::Result<()> {
        let (tx, rx) = tokio::sync::oneshot::channel();

        self.tx.send(Message::RunModel { prompt, tx }).await?;

        rx.await?
    }
}

impl Default for ModelRunner {
    fn default() -> Self {
        Self::new()
    }
}

fn task(mut rx: tokio::sync::mpsc::Receiver<Message>) {
    let mut loaded_model = None;
    while let Some(message) = rx.blocking_recv() {
        match message {
            Message::LoadModel {
                model_path,
                tokenizer_path,
                tx,
            } => {
                let result = load_model(model_path, tokenizer_path).map(|new_loaded_model| {
                    loaded_model = Some(new_loaded_model);
                });
                let _ = tx.send(result).is_ok();
            }
            Message::RunModel { prompt, tx } => {
                let result = run_model(loaded_model.as_mut(), &prompt);
                let _ = tx.send(result).is_ok();
            }
        }
    }
}

fn load_model(
    model_path: Arc<PathBuf>,
    tokenizer_path: Arc<PathBuf>,
) -> anyhow::Result<LoadedModel> {
    let start_time = Instant::now();

    let device = Device::Cpu;

    info!("loading model \"{}\"", model_path.display());

    let model_path_extension = model_path.extension().context("missing extension")?;
    ensure!(model_path_extension == "gguf");

    let mut file = std::fs::File::open(&*model_path).context("failed to open model path")?;
    let model = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    let model_weights = ModelWeights::from_gguf(model, &mut file, &device)
        .context("failed to load model weights")?;

    let model_end_time = Instant::now();
    info!("loaded model in {:?}", model_end_time - start_time);

    let tokenizer = Tokenizer::from_file(&*tokenizer_path)
        .map_err(anyhow::Error::msg)
        .context("failed to load tokenizer")?;
    let tokenizer_end_time = Instant::now();
    info!(
        "loaded tokenizer in {:?}",
        tokenizer_end_time - model_end_time
    );

    let token_output_stream = TokenOutputStream::new(tokenizer);

    Ok(LoadedModel {
        model_weights,
        token_output_stream,
    })
}

struct LoadedModel {
    model_weights: ModelWeights,
    token_output_stream: TokenOutputStream,
}

fn run_model(model: Option<&mut LoadedModel>, prompt: &str) -> anyhow::Result<()> {
    let sample_len: usize = 1024;
    let temperature: f64 = 0.7;
    let top_p = Some(0.9);
    let top_k = Some(50);
    let seed = 1234;
    let repeat_penalty: f32 = 1.1;
    let repeat_last_n = 64;

    let device = Device::Cpu;

    let model = model.context("missing model")?;
    /*
    let prompt = {
        // ChatML
        // https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md
        let mut formatted = String::new();
        formatted.push_str("<|im_start|>system\n");
        formatted.push_str(prompt);
        formatted.push_str("<|im_end|>\n");
        formatted.push_str("<|im_start|>assistant\n");
        formatted
    };*/
    let prompt = {
        let mut formatted = String::new();
        formatted.push_str("<|system|>\n");
        formatted.push_str("You are a friendly assistant.</s>");
        formatted.push_str("<|user|>\n");
        formatted.push_str(prompt);
        formatted.push_str("</s>");
        formatted.push_str("<|assistant|>\n");
        formatted
    };
    
    println!("{prompt}");

    let tokens = model
        .token_output_stream
        .tokenizer()
        .encode(prompt.as_str(), true)
        .map_err(anyhow::Error::msg)?;
    let prompt_tokens = tokens.get_ids();

    let mut logits_processor = {
        let sampling = if temperature <= 0.0 {
            Sampling::ArgMax
        } else {
            match (top_k, top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
        };
        LogitsProcessor::from_sampling(seed, sampling)
    };

    let eos_token = *model
        .token_output_stream
        .tokenizer()
        .get_vocab(true)
        .get("</s>")
        .context("failed to get eos token")?;

    let mut all_tokens = prompt_tokens.to_vec();
    for index in 0..sample_len {
        let context_size = if index > 0 { 1 } else { all_tokens.len() };
        let start_pos = all_tokens.len().saturating_sub(context_size);

        let input = Tensor::new(&all_tokens[start_pos..], &device)?.unsqueeze(0)?;
        let logits = model
            .model_weights
            .forward(&input, start_pos)
            .context("failed to forward model")?;
        let logits = logits.squeeze(0)?;
        let logits = if repeat_penalty == 1.0 {
            logits
        } else {
            let start_at = all_tokens.len().saturating_sub(repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                repeat_penalty,
                &all_tokens[start_at..],
            )?
        };

        let next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);
        
        if next_token == eos_token {
            break;
        }

        dbg!(next_token);
        if let Some(t) = model.token_output_stream.next_token(next_token)? {
            dbg!(t);
        }
    }
    
    dbg!("done");

    Ok(())
}
