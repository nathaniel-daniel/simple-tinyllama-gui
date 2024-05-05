mod token_output_stream;

use self::token_output_stream::TokenOutputStream;
use anyhow::ensure;
use anyhow::Context;
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
        }
    }
}

fn load_model(
    model_path: Arc<PathBuf>,
    tokenizer_path: Arc<PathBuf>,
) -> anyhow::Result<LoadedModel> {
    let start_time = Instant::now();

    let device = candle_core::Device::Cpu;

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
