use anyhow::ensure;
use anyhow::Context;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::Special;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use std::future::Future;
use std::num::NonZero;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;
use tracing::info;

type ProgressHandler = Box<
    dyn FnMut(String) -> Pin<Box<dyn Future<Output = ()> + Send + 'static>> + Send + Sync + 'static,
>;

enum Message {
    LoadModel {
        model_path: Arc<PathBuf>,
        tokenizer_path: Arc<PathBuf>,
        tx: tokio::sync::oneshot::Sender<anyhow::Result<()>>,
    },
    RunModel {
        prompt: Box<str>,
        progress_handler: ProgressHandler,
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

    pub async fn run_model<F, FU>(
        &self,
        prompt: Box<str>,
        mut progress_handler: F,
    ) -> anyhow::Result<()>
    where
        F: FnMut(String) -> FU + Send + Sync + 'static,
        FU: Future<Output = ()> + Send + 'static,
    {
        let (tx, rx) = tokio::sync::oneshot::channel();

        self.tx
            .send(Message::RunModel {
                prompt,
                progress_handler: Box::new(move |token| Box::pin((progress_handler)(token))),
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
            Message::RunModel {
                prompt,
                progress_handler,
                tx,
            } => {
                let result = run_model(loaded_model.as_mut(), &prompt, progress_handler);
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

    let backend = LlamaBackend::init().context("failed to init llama-cpp")?;
    let model_params = LlamaModelParams::default();

    info!("loading model \"{}\"", model_path.display());

    let model_path_extension = model_path.extension().context("missing extension")?;
    ensure!(model_path_extension == "gguf");

    let model = LlamaModel::load_from_file(&backend, &*model_path, &model_params)
        .context("failed to load model")?;

    let model_end_time = Instant::now();
    info!("loaded model in {:?}", model_end_time - start_time);

    Ok(LoadedModel { backend, model })
}

struct LoadedModel {
    backend: LlamaBackend,
    model: LlamaModel,
}

fn run_model(
    loaded_model: Option<&mut LoadedModel>,
    prompt: &str,
    mut progress_handler: ProgressHandler,
) -> anyhow::Result<()> {
    let sample_len: usize = 256;
    let temperature: f64 = 0.7;
    let top_p = Some(0.95);
    let top_k = Some(50);
    let seed = 1234;
    let repeat_penalty: f32 = 1.0;
    let repeat_last_n = 64;

    let n_len = 256;
    let loaded_model = loaded_model.context("missing model")?;
    
    let prompt = {
        let mut formatted = String::new();
        formatted.push_str("<|system|>\n");
        formatted.push_str(
            "You are a friendly chatbot who always responds in the style of a pirate</s>\n",
        );
        formatted.push_str("<|user|>\n");
        formatted.push_str(prompt);
        formatted.push_str("</s>\n");
        formatted.push_str("<|assistant|>\n");
        formatted
    };

    let mut ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZero::new(2048))
        .with_seed(1234);
    let mut ctx = loaded_model
        .model
        .new_context(&loaded_model.backend, ctx_params)
        .context("failed to create llama context")?;

    let tokens_list = loaded_model
        .model
        .str_to_token(&prompt, AddBos::Always)
        .with_context(|| format!("failed to tokenize {prompt}"))?;

    let n_cxt = ctx.n_ctx() as i32;
    let n_kv_req = tokens_list.len() as i32 + (n_len - tokens_list.len() as i32);

    ensure!(
        n_kv_req <= n_cxt,
        "the required kv cache size is not big enough"
    );
    ensure!(
        tokens_list.len() < usize::try_from(n_len)?,
        "the prompt is too long, it has more tokens than n_len"
    );

    let mut batch = LlamaBatch::new(512, 1);

    let last_index: i32 = (tokens_list.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        // llama_decode will output logits only for the last token of the prompt
        let is_last = i == last_index;
        batch.add(token, i, &[0], is_last)?;
    }

    ctx.decode(&mut batch)
        .with_context(|| "llama_decode() failed")?;

    let mut n_cur = batch.n_tokens();
    let mut n_decode = 0;

    let generate_start_time = Instant::now();

    while n_cur <= n_len {
        // sample the next token
        {
            let candidates = ctx.candidates_ith(batch.n_tokens() - 1);

            let candidates_p = LlamaTokenDataArray::from_iter(candidates, false);

            // sample the most likely token
            let new_token_id = ctx.sample_token_greedy(candidates_p);

            // is it an end of stream?
            if new_token_id == loaded_model.model.token_eos() {
                break;
            }

            let output_bytes = loaded_model
                .model
                .token_to_bytes(new_token_id, Special::Tokenize)?;
            let output_string = String::from_utf8(output_bytes)?;

            futures::executor::block_on((progress_handler)(output_string));

            batch.clear();
            batch.add(new_token_id, n_cur, &[0], true)?;
        }

        n_cur += 1;

        ctx.decode(&mut batch).with_context(|| "failed to eval")?;

        n_decode += 1;
    }
    /*
    let device = Device::Cpu;

    let model = model.context("missing model")?;
    

    info!("Prompt: {prompt}");

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

        if let Some(t) = model.token_output_stream.next_token(next_token)? {
            futures::executor::block_on((progress_handler)(t));
        }
    }
    */

    Ok(())
}
