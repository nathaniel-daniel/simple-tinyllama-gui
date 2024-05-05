use anyhow::Context;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;

type ProgressHandler =
    Box<dyn FnMut(f32) -> Pin<Box<dyn Future<Output = ()> + Send + 'static>> + Send + Sync>;

enum Message {
    GetModelPath {
        user: Box<str>,
        repo: Box<str>,
        file: Box<str>,
        tx: tokio::sync::oneshot::Sender<anyhow::Result<Option<PathBuf>>>,
    },
    DownloadModel {
        user: Box<str>,
        repo: Box<str>,
        file: Box<str>,

        progress_handler: ProgressHandler,
        tx: tokio::sync::oneshot::Sender<anyhow::Result<PathBuf>>,
    },
}

#[derive(Debug, Clone)]
pub struct ModelManager {
    tx: tokio::sync::mpsc::Sender<Message>,
}

impl ModelManager {
    /// Make a new model manager task.
    pub fn new() -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(32);

        tokio::spawn(task(rx));

        Self { tx }
    }

    pub async fn get_model_path(
        &self,
        user: Box<str>,
        repo: Box<str>,
        file: Box<str>,
    ) -> anyhow::Result<Option<PathBuf>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(Message::GetModelPath {
                user,
                repo,
                file,
                tx,
            })
            .await?;
        rx.await?
    }

    pub async fn download_model<F, FU>(
        &self,
        user: Box<str>,
        repo: Box<str>,
        file: Box<str>,
        mut progress_handler: F,
    ) -> anyhow::Result<PathBuf>
    where
        F: FnMut(f32) -> FU + Send + Sync + 'static,
        FU: Future<Output = ()> + Send + 'static,
    {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(Message::DownloadModel {
                user,
                repo,
                file,
                progress_handler: Box::new(move |progress| Box::pin((progress_handler)(progress))),
                tx,
            })
            .await?;
        rx.await?
    }
}

async fn task(mut rx: tokio::sync::mpsc::Receiver<Message>) {
    let client = reqwest::Client::new();
    let data_dir =
        dirs::data_dir().map(|data_dir| Arc::new(data_dir.join("simple-tiny-llama-gui")));

    while let Some(message) = rx.recv().await {
        match message {
            Message::GetModelPath {
                user,
                repo,
                file,
                tx,
            } => {
                let data_dir = data_dir.clone();
                let result = get_model_path(data_dir, user, repo, file).await;
                let _ = tx.send(result).is_ok();
            }
            Message::DownloadModel {
                user,
                repo,
                file,
                progress_handler,
                tx,
            } => {
                let data_dir = data_dir.clone();
                let client = client.clone();
                let result =
                    download_model(data_dir, client, user, repo, file, progress_handler).await;
                let _ = tx.send(result).is_ok();
            }
        }
    }
}

async fn get_model_path(
    data_dir: Option<Arc<PathBuf>>,
    user: Box<str>,
    repo: Box<str>,
    file: Box<str>,
) -> anyhow::Result<Option<PathBuf>> {
    let data_dir = data_dir.context("failed to get data dir")?;
    tokio::fs::create_dir_all(&*data_dir)
        .await
        .context("failed to create data dir")?;

    let file_path = {
        let mut path = data_dir.to_path_buf();
        path.extend([&*user, &*repo]);
        path.extend(file.split('/'));
        path
    };

    if let Some(parent_dir) = file_path.parent() {
        tokio::fs::create_dir_all(parent_dir)
            .await
            .context("failed to create parent path dir")?;
    }

    if file_path.try_exists()? {
        Ok(Some(file_path))
    } else {
        Ok(None)
    }
}

/// user: TheBloke
/// repo: TinyLlama-1.1B-Chat-v0.3-GGUF
/// file: tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf
async fn download_model(
    data_dir: Option<Arc<PathBuf>>,
    client: reqwest::Client,
    user: Box<str>,
    repo: Box<str>,
    file: Box<str>,
    mut progress_handler: ProgressHandler,
) -> anyhow::Result<PathBuf> {
    let data_dir = data_dir.context("failed to get data dir")?;
    tokio::fs::create_dir_all(&*data_dir)
        .await
        .context("failed to create data dir")?;

    let file_path = {
        let mut path = data_dir.to_path_buf();
        path.extend([&*user, &*repo]);
        path.extend(file.split('/'));
        path
    };

    if let Some(parent_dir) = file_path.parent() {
        tokio::fs::create_dir_all(parent_dir)
            .await
            .context("failed to create parent path dir")?;
    }

    let tmp_file_path = nd_util::with_push_extension(&file_path, "tmp");
    let mut tmp_file = tokio::fs::File::create(&tmp_file_path)
        .await
        .context("failed to open temporary file")?;

    let url = format!("https://huggingface.co/{user}/{repo}/resolve/main/{file}?download=true");
    let mut response = client
        .get(url)
        .send()
        .await
        .and_then(|response| response.error_for_status())
        .context("failed to send request")?;

    let content_length = response
        .content_length()
        .context("missing content length")?;

    let mut current_bytes = 0;
    while let Some(chunk) = response
        .chunk()
        .await
        .context("failed to download next chunk")?
    {
        tmp_file
            .write_all(&chunk)
            .await
            .context("failed to write to file")?;
        let chunk_len = u64::try_from(chunk.len())?;

        current_bytes += chunk_len;

        let progress = (current_bytes as f32) / (content_length as f32);
        (progress_handler)(progress).await;
    }

    tmp_file
        .flush()
        .await
        .context("failed to flush model file")?;
    tmp_file
        .sync_all()
        .await
        .context("failed to sync model file")?;
    tokio::fs::rename(&tmp_file_path, &file_path)
        .await
        .context("failed to rename temporary model file")?;

    Ok(file_path)
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}
