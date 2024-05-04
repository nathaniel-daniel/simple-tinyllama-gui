use crate::Icon;
use crate::Message;
use crate::Tab;
use anyhow::Context;
use iced::alignment::Horizontal;
use iced::alignment::Vertical;
use iced::futures::SinkExt;
use iced::widget::row;
use iced::widget::Button;
use iced::widget::Column;
use iced::widget::Container;
use iced::widget::ProgressBar;
use iced::widget::Scrollable;
use iced::widget::Text;
use iced::widget::TextInput;
use iced::Element;
use iced::Length;
use iced::Subscription;
use iced_aw::Card;
use iced_aw::Modal;
use iced_aw::TabLabel;
use std::path::Path;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tracing::error;

#[derive(Debug, Clone)]
pub enum ChatMessage {
    DownloadModel,
    DownloadModelError(Arc<anyhow::Error>),
    DownloadModelProgress(f32),
    DownloadModelComplete,

    Input(String),
    SubmitInput,

    CloseErrorModal,
}

#[derive(Debug)]
pub struct ChatTab {
    input: String,
    history: Vec<(String, String)>,
    model_path: Arc<Path>,
    downloaded_model: bool,
    model_download_progress: Option<f32>,

    error: Option<Arc<anyhow::Error>>,
}

impl ChatTab {
    pub fn new() -> Self {
        let model_path: Arc<Path> = Arc::from(Path::new("tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"));
        let downloaded_model = model_path.exists();

        Self {
            input: String::new(),
            history: Vec::new(),
            model_path,
            downloaded_model,
            model_download_progress: None,

            error: None,
        }
    }

    pub fn update(&mut self, message: ChatMessage) {
        match message {
            ChatMessage::DownloadModel => {
                self.model_download_progress = Some(0.0);
            }
            ChatMessage::DownloadModelError(error) => {
                error!("{error:?}");
                self.error = Some(error);
            }
            ChatMessage::DownloadModelProgress(progress) => {
                self.model_download_progress = Some(progress);
            }
            ChatMessage::DownloadModelComplete => {
                self.model_download_progress = None;
                self.downloaded_model = self.model_path.exists();
            }
            ChatMessage::Input(input) => {
                self.input = input;
            }
            ChatMessage::SubmitInput => {
                self.history.push(("User".into(), self.input.clone()));
                self.input.clear();
            }
            ChatMessage::CloseErrorModal => {
                self.error = None;
            }
        }
    }

    pub fn subscription(&self) -> Subscription<ChatMessage> {
        if self.model_download_progress.is_none() {
            Subscription::none()
        } else {
            download_worker(self.model_path.clone())
        }
    }
}

impl Tab for ChatTab {
    type Message = Message;

    fn title(&self) -> String {
        String::from("Chat")
    }

    fn tab_label(&self) -> TabLabel {
        TabLabel::IconText(Icon::Chat.into(), self.title())
    }

    fn content(&self) -> Element<'_, Self::Message> {
        const DEFAULT_PADDING: f32 = 5.0;

        let chat_messages = Container::new(
            Scrollable::new(Column::with_children(
                self.history
                    .iter()
                    .map(|(user, text)| Text::new(format!("{user}: {text}")).into()),
            ))
            .width(Length::Fill)
            .height(Length::Fill),
        );

        let model_upkeep = if !self.downloaded_model {
            let element = match self.model_download_progress {
                Some(progress) => Container::new(row![
                    Container::new(Text::new("Downloading Model..."))
                        .align_y(Vertical::Center)
                        .padding(DEFAULT_PADDING),
                    ProgressBar::new(0.0..=1.0, progress),
                    Container::new(Text::new(format!("{:.02}%", progress * 100.0)))
                        .align_y(Vertical::Center)
                        .padding(DEFAULT_PADDING)
                ])
                .padding([DEFAULT_PADDING, 0.0]),
                None => Container::new(row![
                    Container::new(Text::new("No model detected. Download model to begin."))
                        .align_y(Vertical::Center)
                        .padding(DEFAULT_PADDING),
                    Button::new("Download Model").on_press(ChatMessage::DownloadModel)
                ])
                .align_x(Horizontal::Center)
                .padding([DEFAULT_PADDING, 0.0]),
            };

            Some(element)
        } else {
            None
        };

        let chat_history = Column::new()
            .push(chat_messages)
            .push_maybe(model_upkeep)
            .width(Length::Fill)
            .height(Length::Fill);

        let underlay = Column::new().push(chat_history).push(
            TextInput::new("Start chatting...", &self.input)
                .on_input(ChatMessage::Input)
                .on_submit(ChatMessage::SubmitInput),
        );

        let overlay = self.error.as_ref().map(|error| {
            Card::new(Text::new("Error"), Text::new(format!("{error:?}")))
                .foot(
                    Button::new(Text::new("Close").horizontal_alignment(Horizontal::Center))
                        .width(Length::Shrink)
                        .on_press(ChatMessage::CloseErrorModal),
                )
                .max_width(500.0)
                .on_close(ChatMessage::CloseErrorModal)
        });

        let content: Element<'_, ChatMessage> = Modal::new(underlay, overlay)
            .align_x(Horizontal::Center)
            .align_y(Vertical::Center)
            .on_esc(ChatMessage::CloseErrorModal)
            .into();

        content.map(Message::ChatTab)
    }
}

fn download_worker(model_path: Arc<Path>) -> Subscription<ChatMessage> {
    struct DownloadWorker;

    enum State {
        Starting,
        Working {
            response: reqwest::Response,
            total: u64,
            current: u64,
            model_file: tokio::fs::File,
        },
        Finished,
    }

    let url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf?download=true";
    // let url = "https://google.com";
    let client = reqwest::Client::new();
    iced::subscription::channel(
        std::any::TypeId::of::<DownloadWorker>(),
        100,
        move |mut channel| async move {
            let mut state = State::Starting;
            let tmp_model_path = nd_util::with_push_extension(&model_path, "tmp");

            loop {
                match &mut state {
                    State::Starting => {
                        let result = async {
                            let response = client
                                .get(url)
                                .send()
                                .await
                                .and_then(|response| response.error_for_status())
                                .context("failed to send request")?;
                            let content_length = response
                                .content_length()
                                .context("missing content length")?;

                            let model_file = tokio::fs::File::create(&tmp_model_path)
                                .await
                                .context("failed to open temporary model file")?;

                            anyhow::Ok((response, content_length, model_file))
                        }
                        .await;

                        let (response, content_length, model_file) = match result {
                            Ok(data) => data,
                            Err(error) => {
                                let error = Arc::new(error);
                                let _ = channel
                                    .send(ChatMessage::DownloadModelError(error.clone()))
                                    .await;

                                let _ = channel.send(ChatMessage::DownloadModelComplete).await;
                                state = State::Finished;
                                continue;
                            }
                        };

                        state = State::Working {
                            response,
                            total: content_length,
                            current: 0,
                            model_file,
                        };
                    }
                    State::Working {
                        response,
                        total,
                        current,
                        model_file,
                    } => {
                        let result = async {
                            let maybe_chunk = response
                                .chunk()
                                .await
                                .context("failed to download next chunk")?;

                            match maybe_chunk {
                                Some(chunk) => {
                                    model_file
                                        .write_all(&chunk)
                                        .await
                                        .context("failed to write to model file")?;

                                    anyhow::Ok(Some(u64::try_from(chunk.len())?))
                                }
                                None => {
                                    model_file
                                        .flush()
                                        .await
                                        .context("failed to flush model file")?;
                                    model_file
                                        .sync_all()
                                        .await
                                        .context("failed to sync model file")?;
                                    tokio::fs::rename(&tmp_model_path, &model_path)
                                        .await
                                        .context("failed to rename temporary model file")?;

                                    Ok(None)
                                }
                            }
                        }
                        .await;

                        let chunk_len = match result {
                            Ok(chunk) => chunk,
                            Err(error) => {
                                let error = Arc::new(error);
                                let _ = channel
                                    .send(ChatMessage::DownloadModelError(error.clone()))
                                    .await;

                                let _ = channel.send(ChatMessage::DownloadModelComplete).await;
                                state = State::Finished;
                                continue;
                            }
                        };

                        match chunk_len {
                            Some(chunk_len) => {
                                *current += chunk_len;
                                let progress = (*current as f32) / (*total as f32);
                                let _ = channel
                                    .send(ChatMessage::DownloadModelProgress(progress))
                                    .await;
                            }
                            None => {
                                let _ = channel.send(ChatMessage::DownloadModelProgress(1.0)).await;
                                let _ = channel.send(ChatMessage::DownloadModelComplete).await;
                                state = State::Finished;
                            }
                        }
                    }
                    State::Finished => iced::futures::future::pending().await,
                }
            }
        },
    )
}
