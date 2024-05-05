use crate::Icon;
use crate::Message;
use crate::ModelManager;
use crate::Tab;
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
use iced::Command;
use iced::Element;
use iced::Length;
use iced_aw::Card;
use iced_aw::Modal;
use iced_aw::TabLabel;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::error;
use tracing::info;

const DEFAULT_PADDING: f32 = 5.0;

const MODEL_USER: &str = "TheBloke";
const MODEL_REPO: &str = "TinyLlama-1.1B-Chat-v0.3-GGUF";
const MODEL_FILE: &str = "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf";

const TOKENIZER_USER: &str = "hf-internal-testing";
const TOKENIZER_REPO: &str = "llama-tokenizer";
const TOKENIZER_FILE: &str = "tokenizer.model";

#[derive(Debug, Clone)]
pub enum ChatMessage {
    ModelStatus {
        model_path: Option<Arc<PathBuf>>,
        tokenizer_path: Option<Arc<PathBuf>>,
    },

    DownloadModel,
    DownloadModelError(Arc<anyhow::Error>),
    DownloadModelProgress(f32),
    DownloadModelOk(Arc<PathBuf>),

    DownloadTokenizer,
    DownloadTokenizerError(Arc<anyhow::Error>),
    DownloadTokenizerProgress(f32),
    DownloadTokenizerOk(Arc<PathBuf>),

    Input(String),
    SubmitInput,

    CloseErrorModal,
}

#[derive(Debug)]
pub struct ChatTab {
    model_manager: ModelManager,

    input: String,
    history: Vec<(String, String)>,

    fetching_model_status: bool,
    model_path: Option<Arc<PathBuf>>,
    model_download_progress: Option<f32>,
    tokenizer_path: Option<Arc<PathBuf>>,
    tokenizer_download_progress: Option<f32>,

    error: Option<Arc<anyhow::Error>>,
}

impl ChatTab {
    pub fn new() -> (Self, Command<ChatMessage>) {
        let model_manager = ModelManager::new();

        (
            Self {
                model_manager: model_manager.clone(),

                input: String::new(),
                history: Vec::new(),

                fetching_model_status: true,
                model_path: None,
                model_download_progress: None,
                tokenizer_path: None,
                tokenizer_download_progress: None,

                error: None,
            },
            Command::perform(
                async move {
                    let model_path = model_manager
                        .get_model_path(MODEL_USER.into(), MODEL_REPO.into(), MODEL_FILE.into())
                        .await?;
                    let tokenizer_path = model_manager
                        .get_model_path(
                            TOKENIZER_USER.into(),
                            TOKENIZER_REPO.into(),
                            TOKENIZER_FILE.into(),
                        )
                        .await?;

                    anyhow::Ok((model_path, tokenizer_path))
                },
                |model_path| {
                    let (model_path, tokenizer_path) = match model_path {
                        Ok((model_path, tokenizer_path)) => {
                            (model_path.map(Arc::new), tokenizer_path.map(Arc::new))
                        }
                        Err(error) => {
                            error!("{error:?}");
                            (None, None)
                        }
                    };

                    ChatMessage::ModelStatus {
                        model_path,
                        tokenizer_path,
                    }
                },
            ),
        )
    }

    pub fn update(&mut self, message: ChatMessage) -> Command<ChatMessage> {
        match message {
            ChatMessage::ModelStatus {
                model_path,
                tokenizer_path,
            } => {
                if self.fetching_model_status {
                    info!("model_path = {:?}", model_path);
                    info!("tokenizer_path = {:?}", tokenizer_path);

                    self.model_path = model_path;
                    self.tokenizer_path = tokenizer_path;

                    self.fetching_model_status = false;
                }
            }
            ChatMessage::DownloadModel => {
                self.model_download_progress = Some(0.0);
                return download_model(self.model_manager.clone());
            }
            ChatMessage::DownloadModelError(error) => {
                error!("{error:?}");
                self.error = Some(error);
                self.model_download_progress = None;
            }
            ChatMessage::DownloadModelProgress(progress) => {
                self.model_download_progress = Some(progress);
            }
            ChatMessage::DownloadModelOk(model_path) => {
                self.model_download_progress = None;
                self.model_path = Some(model_path);
            }
            ChatMessage::DownloadTokenizer => {
                self.tokenizer_download_progress = Some(0.0);
                return download_tokenizer(self.model_manager.clone());
            }
            ChatMessage::DownloadTokenizerError(error) => {
                error!("{error:?}");
                self.error = Some(error);
                self.tokenizer_download_progress = None;
            }
            ChatMessage::DownloadTokenizerProgress(progress) => {
                self.tokenizer_download_progress = Some(progress);
            }
            ChatMessage::DownloadTokenizerOk(model_path) => {
                self.tokenizer_download_progress = None;
                self.tokenizer_path = Some(model_path);
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

        Command::none()
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
        let chat_messages = Container::new(
            Scrollable::new(Column::with_children(
                self.history
                    .iter()
                    .map(|(user, text)| Text::new(format!("{user}: {text}")).into()),
            ))
            .width(Length::Fill)
            .height(Length::Fill),
        );

        let model_upkeep = if self.fetching_model_status {
            Some(
                Container::new(Text::new("Checking model status..."))
                    .padding([DEFAULT_PADDING, 0.0]),
            )
        } else if self.model_path.is_none() {
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
        } else if self.tokenizer_path.is_none() {
            let element = match self.tokenizer_download_progress {
                Some(progress) => Container::new(row![
                    Container::new(Text::new("Downloading Tokenizer..."))
                        .align_y(Vertical::Center)
                        .padding(DEFAULT_PADDING),
                    ProgressBar::new(0.0..=1.0, progress),
                    Container::new(Text::new(format!("{:.02}%", progress * 100.0)))
                        .align_y(Vertical::Center)
                        .padding(DEFAULT_PADDING)
                ])
                .padding([DEFAULT_PADDING, 0.0]),
                None => Container::new(row![
                    Container::new(Text::new(
                        "No tokenizer detected. Download tokenizer to begin."
                    ))
                    .align_y(Vertical::Center)
                    .padding(DEFAULT_PADDING),
                    Button::new("Download Tokenizer").on_press(ChatMessage::DownloadTokenizer)
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

        let mut text_input = TextInput::new("Start chatting...", &self.input);
        if self.model_path.is_some() {
            text_input = text_input
                .on_input(ChatMessage::Input)
                .on_submit(ChatMessage::SubmitInput);
        }
        let underlay = Column::new().push(chat_history).push(text_input);

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

fn download_model(model_manager: ModelManager) -> Command<ChatMessage> {
    iced::command::channel(100, move |mut channel| async move {
        let result = {
            let channel = channel.clone();

            model_manager
                .download_model(
                    MODEL_USER.into(),
                    MODEL_REPO.into(),
                    MODEL_FILE.into(),
                    move |progress| {
                        let mut channel = channel.clone();

                        async move {
                            let _ = channel
                                .send(ChatMessage::DownloadModelProgress(progress))
                                .await;
                        }
                    },
                )
                .await
        };

        match result {
            Ok(model_path) => {
                let _ = channel
                    .send(ChatMessage::DownloadModelOk(Arc::new(model_path)))
                    .await;
            }
            Err(error) => {
                let error = Arc::new(error);
                let _ = channel
                    .send(ChatMessage::DownloadModelError(error.clone()))
                    .await;
            }
        }
    })
}

fn download_tokenizer(model_manager: ModelManager) -> Command<ChatMessage> {
    iced::command::channel(100, move |mut channel| async move {
        let result = {
            let channel = channel.clone();

            model_manager
                .download_model(
                    TOKENIZER_USER.into(),
                    TOKENIZER_REPO.into(),
                    TOKENIZER_FILE.into(),
                    move |progress| {
                        let mut channel = channel.clone();

                        async move {
                            let _ = channel
                                .send(ChatMessage::DownloadTokenizerProgress(progress))
                                .await;
                        }
                    },
                )
                .await
        };

        match result {
            Ok(model_path) => {
                let _ = channel
                    .send(ChatMessage::DownloadTokenizerOk(Arc::new(model_path)))
                    .await;
            }
            Err(error) => {
                let error = Arc::new(error);
                let _ = channel
                    .send(ChatMessage::DownloadTokenizerError(error.clone()))
                    .await;
            }
        }
    })
}
