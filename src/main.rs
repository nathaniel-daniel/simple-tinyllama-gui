#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod model_manager;
mod model_runner;
mod tabs;

pub use self::model_manager::ModelManager;
pub use self::model_runner::ModelRunner;
use self::tabs::ChatMessage;
use self::tabs::ChatTab;
use self::tabs::SettingsTab;
use anyhow::Context;
use iced::alignment::Horizontal;
use iced::alignment::Vertical;
use iced::widget::Container;
use iced::Application;
use iced::Command;
use iced::Element;
use iced::Length;
use iced::Settings;
use iced::Theme;
use iced_aw::TabLabel;
use iced_aw::Tabs;
use tracing::info;

const TAB_PADDING: u16 = 16;

#[derive(Debug, Clone)]
enum Message {
    TabSelected(TabId),
    ChatTab(ChatMessage),
}

#[derive(Debug)]
struct State {
    active_tab: TabId,
    chat_tab: ChatTab,
    settings_tab: SettingsTab,
}

#[derive(Clone, PartialEq, Eq, Debug)]
enum TabId {
    Chat,
    Settings,
}

impl Application for State {
    type Message = Message;
    type Theme = Theme;
    type Executor = iced::executor::Default;
    type Flags = ();

    fn new(_flags: ()) -> (Self, Command<Message>) {
        let (chat_tab, command) = ChatTab::new();
        let state = Self {
            active_tab: TabId::Chat,
            chat_tab,
            settings_tab: SettingsTab::new(),
        };

        (state, command.map(Message::ChatTab))
    }

    fn title(&self) -> String {
        "Simple TinyLlama GUI".to_string()
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::TabSelected(selected) => {
                self.active_tab = selected;

                Command::none()
            }
            Message::ChatTab(message) => self.chat_tab.update(message).map(Message::ChatTab),
        }
    }

    fn view(&self) -> Element<Message> {
        Tabs::new(Message::TabSelected)
            .push(TabId::Chat, self.chat_tab.tab_label(), self.chat_tab.view())
            .push(
                TabId::Settings,
                self.settings_tab.tab_label(),
                self.settings_tab.view(),
            )
            .set_active_tab(&self.active_tab)
            .into()
    }

    fn theme(&self) -> Self::Theme {
        Theme::SolarizedDark
    }
}

trait Tab {
    type Message;

    fn title(&self) -> String;

    fn tab_label(&self) -> TabLabel;

    fn view(&self) -> Element<'_, Self::Message> {
        Container::new(self.content())
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(Horizontal::Center)
            .align_y(Vertical::Center)
            .padding(TAB_PADDING)
            .into()
    }

    fn content(&self) -> Element<'_, Self::Message>;
}

enum Icon {
    Chat,
    Settings,
}

impl From<Icon> for char {
    fn from(icon: Icon) -> Self {
        match icon {
            Icon::Chat => '\u{1F5E8}',
            Icon::Settings => '\u{2699}',
        }
    }
}

fn main() -> anyhow::Result<()> {
    // std::env::set_var("ICED_BACKEND", "tiny-skia");
    std::env::set_var("WGPU_POWER_PREF", "low");
    std::env::set_var("WGPU_BACKEND", "opengl");

    let appender = tracing_appender::rolling::never("", "simple-tiny-llama-gui.log");
    let (non_blocking_appender, _guard) = tracing_appender::non_blocking(appender);
    tracing_subscriber::fmt()
        .with_writer(non_blocking_appender)
        .with_ansi(false)
        .try_init()
        .ok()
        .context("failed to install global logger")?;

    info!("starting application");

    let mut settings = Settings::default();
    settings
        .fonts
        .push(iced_aw::core::icons::BOOTSTRAP_FONT_BYTES.into());
    State::run(settings)?;

    Ok(())
}
