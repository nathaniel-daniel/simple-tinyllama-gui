use crate::Icon;
use crate::Message;
use crate::Tab;
use iced::widget::Column;
use iced::widget::Container;
use iced::widget::Scrollable;
use iced::widget::Text;
use iced::widget::TextInput;
use iced::Element;
use iced::Length;
use iced_aw::TabLabel;

#[derive(Debug, Clone)]
pub enum ChatMessage {
    Input(String),
    SubmitInput,
}

#[derive(Debug)]
pub struct ChatTab {
    input: String,
    history: Vec<(String, String)>,
}

impl ChatTab {
    pub fn new() -> Self {
        Self {
            input: String::new(),
            history: Vec::new(),
        }
    }

    pub fn update(&mut self, message: ChatMessage) {
        match message {
            ChatMessage::Input(input) => {
                self.input = input;
            }
            ChatMessage::SubmitInput => {
                self.history.push(("User".into(), self.input.clone()));
                self.input.clear();
            }
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
        let content: Element<'_, ChatMessage> = Column::new()
            .push(
                Container::new(
                    Scrollable::new(Column::with_children(
                        self.history
                            .iter()
                            .map(|(user, text)| Text::new(format!("{user}: {text}")).into()),
                    ))
                    .width(Length::Fill),
                )
                .width(Length::Fill)
                .height(Length::Fill),
            )
            .push(
                TextInput::new("Start chatting...", &self.input)
                    .on_input(ChatMessage::Input)
                    .on_submit(ChatMessage::SubmitInput),
            )
            .into();

        content.map(Message::ChatTab)
    }
}
