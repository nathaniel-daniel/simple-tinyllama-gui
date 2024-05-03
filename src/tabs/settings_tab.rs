use crate::Icon;
use crate::Message;
use crate::Tab;
use iced::widget::column;
use iced::widget::Text;
use iced::Element;
use iced_aw::TabLabel;

#[derive(Debug)]
pub struct SettingsTab {}

impl SettingsTab {
    pub fn new() -> Self {
        Self {}
    }
}

impl Tab for SettingsTab {
    type Message = Message;

    fn title(&self) -> String {
        String::from("Settings")
    }

    fn tab_label(&self) -> TabLabel {
        TabLabel::IconText(Icon::Settings.into(), self.title())
    }

    fn content(&self) -> Element<'_, Self::Message> {
        Text::new("Settings").into()
    }
}
