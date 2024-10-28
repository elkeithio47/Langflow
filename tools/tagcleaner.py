import re
from langflow.custom import Component
from langflow.io import MessageTextInput, Output
from langflow.schema import Data


class TagCleanerComponent(Component):
    display_name = "Tag Cleaner"
    description = "Removes unwanted < > tags from a message string."
    documentation: str = "http://docs.langflow.org/components/custom"
    icon = "custom_components"
    name = "TagCleaner"

    inputs = [
        MultilineInput(
            name="input_value",
            display_name="Text",
            value="",
            info="Message containing tags to be cleaned.",
        )
    ]

    outputs = [
       Output(display_name="Cleaned Message", name="cleaned_message", method="clean_message")
    ]

    def clean_message(self) -> Message:
        """
        This method cleans the input message by removing < > tags and returns the cleaned message.
        """
        # Use regex to remove any text inside < >
        cleaned_text = re.sub(r'<.*?>', '', self.input_value)
        
        self.status = cleaned_text
        return cleaned_text
