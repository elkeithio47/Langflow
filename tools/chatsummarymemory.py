from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langflow.custom import Component
from langflow.field_typing import BaseChatMemory
from langflow.helpers.data import data_to_text
from langflow.inputs import HandleInput
from langflow.io import DropdownInput, IntInput, MessageTextInput, MultilineInput, Output
from langflow.memory import LCBuiltinChatMemory, get_messages
from langflow.schema import Data
from langflow.schema.message import Message
from langflow.utils.constants import MESSAGE_SENDER_AI, MESSAGE_SENDER_USER
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import tiktoken
import langwatch

class MemoryComponent(Component):
    display_name = "Chat Summary Memory"
    description = "Retrieves stored chat messages from Langflow tables or an external memory."
    icon = "message-square-more"
    name = "Memory"

    inputs = [
        HandleInput(name="llm", display_name="Language Model", input_types=["LanguageModel"], required=True),
        HandleInput(
            name="memory",
            display_name="External Memory",
            input_types=["BaseChatMessageHistory"],
            info="Retrieve messages from an external memory. If empty, it will use the Langflow tables.",
        ),
        DropdownInput(
            name="sender",
            display_name="Sender Type",
            options=[MESSAGE_SENDER_AI, MESSAGE_SENDER_USER, "Machine and User"],
            value="Machine and User",
            info="Filter by sender type.",
            advanced=True,
        ),
        MessageTextInput(
            name="sender_name",
            display_name="Sender Name",
            info="Filter by sender name.",
            advanced=True,
        ),
        IntInput(
            name="n_messages",
            display_name="Number of Messages",
            value=100,
            info="Number of messages to retrieve.",
            advanced=True,
        ),
        MessageTextInput(
            name="session_id",
            display_name="Session ID",
            info="The session ID of the chat. If empty, the current session ID parameter will be used.",
            advanced=True,
        ),
        DropdownInput(
            name="order",
            display_name="Order",
            options=["Ascending", "Descending"],
            value="Ascending",
            info="Order of the messages.",
            advanced=True,
        ),
        MultilineInput(
            name="template",
            display_name="Template",
            info="The template to use for formatting the data. "
            "It can contain the keys {text}, {sender} or any other key in the message data.",
            value="{sender_name}: {text}",
            advanced=True,
        ),
        IntInput(
            name="n_character_limit",
            display_name="Summarization Character Limit",
            value=1000,
            info="Desired character limit used to summarize conversation message history",
            advanced=True,
        ),
        MultilineInput(
            name="user_prompt", display_name="Prompt", info="Summarization strategies for controlling conversational growth", value="{text} {char_limit}"
        ),
    ]

    outputs = [
        Output(display_name="Messages (Text)", name="messages_text", method="retrieve_messages_as_text"),
    ]

    def retrieve_messages(self) -> Data:
        sender = self.sender
        sender_name = self.sender_name
        session_id = self.session_id
        n_messages = self.n_messages
        order = "DESC" if self.order == "Descending" else "ASC"

        if sender == "Machine and User":
            sender = None

        if self.memory:
            # override session_id
            self.memory.session_id = session_id

            stored = self.memory.messages
            # langchain memories are supposed to return messages in ascending order
            if order == "DESC":
                stored = stored[::-1]
            if n_messages:
                stored = stored[:n_messages]
            stored = [Message.from_lc_message(m) for m in stored]
            if sender:
                expected_type = MESSAGE_SENDER_AI if sender == MESSAGE_SENDER_AI else MESSAGE_SENDER_USER
                stored = [m for m in stored if m.type == expected_type]
        else:
            stored = get_messages(
                sender=sender,
                sender_name=sender_name,
                session_id=session_id,
                limit=n_messages,
                order=order,
            )
        self.status = stored
        return stored

    def retrieve_messages_as_text(self) -> Message:
        retrieved_messages = self.retrieve_messages()

        # Check if there are any messages to summarize
        if not retrieved_messages:
            print("No conversation message history found. Skipping summarization.")
            return Message(text="No conversation history available for summarization.")

        message_text = data_to_text(self.template, retrieved_messages)

        # Dynamically determine the token counting method based on model name
        original_token_count = self.count_tokens(message_text, model_name=self.llm.model_id)
        print(f"Current message text token count: {original_token_count}")

        if original_token_count > self.n_character_limit:
            summarized_text = self.summarize_text(message_text, self.n_character_limit)
    
            summarized_token_count = self.count_tokens(summarized_text, model_name=self.llm.model_id)
            print(f"Summarized message text token count: {summarized_token_count}")
     
            self.status = summarized_text
            print(f"New conversation message history:\n{summarized_text}")

            return Message(text=summarized_text)
        
        return Message(text="No conversation history available for summarization.")

    @langwatch.trace()
    def summarize_text(self, text: str, char_limit: int) -> str:

        prompt_template = PromptTemplate(
            template=self.user_prompt,
            input_variables=["text", "char_limit"] 
        )
        
        summarization_chain = LLMChain(llm=self.llm, prompt=prompt_template)

        summarized_output = summarization_chain.predict(
            text=text,  # Pass the conversation text directly (as a string, not wrapped in a list)
            char_limit=char_limit,  # Pass the character limit
            callbacks=[langwatch.get_current_trace().get_langchain_callback()]  # Passing callbacks
        )
        
        return summarized_output
    
    # Function for approximate token counting for Anthropic's models
    def approximate_anthropic_token_count(self,text: str) -> int:
	    """Approximate the token count for Anthropic's Claude models."""
	    # Approximate 1 token ≈ 4 characters (including spaces)
	    return len(text) // 4
    
    def count_tokens(self, text: str, model_name: str) -> int:
        """Dynamically count the tokens based on the model type."""
        if "claude" in model_name.lower():  # Check if it's an Anthropic Claude model
            print(f"Using Anthropic's approximate token counting for model: {model_name}")
            return self.approximate_anthropic_token_count(text)
        else:
            print(f"Using OpenAI tiktoken for model: {model_name}")
            try:
                # Load the tokenizer for the specified model using tiktoken
                encoding = tiktoken.encoding_for_model(model_name)
                return len(encoding.encode(text))
            except Exception as e:
                print(f"Error in token counting: {str(e)}")
                return len(text) // 4  # Fallback approximation

