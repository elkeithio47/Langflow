from langchain.memory import ConversationBufferMemory,ConversationSummaryBufferMemory

from langflow.custom import Component
from langflow.field_typing import BaseChatMemory
from langflow.helpers.data import data_to_text
from langflow.inputs import HandleInput
from langflow.io import DropdownInput, IntInput, MessageTextInput, MultilineInput, Output
from langflow.memory import LCBuiltinChatMemory, get_messages
from langflow.schema import Data
from langflow.schema.message import Message
from langflow.utils.constants import MESSAGE_SENDER_AI, MESSAGE_SENDER_USER
#from langchain.utils import count_tokens  # Utility to count tokens

from langchain.chat_models import ChatOpenAI  # Use ChatOpenAI for chat-based models
from langchain.llms import OpenAI  # Assuming OpenAI model is used for summarization
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import tiktoken  # Tokenizer for OpenAI models
import langwatch
#from openai import OpenAI

#from langchain.callbacks.base import CallbackManager, BaseCallbackHandler

class MemoryComponent(Component):
    display_name = "Chat Memory"
    description = "Retrieves stored chat messages from Langflow tables or an external memory."
    icon = "message-square-more"
    name = "Memory"

    inputs = [
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
            info="Desired character limit used to summarize converstion message history",
            advanced=True,
        )
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
        # Step 1: Retrieve the messages using the existing retrieve_messages() method
        retrieved_messages = self.retrieve_messages()

            # Step 2: Check if there are any messages to summarize
        if not retrieved_messages:
            # If no messages exist, handle the case appropriately (e.g., return a message or empty response)
            print("No conversation message history found. Skipping summarization.")
            return Message(text="No conversation history available for summarization.")

        # Step 2: Convert the message data into a format suitable for summarization
        message_text = data_to_text(self.template, retrieved_messages)

        # Step 3: Perform summarization using LangChain's summarization capabilities
        # Pass a desired character limit to summarize_text method
        summarized_text = self.summarize_text(message_text, self.n_character_limit)  # Example limit

        # Step 4: Update the status and return the summarized message
        self.status = summarized_text
        return Message(text=summarized_text)        # Step 1: Retrieve the messages using the existing retrieve_messages() method
        retrieved_messages = self.retrieve_messages()

        # Step 2: Convert the message data into a format suitable for summarization
        message_text = data_to_text(self.template, retrieved_messages)

        # Step 3: Print the original message text and token count before summarization
        original_token_count = self.count_tokens(message_text, model_name="gpt-3.5-turbo")
        print(f"Original message text token count: {original_token_count}")
        print(f"Original message text:\n{message_text}\n")

        # Step 4: Perform summarization using LangChain's summarization capabilities
        summarized_text = self.summarize_text(message_text, char_limit=500)  # Example limit

        # Step 5: Print the summarized message text and token count after summarization
        summarized_token_count = self.count_tokens(summarized_text, model_name="gpt-3.5-turbo")
        print(f"Summarized message text token count: {summarized_token_count}")
        print(f"Summarized message text:\n{summarized_text}\n")

        # Step 6: Print the conversation message history (after summarization)
        self.status = summarized_text
        print(f"New conversation message history:\n{summarized_text}")

        return Message(text=summarized_text)

    @langwatch.trace()
    def summarize_text(self, text: str, char_limit: int) -> str:
        
        #client = OpenAI()
        #langwatch.get_current_trace().autotrack_openai_calls(client)
        # Step 7: Implement the summarization logic using an LLM chain with character limit

        openai_api_key = "key"  # Replace with your OpenAI API key

        # Use ChatOpenAI for chat models like gpt-3.5-turbo
        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)

        # Modify the prompt template to include the character limit instruction
        prompt_template = PromptTemplate(
            template=f"Summarize the following conversation in {char_limit} characters or fewer:\n\n{{text}}\n",
            input_variables=["text"]
        )

        summarization_chain = LLMChain(llm=llm, prompt=prompt_template)

        # Run the summarization chain using the 'predict' method
        summarized_output = summarization_chain.predict(
            text=text,  # Pass the conversation text directly (as a string, not wrapped in a list)
            callbacks=[langwatch.get_current_trace().get_langchain_callback()]  # Passing callbacks
        )
        
        # Print the summarized result
        print(summarized_output)


        return summarized_output

    def count_tokens(self, text: str, model_name: str) -> int:
        """Count the tokens in the text using tiktoken."""
        # Load the tokenizer for the specified model
        encoding = tiktoken.encoding_for_model(model_name)

        # Encode the text and return the number of tokens
        return len(encoding.encode(text))


