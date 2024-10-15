from typing import Any

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from langflow.base.langchain_utilities.model import LCToolComponent
from langflow.field_typing import Tool
from langflow.inputs import IntInput, MessageTextInput, SecretStrInput
from langflow.schema import Data
import requests


# Define the component class for News API
class NewsAPIComponent(LCToolComponent):
    display_name: str = "News API"
    description: str = "Retrieve the latest news articles using the News API."
    name = "NewsAPI"
    documentation: str = "https://newsapi.org/docs/endpoints/everything"

    # Define the inputs needed for this component
    inputs = [
        SecretStrInput(name="api_key", display_name="News API Key", required=True),
        MessageTextInput(name="query", display_name="Search Query", required=True),
        IntInput(name="page_size", display_name="Number of Results", value=10, required=False),
        MessageTextInput(name="language", display_name="Language", value="en", required=False),
        MessageTextInput(name="sort_by", display_name="Sort By", value="publishedAt", required=False),
    ]

    # Define the schema for the API tool arguments
    class NewsAPISchema(BaseModel):
        api_key: str = Field(..., description="Your News API Key")
        query: str = Field(..., description="The search query to look for in articles")
        page_size: int = Field(10, description="Number of articles to return")
        language: str = Field("en", description="The language of the articles (default is 'en')")
        sort_by: str = Field("publishedAt", description="Sort by relevancy, popularity, or published date (default is 'publishedAt')")

    # Define the API Wrapper for News API
    class NewsAPIWrapper:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.url = "https://newsapi.org/v2/everything"
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
            }

        def get_news(self, query: str, page_size: int = 10, language: str = "en", sort_by: str = "publishedAt") -> list[dict[str, Any]]:
            params = {
                "q": query,
                "pageSize": page_size,
                "language": language,
                "sortBy": sort_by,
            }
            response = requests.get(self.url, headers=self.headers, params=params)

            if response.status_code == 401:
                raise Exception("Error 401: Unauthorized. Please check your API key.")
            elif response.status_code != 200:
                raise Exception(f"Error {response.status_code}: {response.json().get('message', 'Unknown error')}")

            return response.json().get('articles', [])

    # Build the API wrapper
    def _build_wrapper(self, api_key: str):
        return self.NewsAPIWrapper(api_key=api_key)

    # Tool builder function
    def build_tool(self) -> Tool:
        def get_news(api_key: str, query: str, page_size: int = 10, language: str = "en", sort_by: str = "publishedAt") -> list[dict[str, Any]]:
            wrapper = self._build_wrapper(api_key=api_key)
            return wrapper.get_news(query=query, page_size=page_size, language=language, sort_by=sort_by)

        tool = StructuredTool.from_function(
            name="news_api",
            description="Fetch the latest news articles using the News API",
            func=get_news,
            args_schema=self.NewsAPISchema,
        )

        self.status = f"News API Tool created successfully with API Key."
        return tool

    # Run model function to trigger the API call
    def run_model(self) -> list[Data]:
        tool = self.build_tool()
        # The API key and other inputs are passed dynamically
        results = tool.run(
            {
                "api_key": self.api_key,
                "query": self.query,
                "page_size": self.page_size,
                "language": self.language,
                "sort_by": self.sort_by,
            }
        )

        # Format the results for output
        data_list = [Data(data=result, text=f"Title: {result['title']}, Description: {result['description']}, URL: {result['url']}") for result in results]

        self.status = data_list
        return data_list
