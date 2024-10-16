from typing import Any, List, Dict
from pydantic import BaseModel, Field
from langflow.base.langchain_utilities.model import LCToolComponent
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
        MessageTextInput(name="from_date", display_name="From Date (YYYY-MM-DD)", required=False),
        MessageTextInput(name="to_date", display_name="To Date (YYYY-MM-DD)", required=False),
        MessageTextInput(name="sources", display_name="News Sources", required=False),
        MessageTextInput(name="domains", display_name="Domains", required=False),
    ]

    # Define the schema for the API tool arguments
    class NewsAPISchema(BaseModel):
        api_key: str = Field(..., description="Your News API Key")
        query: str = Field(..., description="The search query to look for in articles")
        page_size: int = Field(10, description="Number of articles to return")
        language: str = Field("en", description="The language of the articles (default is 'en')")
        sort_by: str = Field("publishedAt", description="Sort by relevancy, popularity, or published date (default is 'publishedAt')")
        from_date: str = Field(None, description="Start date for the news articles (YYYY-MM-DD)")
        to_date: str = Field(None, description="End date for the news articles (YYYY-MM-DD)")
        sources: str = Field(None, description="Comma-separated list of news sources (e.g., 'bbc-news, techcrunch')")
        domains: str = Field(None, description="Comma-separated list of domains (e.g., 'wsj.com, bbc.co.uk')")

    # Define the API Wrapper for News API
    class NewsAPIWrapper:
        BASE_URL = "https://newsapi.org/v2/everything"

        def __init__(self, api_key: str):
            self.api_key = api_key

        def get_news(self, query: str, page_size: int = 10, language: str = "en", sort_by: str = "publishedAt", from_date: str = None, to_date: str = None, sources: str = None, domains: str = None) -> List[Dict[str, Any]]:
            # Prepare request parameters based on inputs
            params = {
                "q": query,
                "pageSize": page_size,
                "language": language,
                "sortBy": sort_by,
                "from": from_date,
                "to": to_date,
                "sources": sources,
                "domains": domains,
                "apiKey": "32e88604cf474c199dea193cccd63e96"  # self.api_key,  # Use the provided API key dynamically
            }

            # Remove None values from params
            params = {k: v for k, v in params.items() if v is not None}

            # Print out the details of the API request
            print("\n[INFO] - Sending request to News API...")
            print(f"[INFO] - API Endpoint: {self.BASE_URL}")
            print(f"[INFO] - Request Parameters: {params}")

            response = requests.get(self.BASE_URL, params=params)

            # Handle common errors gracefully
            if response.status_code == 401:
                print("[ERROR] - Unauthorized request. Check your API key.")
                raise Exception("Error 401: Unauthorized. Please check your API key.")
            elif response.status_code == 429:
                print("[ERROR] - Rate limit exceeded.")
                raise Exception("Error 429: Too many requests. Rate limit exceeded.")
            elif response.status_code != 200:
                error_message = response.json().get('message', 'Unknown error')
                print(f"[ERROR] - Error {response.status_code}: {error_message}")
                raise Exception(f"Error {response.status_code}: {error_message}")

            print("[INFO] - Request successful. Processing response data...")
            return response.json().get('articles', [])

    # Build the API wrapper
    def _build_wrapper(self, api_key: str):
        print(f"[INFO] - Building API wrapper with API Key: {api_key[:5]}***")
        return self.NewsAPIWrapper(api_key=api_key)

    # Tool builder function
    def build_tool(self) -> Tool:
        def get_news(api_key: str, query: str, page_size: int = 10, language: str = "en", sort_by: str = "publishedAt", from_date: str = None, to_date: str = None, sources: str = None, domains: str = None) -> List[Dict[str, Any]]:
            wrapper = self._build_wrapper(api_key=api_key)
            return wrapper.get_news(query=query, page_size=page_size, language=language, sort_by=sort_by, from_date=from_date, to_date=to_date, sources=sources, domains=domains)

        tool = StructuredTool.from_function(
            name="news_api",
            description="Fetch the latest news articles using the News API",
            func=get_news,
            args_schema=self.NewsAPISchema,
        )

        print("[INFO] - News API Tool created successfully.")
        return tool

    # Run model function to trigger the API call
    def run_model(self, inputs: Dict[str, Any]) -> List[Data]:
        tool = self.build_tool()

        # Print the inputs received
        print("[INFO] - Running model with the following inputs:")
        for key, value in inputs.items():
            print(f"  {key}: {value}")

        # The API key and other inputs are passed dynamically
        results = tool.run(
            {
                "api_key": inputs["api_key"],
                "query": inputs["query"],
                "page_size": inputs.get("page_size", 10),
                "language": inputs.get("language", "en"),
                "sort_by": inputs.get("sort_by", "publishedAt"),
                "from_date": inputs.get("from_date", None),
                "to_date": inputs.get("to_date", None),
                "sources": inputs.get("sources", None),
                "domains": inputs.get("domains", None),
            }
        )

        # Format the results for output
        data_list = [
            Data(
                data=result,
                text=f"Title: {result.get('title', 'N/A')}, Description: {result.get('description', 'N/A')}, URL: {result.get('url', 'N/A')}"
            )
            for result in results
        ]

        print("[INFO] - Finished processing the News API response.")
        
        # Print out a cURL command for manual testing
        print("\n[INFO] - cURL Command for manual test:")
        curl_command = (
            f"curl -X GET 'https://newsapi.org/v2/everything?"
            f"q={inputs['query']}&"
            f"pageSize={inputs.get('page_size', 10)}&"
            f"language={inputs.get('language', 'en')}&"
            f"sortBy={inputs.get('sort_by', 'publishedAt')}&"
            f"from={inputs.get('from_date', '')}&"
            f"to={inputs.get('to_date', '')}&"
            f"apiKey={inputs['api_key']}'"
        )
        print(curl_command)

        self.status = data_list
        return data_list
