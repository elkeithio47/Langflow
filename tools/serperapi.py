from typing import Any
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from langflow.base.langchain_utilities.model import LCToolComponent
from langflow.field_typing import Tool
from langflow.inputs import IntInput, MessageTextInput, SecretStrInput
from langflow.schema import Data
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define the component class for SerpApi
class GoogleSerpAPIComponent(LCToolComponent):
    display_name: str = "Google Serp API"
    description: str = "Retrieve search results using the Google Serp API."
    name = "GoogleSerpAPI"
    documentation: str = "https://serpapi.com/docs/"

    # Define the inputs needed for this component
    inputs = [
        SecretStrInput(name="api_key", display_name="SerpApi API Key", required=True),
        MessageTextInput(name="query", display_name="Search Query", required=True),
        IntInput(name="num_results", display_name="Number of Results", value=10, required=True),
        MessageTextInput(name="gl", display_name="Geographical Location (gl)", value="us"),
        MessageTextInput(name="hl", display_name="Language (hl)", value="en"),
    ]

    # Define the schema for the API tool arguments
    class GoogleSerpAPISchema(BaseModel):
        api_key: str = Field(..., description="SerpApi API Key")
        query: str = Field(..., description="The search query string")
        num_results: int = Field(10, description="Number of search results to retrieve")
        gl: str = Field("us", description="Geographical location (e.g., 'us' for United States)")
        hl: str = Field("en", description="Language for the search (e.g., 'en' for English)")

    # Define the API Wrapper for SerpApi
    class GoogleSerpAPIWrapper:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.url = "https://serpapi.com/search"
    
        def get_search_results(self, query: str, num_results: int = 10, gl: str = "us", hl: str = "en") -> list[dict[str, Any]]:
            params = {
                "q": query,
                "num": num_results,
                "gl": gl,
                "hl": hl,
                "engine": "google",
                "api_key": "key hard coded here" #self.api_key""
            }

            # Print the API key being used
            logging.debug(f"API Key: {self.api_key}")

            # Log the URL and parameters being sent to SerpApi
            logging.debug(f"Request URL: {self.url}")
            logging.debug(f"Request Params: {params}")

            response = requests.get(self.url, params=params)

            # Log the response status and content
            logging.debug(f"Response Status Code: {response.status_code}")
            logging.debug(f"Response Content: {response.text}")

            # Check for errors
            if response.status_code == 403:
                raise Exception("Error 403: Forbidden. Please check your API key or request quota.")
            elif response.status_code == 401:
                raise Exception("Error 401: Invalid API key. Please verify your API key.")
            elif response.status_code != 200:
                raise Exception(f"Error {response.status_code}: {response.json().get('error', 'Unknown error')}")

            # Return the organic results from the response
            return response.json().get('organic_results', [])

    # Build the API wrapper
    def _build_wrapper(self, api_key: str):
        return self.GoogleSerpAPIWrapper(api_key=api_key)

    # Tool builder function
    def build_tool(self) -> Tool:
        def get_search_results(api_key: str, query: str, num_results: int = 10, gl: str = "us", hl: str = "en") -> list[dict[str, Any]]:
            wrapper = self._build_wrapper(api_key=api_key)
            return wrapper.get_search_results(query=query, num_results=num_results, gl=gl, hl=hl)

        tool = StructuredTool.from_function(
            name="google_serp_api",
            description="Fetch search results using Google Serp API",
            func=get_search_results,
            args_schema=self.GoogleSerpAPISchema,
        )

        self.status = "Google Serp API Tool created successfully."
        return tool

    # Run model function to trigger the API call
    def run_model(self) -> list[Data]:
        tool = self.build_tool()
        results = tool.run(
            {
                "api_key": self.api_key,
                "query": self.query,
                "num_results": self.num_results,
                "gl": self.gl,
                "hl": self.hl,
            }
        )

        # Format the results for output
        data_list = [Data(data=result, text=f"Title: {result['title']}, Snippet: {result['snippet']}") for result in results]

        self.status = data_list
        return data_list
