from typing import Any

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from langflow.base.langchain_utilities.model import LCToolComponent
from langflow.field_typing import Tool
from langflow.inputs import IntInput, MessageTextInput, SecretStrInput
from langflow.schema import Data
import requests


# Define the component class for Google Serper API
class GoogleSerperAPIComponent(LCToolComponent):
    display_name: str = "Google Serper API"
    description: str = "Retrieve search results using the Google Serper API."
    name = "GoogleSerperAPI"
    documentation: str = "https://serpapi.com/docs/"
    
        # Define the inputs needed for this component
    inputs = [
        SecretStrInput(name="api_key", display_name="Serper API Key", required=True),
        MessageTextInput(name="query", display_name="Search Query"),
        IntInput(name="num_results", display_name="Number of Results", value=4, required=True),
        MessageTextInput(name="gl", display_name="Geographical Location (gl)", value="us"),
        MessageTextInput(name="hl", display_name="Language (hl)", value="en"),
    ]
    
        # Define the schema for the API tool arguments
    class GoogleSerperSchema(BaseModel):
        api_key: str = Field(..., description="Google Serper API Key")
        query: str = Field(..., description="The search query string")
        num_results: int = Field(4, description="Number of search results to retrieve")
        gl: str = Field("us", description="Geographical location (e.g., 'us' for United States)")
        hl: str = Field("en", description="Language for the search (e.g., 'en' for English)")
    
    
    # Define the API Wrapper for Google Serper API
    class GoogleSerperAPIWrapper:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.url = "https://google.serper.dev/search"
            self.headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",  # API key passed in the headers
            }
    
        def get_search_results(self, query: str, num_results: int = 4, gl: str = "us", hl: str = "en") -> list[dict[str, Any]]:
            params = {"q": query, "gl": gl, "hl": hl, "num": num_results}
            response = requests.get(self.url, headers=self.headers, params=params)
            
            if response.status_code == 403:
                raise Exception("Error 403: Forbidden. Please check your API key or request quota.")
            elif response.status_code != 200:
                raise Exception(f"Error {response.status_code}: {response.json().get('error', 'Unknown error')}")
    
            return response.json().get('organic', [])  # Extract organic search results

    # Build the API wrapper
    def _build_wrapper(self, api_key: str):
        # Pass the api_key dynamically when building the wrapper
        return self.GoogleSerperAPIWrapper(api_key=api_key)

    # Tool builder function
    def build_tool(self) -> Tool:
        #wrapper = self._build_wrapper()

        def get_search_results(api_key: str, query: str, num_results: int = 4, gl: str = "us", hl: str = "en") -> list[dict[str, Any]]:
            wrapper = self._build_wrapper(api_key=api_key)  # Pass API key dynamically
            return wrapper.get_search_results(query=query, num_results=num_results, gl=gl, hl=hl)

        tool = StructuredTool.from_function(
            name="google_serper_api",
            description="Fetch search results using Google Serper API",
            func=get_search_results,
            args_schema=self.GoogleSerperSchema,  # Ensure the schema is correctly referenced here
        )

        self.status = f"Google Serper API Tool created with API Key."
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
