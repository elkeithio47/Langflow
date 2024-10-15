"""
from langchain_core.tools import Tool

from langflow.base.langchain_utilities.model import LCToolComponent
from langflow.inputs import IntInput, MultilineInput, SecretStrInput
from langflow.schema import Data
"""


from typing import Any

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from langflow.base.langchain_utilities.model import LCToolComponent
from langflow.field_typing import Tool
from langflow.inputs import DictInput, IntInput, MultilineInput, SecretStrInput
from langflow.schema import Data
import requests




class CoinMarketCapAPIComponent(LCToolComponent):
    display_name: str = "CoinMarketCap API"
    description: str = "Retrieve latest cryptocurrency data from CoinMarketCap."
    name = "CoinMarketCapAPI"
    documentation: str = "https://coinmarketcap.com/api/documentation/v1/"

    class CoinMarketCapAPIWrapper:
        """Wrapper class to call CoinMarketCap API and retrieve cryptocurrency data."""
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
            self.headers = {
                "Accepts": "application/json",
                "X-CMC_PRO_API_KEY": self.api_key,
            }
    
        def get_cryptocurrency_data(self, start: int = 1, limit: int = 10) -> list[dict[str, Any]]:
            parameters = {"start": start, "limit": limit, "convert": "USD"}
            response = requests.get(self.url, headers=self.headers, params=parameters)
            data = response.json()
    
            if response.status_code != 200:
                raise Exception(f"Error {response.status_code}: {data.get('status', {}).get('error_message', 'Unknown error')}")
    
            return data["data"]


    """
    inputs = [
        SecretStrInput(name="google_api_key", display_name="Google API Key", required=True),
        SecretStrInput(name="google_cse_id", display_name="Google CSE ID", required=True),
        MultilineInput(
            name="input_value",
            display_name="Input",
        ),
        IntInput(name="k", display_name="Number of results", value=4, required=True),
    ]
    """
    """
    # Define the inputs needed for this component
    inputs = [
        SecretStrInput(name="api_key", display_name="CoinMarketCap API Key", required=True),
        IntInput(name="start", display_name="Start", value=1, description="The rank from which to start the listing"),
        IntInput(name="limit", display_name="Limit", value=10, description="The number of cryptocurrencies to retrieve"),
    ]
    
    """
    
        # Define the inputs needed for this component
    inputs = [
        SecretStrInput(name="api_key", display_name="CoinMarketCap API Key", required=True),
        IntInput(name="start", display_name="Start", value=1 , required=True),
        IntInput(name="limit", display_name="Limit", value=10, required=True),
    ]
    
        # Define the schema for the API tool arguments
    class CoinMarketCapSchema(BaseModel):
        start: int = Field(1, description="The rank from which to start the listing")
        limit: int = Field(10, description="The number of cryptocurrencies to retrieve")
    
    
    # Helper function to build the API wrapper
    def _build_wrapper(self):
        return CoinMarketCapAPIWrapper(api_key=self.api_key)

    # Tool builder function
    def build_tool(self) -> Tool:
        wrapper = self._build_wrapper()

        def get_crypto_data(
            start: int = 1, limit: int = 10
        ) -> list[dict[str, Any]]:
            # Call the wrapper function to get cryptocurrency data
            return wrapper.get_cryptocurrency_data(start=start, limit=limit)

        # Return the StructuredTool
        tool = StructuredTool.from_function(
            name="coinmarketcap_api",
            description="Fetch latest cryptocurrency data from CoinMarketCap API",
            func=get_crypto_data,
            args_schema=self.CoinMarketCapSchema,
        )

        self.status = f"CoinMarketCap API Tool created with API Key."
        return tool

    # Run model function to trigger the API call
    def run_model(self) -> list[Data]:
        tool = self.build_tool()
        results = tool.run(
            {
                "start": self.start,
                "limit": self.limit,
            }
        )

        data_list = [Data(data=result, text=f"Name: {result['name']}, Price: {result['quote']['USD']['price']}") for result in results]

        self.status = data_list
        return data_list    
    
    
    
    """
    def run_model(self) -> Data | list[Data]:
        wrapper = self._build_wrapper()
        results = wrapper.results(query=self.input_value, num_results=self.k)
        data = [Data(data=result, text=result["snippet"]) for result in results]
        self.status = data
        return data

    def build_tool(self) -> Tool:
        wrapper = self._build_wrapper()
        return Tool(
            name="google_search",
            description="Search Google for recent results.",
            func=wrapper.run,
        )

    def _build_wrapper(self):
        try:
            from langchain_google_community import GoogleSearchAPIWrapper
        except ImportError as e:
            msg = "Please install langchain-google-community to use GoogleSearchAPIWrapper."
            raise ImportError(msg) from e
        return GoogleSearchAPIWrapper(google_api_key=self.google_api_key, google_cse_id=self.google_cse_id, k=self.k)
    """