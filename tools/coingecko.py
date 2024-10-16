from typing import Any, Dict
from pydantic import BaseModel, Field
from langflow.base.langchain_utilities.model import LCToolComponent
from langflow.inputs import MessageTextInput
from langflow.schema import Data
import requests

class CoinGeckoComponent(LCToolComponent):
    display_name: str = "CoinGecko API"
    description: str = "Retrieve cryptocurrency data using the CoinGecko API."
    name = "CoinGeckoAPI"
    documentation: str = "https://www.coingecko.com/en/api"

    # Define the inputs needed for this component
    inputs = [
        MessageTextInput(name="crypto_id", display_name="Cryptocurrency ID", required=True),
        MessageTextInput(name="currency", display_name="Fiat Currency", required=False, value="usd"),
        MessageTextInput(name="metric", display_name="Metric Type", required=False, value="price"),
    ]

    # Define the schema for the API tool arguments
    class CoinGeckoSchema(BaseModel):
        crypto_id: str = Field(..., description="The ID of the cryptocurrency (e.g., 'bitcoin', 'ethereum').")
        currency: str = Field("usd", description="The fiat currency to convert the price into (default is 'usd').")
        metric: str = Field("price", description="The type of metric to retrieve (e.g., 'price', 'market_cap').")

    # Define the API Wrapper for CoinGecko
    class CoinGeckoAPIWrapper:
        BASE_URL = "https://api.coingecko.com/api/v3"

        def __init__(self, crypto_id: str, currency: str = "usd", metric: str = "price"):
            self.crypto_id = crypto_id
            self.currency = currency
            self.metric = metric

        def get_crypto_data(self) -> Dict[str, Any]:
            url = f"{self.BASE_URL}/simple/price"
            params = {
                "ids": self.crypto_id,
                "vs_currencies": self.currency,
                "include_market_cap": "true" if self.metric == "market_cap" else "false",
                "include_24hr_vol": "true" if self.metric == "volume" else "false",
                "include_24hr_change": "true" if self.metric == "price_change" else "false",
            }

            try:
                response = requests.get(url, params=params)

                if response.status_code == 429:
                    # Rate limiting occurred, return a safe message without throwing an exception
                    print("Rate limit exceeded. Please try again later.")
                    return {"error": "Rate limit exceeded"}

                if response.status_code != 200:
                    raise Exception(f"Error {response.status_code}: {response.text}")

                data = response.json()

                if self.crypto_id not in data:
                    raise ValueError(f"Cryptocurrency '{self.crypto_id}' not found.")

                return data[self.crypto_id]

            except Exception as e:
                # Catch any other exceptions and handle them gracefully
                print(f"An error occurred: {e}")
                return {"error": str(e)}

    # Build the API wrapper
    def _build_wrapper(self, crypto_id: str, currency: str, metric: str):
        return self.CoinGeckoAPIWrapper(crypto_id=crypto_id, currency=currency, metric=metric)

    # Tool builder function
    def build_tool(self) -> Tool:
        def get_crypto_data(crypto_id: str, currency: str = "usd", metric: str = "price") -> Dict[str, Any]:
            wrapper = self._build_wrapper(crypto_id=crypto_id, currency=currency, metric=metric)
            return wrapper.get_crypto_data()

        tool = StructuredTool.from_function(
            name="coingecko_api",
            description="Fetch cryptocurrency data using the CoinGecko API.",
            func=get_crypto_data,
            args_schema=self.CoinGeckoSchema,
        )

        self.status = "CoinGecko API Tool created successfully."
        return tool

    # Run model function to trigger the API call
    def run_model(self) -> list[Data]:
        tool = self.build_tool()

        # Fetch the CoinGecko cryptocurrency data based on user inputs
        results = tool.run(
            {
                "crypto_id": self.crypto_id,
                "currency": self.currency,
                "metric": self.metric,
            }
        )

        # Handle rate limit or errors gracefully
        if "error" in results:
            return [Data(data=results, text=results["error"])]

        # Format the results for output
        if self.metric == "price":
            formatted_result = Data(data=results, text=f"Price: {results[self.currency]}")
        elif self.metric == "market_cap":
            formatted_result = Data(data=results, text=f"Market Cap: {results[f'{self.currency}_market_cap']}")
        elif self.metric == "volume":
            formatted_result = Data(data=results, text=f"24h Volume: {results[f'{self.currency}_24h_vol']}")
        elif self.metric == "price_change":
            formatted_result = Data(data=results, text=f"24h Price Change: {results[f'{self.currency}_24h_change']}")
        else:
            formatted_result = Data(data=results, text="No valid data found.")

        self.status = [formatted_result]
        return [formatted_result]
