from typing import Any

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from langflow.base.langchain_utilities.model import LCToolComponent
from langflow.field_typing import Tool
from langflow.inputs import MessageTextInput, SecretStrInput
from langflow.schema import Data
import yfinance as yf


# Define the component class for Yahoo Finance Crypto API
class YahooFinanceCryptoComponent(LCToolComponent):
    display_name: str = "Yahoo Finance Crypto"
    description: str = "Retrieve cryptocurrency and blockchain-related data using Yahoo Finance."
    name = "YahooFinanceCrypto"
    documentation: str = "https://www.yahoofinanceapi.com/"

    # Define the inputs needed for this component
    inputs = [
        MessageTextInput(name="crypto_symbol", display_name="Cryptocurrency Symbol", required=True),
        MessageTextInput(name="metric", display_name="Financial Metric", required=False, value="summary"),
    ]

    # Define the schema for the API tool arguments
    class YahooFinanceCryptoSchema(BaseModel):
        crypto_symbol: str = Field(..., description="The cryptocurrency symbol (e.g., BTC-USD for Bitcoin).")
        metric: str = Field("summary", description="The financial metric to retrieve (e.g., 'summary', 'price').")

    # Define the API Wrapper for Yahoo Finance
    class YahooFinanceCryptoWrapper:
        def __init__(self, crypto_symbol: str, metric: str = "summary"):
            self.crypto_symbol = crypto_symbol
            self.metric = metric
            self.ticker = yf.Ticker(crypto_symbol)  # Initialize with the cryptocurrency symbol

        def get_crypto_data(self) -> dict[str, Any]:
            # Based on the metric, retrieve the corresponding financial data
            if self.metric == "summary":
                return self.ticker.info  # Fetch general summary info
            elif self.metric == "price":
                return {"price": self.ticker.history(period="1d")["Close"].iloc[-1]}
            elif self.metric == "history":
                return self.ticker.history(period="1mo")  # Fetch price history
            else:
                raise ValueError("Invalid metric. Please choose 'summary', 'price', or 'history'.")

    # Build the API wrapper
    def _build_wrapper(self, crypto_symbol: str, metric: str):
        return self.YahooFinanceCryptoWrapper(crypto_symbol=crypto_symbol, metric=metric)

    # Tool builder function
    def build_tool(self) -> Tool:
        def get_crypto_data(crypto_symbol: str, metric: str = "summary") -> dict[str, Any]:
            wrapper = self._build_wrapper(crypto_symbol=crypto_symbol, metric=metric)
            return wrapper.get_crypto_data()

        tool = StructuredTool.from_function(
            name="yahoo_finance_crypto",
            description="Fetch cryptocurrency data using Yahoo Finance.",
            func=get_crypto_data,
            args_schema=self.YahooFinanceCryptoSchema,
        )

        self.status = "Yahoo Finance Crypto Tool created successfully."
        return tool

    # Run model function to trigger the API call
    def run_model(self) -> list[Data]:
        tool = self.build_tool()
        # Retrieve and pass all necessary inputs
        results = tool.run(
            {
                "crypto_symbol": self.crypto_symbol,
                "metric": self.metric,
            }
        )

        # Format the results for output
        if self.metric == "summary":
            formatted_results = [Data(data=results, text=f"Summary: {results}")]
        elif self.metric == "price":
            formatted_results = [Data(data=results, text=f"Latest Price: {results['price']}")]
        elif self.metric == "history":
            formatted_results = [
                Data(data=results, text=f"Price History: {results[['Close']].to_dict(orient='list')}")
            ]
        else:
            formatted_results = [Data(data=results, text="No valid data found.")]

        self.status = formatted_results
        return formatted_results
