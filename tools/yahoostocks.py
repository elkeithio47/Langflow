import ast
import pprint

import yfinance as yf
from langchain.tools import StructuredTool
from loguru import logger
from pydantic import BaseModel, Field

from langflow.base.langchain_utilities.model import LCToolComponent
from langflow.field_typing import Tool
from langflow.inputs import DropdownInput, IntInput, MessageTextInput
from langflow.schema import Data


class YfinanceToolComponent(LCToolComponent):
    display_name = "Yahoo Finance Tool"
    description = "Access financial data and market information using Yahoo Finance."
    icon = "trending-up"
    name = "YahooFinanceTool"

    # Define the inputs required for the component
    inputs = [
        MessageTextInput(
            name="symbol",
            display_name="Stock Symbol",
            info="The stock symbol to retrieve data for (e.g., AAPL, GOOG).",
            required=True,
        ),
        DropdownInput(
            name="method",
            display_name="Data Method",
            info="The type of data to retrieve.",
            options=[
                "get_actions",
                "get_analysis",
                "get_balance_sheet",
                "get_calendar",
                "get_cashflow",
                "get_info",
                "get_institutional_holders",
                "get_news",
                "get_recommendations",
                "get_sustainability",
            ],
            value="get_news",
        ),
        IntInput(
            name="num_news",
            display_name="Number of News",
            info="The number of news articles to retrieve (only applicable for get_news).",
            value=5,
        ),
    ]

    # Define the schema for the API tool arguments
    class YahooFinanceSchema(BaseModel):
        symbol: str = Field(..., description="The stock symbol to retrieve data for.")
        method: str = Field("get_info", description="The type of data to retrieve.")
        num_news: int | None = Field(5, description="The number of news articles to retrieve.")

    # Correct run_model to properly retrieve and pass arguments to _yahoo_finance_tool
    def run_model(self, inputs: Dict[str, Any]) -> list[Data]:
        # Retrieve inputs from the schema inputs argument
        symbol = inputs.get("symbol")
        method = inputs.get("method", "get_info")  # Default to get_info if method not provided
        num_news = inputs.get("num_news", 5)

        # Pass the retrieved values into _yahoo_finance_tool
        return self._yahoo_finance_tool(
            symbol=symbol,
            method=method,
            num_news=num_news,
        )

    def build_tool(self) -> Tool:
        return StructuredTool.from_function(
            name="yahoo_finance",
            description="Access financial data and market information from Yahoo Finance.",
            func=self._yahoo_finance_tool,
            args_schema=self.YahooFinanceSchema,
        )

    def _yahoo_finance_tool(
        self,
        symbol: str,
        method: str,
        num_news: int | None = 5,
    ) -> list[Data]:
        # Initialize the Yahoo Finance Ticker object
        ticker = yf.Ticker(symbol)

        try:
            # Select the correct method from the ticker based on the user's input
            if method == "get_info":
                result = ticker.info
            elif method == "get_news":
                result = ticker.news[:num_news]
            else:
                result = getattr(ticker, method)()

            # Format the result for output
            result = pprint.pformat(result)

            if method == "get_news":
                data_list = [Data(data=article) for article in ast.literal_eval(result)]
            else:
                data_list = [Data(data={"result": result})]

        except Exception as e:
            error_message = f"Error retrieving data: {e}"
            logger.opt(exception=True).debug(error_message)
            self.status = error_message
            return [Data(data={"error": error_message})]

        return data_list
