from typing import Any, Dict, List
from pydantic import BaseModel, Field
from langflow.base.langchain_utilities.model import LCToolComponent
from langflow.inputs import MessageTextInput
from langflow.schema import Data


class TradingViewComponent(LCToolComponent):
    display_name: str = "TradingView Chart"
    description: str = "Embed a TradingView chart for any financial asset."
    name = "TradingViewChart"
    documentation: str = "https://www.tradingview.com/widget/"

    # Define the inputs needed for this component
    inputs = [
        MessageTextInput(name="symbol", display_name="Symbol", required=True, value="BTCUSD"),
        MessageTextInput(name="interval", display_name="Chart Interval", required=False, value="D"),
        MessageTextInput(name="theme", display_name="Chart Theme", required=False, value="light"),
    ]

    # Define the schema for the input arguments
    class TradingViewSchema(BaseModel):
        symbol: str = Field(..., description="The trading symbol (e.g., 'BTCUSD' for Bitcoin).")
        interval: str = Field("D", description="The chart interval (e.g., 'D' for daily, '1' for 1-minute, etc.).")
        theme: str = Field("light", description="The chart theme, either 'light' or 'dark'.")

    # Function to generate the TradingView widget URL
    def generate_tradingview_chart(self, symbol: str, interval: str, theme: str) -> str:
        base_url = "https://s.tradingview.com/widgetembed/?"
        params = f"symbol={symbol}&interval={interval}&theme={theme}&style=1&timezone=Etc%2FUTC"
        widget_url = f"{base_url}{params}"
        return widget_url  # Ensure this method explicitly returns a string

    # Tool builder function with an explicit return type
    def build_tool(self) -> Tool:
        def get_tradingview_chart(symbol: str, interval: str = "D", theme: str = "light") -> Dict[str, Any]:
            # Generate the TradingView URL with the given parameters
            chart_url = self.generate_tradingview_chart(symbol=symbol, interval=interval, theme=theme)
            return {"chart_url": chart_url}  # Explicitly returning a dictionary

        tool = StructuredTool.from_function(
            name="tradingview_chart",
            description="Generate a TradingView chart embed URL for a specified symbol.",
            func=get_tradingview_chart,
            args_schema=self.TradingViewSchema,
        )

        self.status = "TradingView Chart Tool created successfully."
        return tool  # Return the tool explicitly

    # Run model function with an explicit return type
    def run_model(self) -> List[Data]:
        tool = self.build_tool()

        # Fetch the TradingView chart URL based on user inputs
        results = tool.run(
            {
                "symbol": self.symbol,
                "interval": self.interval,
                "theme": self.theme,
            }
        )

        # Format the results for output
        chart_url = results["chart_url"]
        formatted_result = Data(data=results, text=f"TradingView Chart URL: {chart_url}")

        self.status = [formatted_result]
        return [formatted_result]  # Explicitly return a list of Data objects
