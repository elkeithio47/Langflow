from typing import Any, Dict, List
from pydantic import BaseModel, Field
from langflow.base.langchain_utilities.model import LCToolComponent
from langflow.inputs import MessageTextInput, SecretStrInput
from langflow.schema import Data
import praw

class RedditNewsComponent(LCToolComponent):
    display_name: str = "Reddit News Fetcher"
    description: str = "Fetch news posts from specific subreddits using Reddit API."
    name = "RedditNewsFetcher"
    documentation: str = "https://www.reddit.com/dev/api/"

    # Define the inputs needed for this component
    inputs = [
        SecretStrInput(name="client_id", display_name="Reddit Client ID", required=True),
        SecretStrInput(name="client_secret", display_name="Reddit Client Secret", required=True),
        MessageTextInput(name="user_agent", display_name="User Agent", required=True, value="my_user_agent"),
        MessageTextInput(name="subreddit", display_name="Subreddit", required=True, value="news"),
        MessageTextInput(name="filter", display_name="Post Filter (new/top/hot)", required=False, value="new"),
        MessageTextInput(name="limit", display_name="Number of Posts to Fetch", required=False, value="5"),
    ]

    # Define the schema for the API tool arguments
    class RedditNewsSchema(BaseModel):
        client_id: str = Field(..., description="Your Reddit API Client ID.")
        client_secret: str = Field(..., description="Your Reddit API Client Secret.")
        user_agent: str = Field(..., description="Your User Agent for Reddit.")
        subreddit: str = Field(..., description="The name of the subreddit to fetch news from (e.g., 'news').")
        filter: str = Field("new", description="Filter for posts (e.g., 'new', 'top', 'hot').")
        limit: int = Field(5, description="Number of posts to retrieve.")

    # Define the API Wrapper for Reddit using PRAW
    class RedditAPIWrapper:
        def __init__(self, client_id: str, client_secret: str, user_agent: str):
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )

        def fetch_posts(self, subreddit: str, filter: str = "new", limit: int = 5) -> List[Dict[str, Any]]:
            subreddit_obj = self.reddit.subreddit(subreddit)

            if filter == "new":
                posts = subreddit_obj.new(limit=limit)
            elif filter == "top":
                posts = subreddit_obj.top(limit=limit)
            elif filter == "hot":
                posts = subreddit_obj.hot(limit=limit)
            else:
                raise ValueError(f"Invalid filter '{filter}'. Choose from 'new', 'top', or 'hot'.")

            return [{"title": post.title, "url": post.url, "author": post.author.name} for post in posts]

    # Build the API wrapper
    def _build_wrapper(self, client_id: str, client_secret: str, user_agent: str):
        return self.RedditAPIWrapper(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

    # Tool builder function
    def build_tool(self)->:
        def fetch_news(client_id: str, client_secret: str, user_agent: str, subreddit: str, filter: str = "new", limit: int = 5) -> List[Dict[str, Any]]:
            wrapper = self._build_wrapper(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
            return wrapper.fetch_posts(subreddit=subreddit, filter=filter, limit=limit)

        tool = StructuredTool.from_function(
            name="reddit_news_fetcher",
            description="Fetch news posts from specific subreddits using the Reddit API.",
            func=fetch_news,
            args_schema=self.RedditNewsSchema,
        )

        self.status = "Reddit News Fetcher Tool created successfully."
        return tool

    # Run model function to trigger the API call
    def run_model(self) -> list[Data]:
        tool = self.build_tool()

        # Fetch the Reddit posts based on user inputs
        results = tool.run(
            {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "user_agent": self.user_agent,
                "subreddit": self.subreddit,
                "filter": self.filter,
                "limit": int(self.limit),
            }
        )

        # Format the results for output
        formatted_results = [
            Data(data=post, text=f"Title: {post['title']}, Author: {post['author']}, URL: {post['url']}")
            for post in results
        ]

        self.status = formatted_results
        return formatted_results
