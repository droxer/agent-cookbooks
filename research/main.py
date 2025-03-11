from deepsearcher.configuration import Configuration, init_config
from deepsearcher.online_query import query
from icecream import ic

from dotenv import load_dotenv

load_dotenv()

config = Configuration()

# Customize your config here,
# more configuration see the Configuration Details section below.
config.set_provider_config("llm", "OpenAI", {"model": "gpt-4o-mini"})
config.set_provider_config("embedding", "OpenAIEmbedding", {"model": "text-embedding-ada-002"})
init_config(config = config)

# Load your local data
# from deepsearcher.offline_loading import load_from_local_files
# load_from_local_files(paths_or_directory="research/data")

# (Optional) Load from web crawling (`FIRECRAWL_API_KEY` env variable required)
from deepsearcher.offline_loading import load_from_website
load_from_website(urls="https://www.wikiwand.com/en/articles/DeepSeek")

# Query
result = query("introduce deepseek.") # Your question here

ic(result)