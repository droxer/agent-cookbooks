from os import environ
from langfuse.callback import CallbackHandler


langfuse_public_key = environ.get("LANGFUSE_PUBLIC_KEY")
langfuse_secret_key = environ.get("LANGFUSE_SECRET_KEY")
langfuse_host = environ.get("LANGFUSE_HOST")


langfuse_callback_handler = CallbackHandler(secret_key=langfuse_secret_key, 
                                            public_key=langfuse_public_key, 
                                            host=langfuse_host)
