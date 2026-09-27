"""One place to build the Gemini client, so retry behaviour is set consistently."""

from google import genai
from google.genai import types

from obrag.config import Settings


def http_options_for(settings: Settings) -> types.HttpOptions | None:
    """HTTP options for the client: None keeps the SDK's default retries.

    Batch work (the eval) wants the SDK to wait out 429s. The demo does not: on
    an exhausted free-tier quota the retries hold a visitor for about a minute
    before any message appears, so the app sets gemini_retries=0.

    The SDK passes HttpRetryOptions.attempts through as the number of *retries*,
    and each retry waits out the server's Retry-After (~59s on a quota 429), so
    attempts=1 still meant a minute's wait. 0 means fail on the first error.
    """
    if settings.gemini_retries is None:
        return None
    return types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=settings.gemini_retries))


def client_for(settings: Settings) -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key, http_options=http_options_for(settings))
