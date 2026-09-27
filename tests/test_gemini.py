from obrag.config import Settings
from obrag.gemini import http_options_for


def make_settings(**kw):
    return Settings(gemini_api_key="ai-x", voyage_api_key="pa-x", **kw)


def test_by_default_the_sdk_keeps_its_own_retries():
    assert http_options_for(make_settings()) is None


def test_zero_retries_is_passed_to_the_sdk():
    # The demo sets this: on an exhausted free-tier quota the SDK otherwise waits
    # out 429s for about a minute before the visitor sees any message.
    # The SDK reads HttpRetryOptions.attempts as a retry count, so 0 = no retries.
    options = http_options_for(make_settings(gemini_retries=0))
    assert options.retry_options.attempts == 0
