from __future__ import annotations

import os
from functools import lru_cache

import braintrust


@lru_cache(maxsize=1)
def braintrust_logger():
    """Configure Braintrust once, while keeping unconfigured environments usable."""
    api_key = os.getenv("BRAINTRUST_API_KEY")
    if not api_key:
        return None

    # The SDK reads the data-plane URL from BRAINTRUST_API_URL.
    os.environ.setdefault("BRAINTRUST_API_URL", "https://api.braintrust.dev")
    return braintrust.init_logger(
        api_key=api_key,
        app_url=os.getenv("BRAINTRUST_APP_URL", "https://www.braintrust.dev"),
        project_id=os.getenv(
            "BRAINTRUST_PROJECT_ID", "1ceada4e-d99a-4aa9-9d69-682187c7f6d0"
        ),
    )
