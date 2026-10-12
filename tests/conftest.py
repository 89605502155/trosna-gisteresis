"""Общие настройки тестов.

Hypothesis работает детерминированно (``derandomize=True``, без базы примеров), поэтому
прогон воспроизводится побайтно на любой машине. Для поиска новых контрпримеров:
``HYPOTHESIS_PROFILE=explore uv run pytest``.
"""

import os

from hypothesis import HealthCheck, settings

settings.register_profile(
    "default",
    derandomize=True,
    database=None,
    max_examples=200,
    deadline=None,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("explore", max_examples=5000, deadline=None, print_blob=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
