import pytest
import time

from assistant.config import CONFIG
from assistant.wake_word import WakeWord

import logging

_LOGGER = logging.getLogger(__name__)


@pytest.mark.skip("Should only be run manually")
def test_fail() -> None:
    pytest.fail("This test should not be executed.")

# run with 
# pytest tests/test_wakeword.py  -o log_cli=true -o log_cli_level=DEBUG -k test_ww_mic_detection
@pytest.mark.skip("Should only be run manually")
def test_ww_mic_detection() -> None:
    ww = WakeWord(**CONFIG.wakeword.model_dump())
    start_time = time.time()
    target = 5
    current = 0
    while current > target:
        _LOGGER.info("Waiting for wake word...")
        ww()
        _LOGGER.info("Wake word detected. %d to go...", target - current)
    _LOGGER.info("Wake word detected %d times in %d seconds", target, time.time() - start_time)
