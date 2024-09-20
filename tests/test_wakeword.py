import logging
import time
import pytest

from voice_assistant.wake_word import WakeWord

_LOGGER = logging.getLogger(__name__)

@pytest.mark.skip("Should only be run manually")
def test_fail() -> None:
    pytest.fail("This test should not be executed.")



# run with
# pytest tests/test_wakeword.py -k test_ww_manual -o log_cli=true -o log_cli_level=INFO --no-skip
@pytest.mark.skip("Should only be run manually")
def test_ww_manual() -> None:
    ww = WakeWord()
    current, target = 0, 3
    start_time = time.time()
    while current < target:
        _LOGGER.info("Waiting for wake word...")
        ww()
        _LOGGER.info("Wakeword detected! %d/%d", current + 1, target)
        current += 1
    _LOGGER.info("Wakeword detection test completed in %d seconds", time.time() - start_time)
