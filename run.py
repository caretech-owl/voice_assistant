import json
import logging
import sys

from assistant.llm import LLM
from assistant.stt import STT
from assistant.tts import TTS_COCQUI, TTS_TF, TTS_Suno
from assistant.wake_word import WakeWord

root = logging.getLogger()
root.setLevel(logging.DEBUG)
root.addHandler(logging.StreamHandler(sys.stdout))

with open("config/config.json", "r") as f:
    config = json.load(f)


ww = WakeWord(**config["wakeword"])
stt = STT(**config["stt"])
llm = LLM(**config["llm"])
if config["tts"]["class"] == "TTS_TF":
    tts = TTS_TF(**config["tts"])
elif config["tts"]["class"] == "TTS_Suno":
    tts = TTS_Suno(**config["tts"])
elif config["tts"]["class"] == "TTS_COCQUI":
    tts = TTS_COCQUI(**config["tts"])


def check_keywords(input: str) -> bool:
    """
    Checks if the input contains certain keywords and performs corresponding actions.

    Args:
        input (str): The input string to be checked.

    Returns:
        bool: True if the input contains a keyword and an action is performed, False otherwise.
    """
    if "reset" in input.lower():
        llm.reset()
        return True
    if "exit" in input.lower():
        sys.exit(0)
    return False


while True:
    ww()
    while True:
        input = stt()
        if check_keywords(input):
            break
        if input == "":
            continue
        response = llm(input)
        tts(response)
