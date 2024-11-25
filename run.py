import logging
import sys

from voice_assistant.stt import STT
from voice_assistant.tts import TTS_COCQUI, TTS_TF, TTS_Suno
from voice_assistant.wake_word import WakeWord
from voice_assistant.config import CONFIG

from gerd.gen.chat_service import ChatService

ww = WakeWord(**CONFIG.wakeword.model_dump())
stt = STT(**CONFIG.stt.model_dump())
llm = ChatService(CONFIG.llm)
tts = {"TTS_TF": TTS_TF, "TTS_Suno": TTS_Suno, "TTS_COCQUI": TTS_COCQUI}[
    CONFIG.tts.provider
](**CONFIG.tts.model_dump())


def check_keywords(input: str) -> bool:
    """
    Checks if the input contains certain keywords and performs corresponding actions.

    Args:
        input (str): The input string to be checked.

    Returns:
        bool: True if the input contains a keyword and an action is performed, False otherwise.
    """
    if any(word in input.lower() for word in ["reset", "danke", "copyright WDR"]):
        llm.reset()
        return True
    if "exit" in input.lower():
        sys.exit(0)
    return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    tts("Hallo! Ich bin einsatzbereit!")
    while True:
        ww()
        while True:
            input = stt()
            if not input or check_keywords(input):
                break
            response = llm.submit_user_message({"message": input})
            tts(response.text)
