import logging

import speech_recognition as sr
import torch


class STT:
    """
    Speech-to-Text (STT) class for converting spoken language into written text.

    Args:
        model (str): The speech recognition model to use. Default is "tiny".
        pause_threshold (float): seconds of non-speaking audio before a phrase is considered complete. Default is 0.2 seconds.
        non_speaking_duration (float): # seconds of non-speaking audio to keep on both sides of the recording. Default is 0.2 seconds.
        timeout (float): The maximum time to wait for audio input. Default is 5.0 seconds.
        phrase_time_limit (float): The maximum time allowed for a phrase to be spoken. Default is 10.0 seconds.
        language (str): The language of the speech to be recognized. Default is "german".
        logger: The logger object to use for logging. If None, a default logger will be used.

    Returns:
        str: The recognized text from the speech input.

    """

    def __init__(
        self,
        model: str = "tiny",
        pause_threshold: float = 0.2,
        non_speaking_duration: float = 0.2,
        timeout: float = 5.0,
        phrase_time_limit: float = 10.0,
        language: str = "german",
        logger=None,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit
        self.language = language

        self.r = sr.Recognizer()
        self.r.pause_threshold = pause_threshold
        self.r.non_speaking_duration = non_speaking_duration
        # r.energy_threshold = ?
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_options = {"device": device}

        self.logger.info("STT: Initialized")

    def __call__(self) -> str:
        """
        Perform speech recognition on the audio input from the microphone.

        Returns:
            str: The recognized text from the speech input.

        """
        with sr.Microphone() as source:
            self.r.adjust_for_ambient_noise(source)
            self.logger.info("Listening...")
            audio = self.r.listen(
                source, timeout=self.timeout, phrase_time_limit=self.phrase_time_limit
            )
        self.logger.info("Recognizing speech from audio data")
        ret = self.r.recognize_whisper(
            audio,
            language=self.language,
            model=self.model,
            load_options=self.load_options,
        )
        self.logger.info(f"STT: {ret}")
        return str(ret)
