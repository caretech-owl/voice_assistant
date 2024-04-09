import logging

import numpy as np
import pyaudio
import scipy.signal as sps
import torch


class TTS_BASE:
    """
    Base class for Text-to-Speech (TTS) functionality.

    Attributes:
        logger (logging.Logger): The logger object for logging messages.
        p (pyaudio.PyAudio): The PyAudio object for audio processing.
        audio_device_id (int): The ID of the default audio output device.
        output_rate (float): The default sample rate of the audio output device.
        stream (pyaudio.Stream): The audio stream for playing audio.
        device (str): The device used for TTS processing (either "cuda" or "cpu").
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes the TTS_BASE object.

        Args:
            logger (logging.Logger, optional): The logger object for logging messages.
        """
        if "logger" in kwargs and kwargs["logger"] is not None:
            self.logger = kwargs["logger"]
        else:
            self.logger = logging.getLogger(__name__)

        self.p = pyaudio.PyAudio()
        self.audio_device_id = int(self.p.get_default_output_device_info()["index"])
        self.output_rate = self.p.get_device_info_by_index(self.audio_device_id)[
            "defaultSampleRate"
        ]
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=int(self.output_rate),
            output=True,
            output_device_index=self.audio_device_id,
        )
        self.logger.info(f"Audio device ID: {self.audio_device_id}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.logger.info("TTS initialized")

    def resample_audio(self, wav: list[float]) -> list[float]:
        """
        Resamples the audio waveform to match the output sample rate.

        Args:
            wav (list[float]): The input audio waveform.

        Returns:
            list[float]: The resampled audio waveform.
        """
        num_samples = round(
            len(wav) * float(self.output_rate) / float(self.sample_rate)
        )
        self.logger.debug(
            f"Resampling audio from {self.sample_rate} to {self.output_rate} with {num_samples} samples."
        )
        return sps.resample(wav, num_samples)


class TTS_TF(TTS_BASE):
    """
    Text-to-Speech class using the Transformers library and VitsModel for German language.
    (Very fast)

    Args:
        speaking_rate (float): The speaking rate for the generated speech. Default is 1.5.
        **kwargs: Additional keyword arguments to be passed to the base class.

    Attributes:
        voice_model: The VitsModel instance for generating speech.
        voice_processor: The AutoTokenizer instance for processing text.
        sample_rate: The sample rate of the generated speech.

    """

    def __init__(self, speaking_rate: float = 1.5, **kwargs) -> None:
        super().__init__(**kwargs)
        from transformers import AutoTokenizer, VitsModel

        self.voice_model = VitsModel.from_pretrained(
            "facebook/mms-tts-deu", speaking_rate=speaking_rate
        ).to(self.device)
        self.voice_processor = AutoTokenizer.from_pretrained("facebook/mms-tts-deu")

        self.sample_rate = self.voice_model.config.sampling_rate

    def __call__(self, text: str) -> None:
        """
        Generate speech from the given text.

        Args:
            text (str): The input text to be converted to speech.

        """
        for sentence in text.split("."):
            if sentence == "":
                continue
            input = self.voice_processor(text=sentence, return_tensors="pt")
            with torch.no_grad():
                speech_values = self.voice_model(**input).waveform
                data = speech_values.cpu().numpy().squeeze().astype(np.float32)
            data = self.resample_audio(data)
            self.stream.write(data.tobytes())


class TTS_Suno(TTS_BASE):
    """
    TTS_Suno class represents a text-to-speech system using the Suno model.

    Args:
        voice_preset (str): The voice preset to use for synthesis. Default is "v2/de_speaker_3".
        **kwargs: Additional keyword arguments.

    Attributes:
        voice_processor: The AutoProcessor for text preprocessing.
        voice_model: The BarkModel for speech synthesis.
        voice_preset (str): The voice preset used for synthesis.
        sample_rate: The sample rate of the generated speech.

    """

    def __init__(self, voice_preset: str = "v2/de_speaker_3", **kwargs) -> None:
        super().__init__(**kwargs)
        from transformers import AutoProcessor, BarkModel

        self.voice_processor = AutoProcessor.from_pretrained("suno/bark-small").to(
            self.device
        )
        self.voice_model = BarkModel.from_pretrained("suno/bark-small")

        self.voice_model = self.voice_model.to_bettertransformer()
        self.voice_preset = voice_preset
        self.sample_rate = self.voice_model.generation_config.sample_rate

    def __call__(self, text: str) -> None:
        """
        Synthesizes the given text and plays the generated speech.

        Args:
            text (str): The input text to synthesize.

        Returns:
            None

        """
        input = self.voice_processor(
            text=text, return_tensors="pt", voice_preset=self.voice_preset
        )
        speech_values = self.voice_model.generate(**input, do_sample=True)
        data = speech_values.cpu().numpy().squeeze().astype(np.float32)

        data = self.resample_audio(data)
        self.stream.write(data.tobytes())
        # sd.play(data, self.sample_rate)
        # sd.wait()


class TTS_COCQUI(TTS_BASE):
    """
    TTS_COCQUI class for text-to-speech synthesis using the COCQUI TTS model.

    Args:
        model (str): Path to the COCQUI TTS model.
        language (str): Language for synthesis.
        speaker (str): Speaker for synthesis.
        split_sentences (bool): Whether to split input text into sentences.

    Attributes:
        model: The COCQUI TTS model.
        xtts (bool): Flag indicating whether the model is an XTTS model.
        speaker (str): Speaker for synthesis.
        language (str): Language for synthesis.
        sample_rate: Output sample rate of the synthesizer.
        split_sentences (bool): Whether to split input text into sentences.
    """

    def __init__(
        self,
        model: str = "tts_models/de/thorsten/vits",
        language: str = "de",
        speaker: str = "Aaron Dreschner",
        split_sentences: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        from TTS.api import TTS

        # Get device
        self.model = TTS(model).to(self.device)
        if "xtts" in model:
            self.xtts = True
            self.speaker = speaker
            self.language = language
        else:
            self.xtts = False
            self.speaker = None
            self.language = None
        # self.sample_rate = self.model.tts_config.audio["sample_rate"]
        self.sample_rate = self.model.synthesizer.output_sample_rate
        self.split_sentences = split_sentences

    def __call__(self, text: str) -> None:
        """
        Synthesizes the given text and plays the audio.

        Args:
            text (str): The input text to synthesize.
        """
        data = (
            np.array(
                self.model.tts(
                    text,
                    language=self.language,
                    speaker=self.speaker,
                    split_sentences=self.split_sentences,
                )
            )
            .squeeze()
            .astype(np.float32)
        )

        self.logger.info(f"Synthesized audio length {len(data)}")
        self.stream.write(self.resample_audio(data).tobytes())

        # sd.play(data, self.sample_rate)
        # sd.wait()


if __name__ == "__main__":
    text = "Hallo, ich bin ein AI EI Assistant. Wie kann ich Ihnen helfen?"
    tts = TTS_COCQUI()
    tts(text)
    tts = TTS_COCQUI(model="tts_models/multilingual/multi-dataset/xtts_v2")
    tts(text)
    tts = TTS_Suno()
    tts(text)
    tts = TTS_TF()
    tts(text)
