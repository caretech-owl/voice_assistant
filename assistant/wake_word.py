# Adapted from https://github.com/dscripka/openWakeWord/blob/main/examples/detect_from_microphone.py
# Copyright 2022 David Scripka. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Imports
import logging

import numpy as np
import openwakeword
import pyaudio
from openwakeword.model import Model


class WakeWord:
    """
    Class representing a wake word listener.

    Args:
        chunk_size (int): The size of each audio chunk to process.
        model_path (str): The path to the wake word model.
        inference_framework (str): The framework used for inference.
        logger: The logger object for logging messages.

    Attributes:
        mic_stream: The microphone stream for audio input.
        owwModel: The openwakeword model for wake word detection.
        logger: The logger object for logging messages.
        CHUNK: The size of each audio chunk to process.

    """

    def __init__(
        self,
        chunk_size: int = 1280,
        model_path: str = "hey_jarvis",
        inference_framework: str = "onnx",
        logger=None,
    ) -> None:
        # Get microphone stream
        openwakeword.utils.download_models([model_path])
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        self.CHUNK = chunk_size
        audio = pyaudio.PyAudio()
        self.mic_stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )

        # Load pre-trained openwakeword models
        if model_path != "":
            self.owwModel = Model(
                wakeword_models=[model_path], inference_framework=inference_framework
            )
        else:
            self.owwModel = Model(inference_framework=inference_framework)
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
        self.logger.info("WakewordListener initialized")
        # n_models = len(owwModel.models.keys())

    def __call__(self) -> None:
        """
        Start the wakeword detection.

        This method continuously listens to audio input from the microphone,
        feeds it to the openwakeword model for prediction, and checks if the
        predicted wake word score is above a threshold. If a wake word is detected,
        it logs the detection and resets the openwakeword model.

        Returns:
            None

        """
        self.logger.info("WakewordListener.__call__(): Starting wakeword detection")
        while True:
            # Get audio
            audio = np.frombuffer(self.mic_stream.read(self.CHUNK), dtype=np.int16)

            # Feed to openWakeWord model
            prediction = self.owwModel.predict(audio)
            for wakeword, score in prediction.items():
                self.logger.debug(f"WakewordListener.run(): {wakeword} score = {score}")
                if score > 0.5:
                    logging.info(
                        f"WakewordListener.run(): Wakeword detected! score = {score}"
                    )
                    self.owwModel.reset()
                    return
