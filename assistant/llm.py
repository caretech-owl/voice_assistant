import logging
from collections.abc import Generator

from llama_cpp import Llama


class LLM:
    """
    Language Model for the Voice Assistant.

    Args:
        model_path (str): The path to the pretrained model. Defaults to "TheBloke/DiscoLM_German_7b_v1-GGUF".
        filename (str): The filename of the pretrained model. Defaults to "discolm_german_7b_v1.Q4_K_M.gguf".
        chat_template (str): The template for generating chat responses. Defaults to a predefined template.
        system_prompt (str): The system prompt for the assistant. Defaults to "Du bist ein hilfreicher Assistent."
        logger (Optional): The logger object for logging messages. Defaults to None.
        stop_string (str): The string used to stop the generation of chat responses. Defaults to an empty string.

    Attributes:
        llama (Llama): The pretrained language model.
        system_prompt (str): The system prompt for the assistant.
        stop_string (List[str]): The list of stop strings used to stop the generation of chat responses.
        logger (Logger): The logger object for logging messages.
        messages (List[Dict[str, str]]): The list of messages exchanged between the user and the assistant.

    Methods:
        reset(): Resets the language model by clearing the messages.
        __call__(userprompt: str) -> str: Generates a chat response given a user prompt.
        predict(userprompt: str) -> Generator[str, None, None]: Generates a sequence of chat responses given a user prompt.
    """

    default_model = "TheBloke/DiscoLM_German_7b_v1-GGUF"
    default_filename = "discolm_german_7b_v1.Q4_K_M.gguf"
    default_system_prompt = "Du bist ein hilfreicher Assistent."

    def __init__(
        self,
        model_path: str = default_model,
        filename: str = default_filename,
        chat_format: str = "chatml",
        system_prompt: str = default_system_prompt,
        logger=None,
        stop_string: str = "",
    ) -> None:
        self.llama = Llama.from_pretrained(
            repo_id=model_path,
            filename=filename,
            chat_format=chat_format,
            verbose=False,
        )
        self.system_prompt = system_prompt
        self.stop_string = [] if stop_string == "" else [stop_string]

        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
        self.logger.info("LLM initialized")
        self.reset()

    def reset(self) -> None:
        self.logger.info("LLM.reset(): Resetting LLM")
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def __call__(self, userprompt: str) -> str:
        self.messages.append({"role": "user", "content": userprompt})
        self.logger.info(f"User: {userprompt}")
        self.logger.debug(f"Messages: {self.messages}")
        response = self.llama.create_chat_completion(
            messages=self.messages, stop=self.stop_string
        )["choices"][0]["message"]["content"]
        self.logger.info(f"LLM Antwort: {response}")
        self.messages.append({"role": "assistant", "content": response})
        return response

    def predict(self, userprompt: str) -> Generator[str, None, None]:
        self.messages.append({"role": "user", "content": userprompt})
        self.logger.info(f"User: {userprompt}")
        response = self.llama.create_chat_completion_openai_v1(
            messages=self.messages, stream=True
        )
        text = ""
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                text += content
                yield text
        self.logger.info(f"Assistant: {text}")
        self.messages.append({"role": "assistant", "content": text})
