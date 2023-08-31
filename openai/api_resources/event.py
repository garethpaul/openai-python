import time

from openai import util
from openai.api_resources.abstract.engine_api_resource import EngineAPIResource
from openai.error import TryAgain
import tiktoken


class Event(EngineAPIResource):
    engine_required = False
    OBJECT_NAME = "chat.completions"

    @classmethod
    def choose_model(messages):
        # Tokenize the messages
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        # encode the messages
        num_tokens = sum(len(tokenizer.encode(m["content"])) for m in messages)

        # For simplicity's sake, I'll use two hypothetical models.
        # Adjust the token thresholds and models as necessary.
        if num_tokens <= 4096:
            return "gpt-3.5-turbo"
        if num_tokens >= 4096 and num_tokens <= 16385:
            return "gpt-3.5-turbo-16k"
        return "gpt-3.5-turbo"

    @classmethod
    def get_best_model_for_tokens(cls, token_count):
        """Select an appropriate model based on token count"""
        if token_count <= 4096:
            return "gpt-3.5-turbo-0613"
        if token_count >= 4096 and token_count <= 16385:
            return "gpt-3.5-turbo-16k-0613"
        # You can add more conditions here for other models if necessary
        return "gpt-3.5-turbo"

    @classmethod
    def start_chat_session(cls, initial_system_message):
        return [
            {"role": "system", "content": initial_system_message}
        ]

    @classmethod
    def add_message_to_session(cls, session, role, content):
        session.append({"role": role, "content": content})
        return session

    @classmethod
    def ask_question(cls, session, question):
        return cls.add_message_to_session(session, "user", question)

    @classmethod
    def add_system_instruction(cls, session, instruction):
        return cls.add_message_to_session(session, "system", instruction)

    @classmethod
    def create(cls, *args, **kwargs):
        start = time.time()

        # Count tokens in all messages
        total_tokens = sum(cls.count_tokens(
            message["content"]) for message in kwargs.get("messages", []))
        model = kwargs.pop(
            "model", cls.get_best_model_for_tokens(total_tokens))

        temperature = kwargs.pop("temperature", 0.7)
        timeout = kwargs.pop("timeout", None)

        while True:
            try:
                return super().create(*args, model=model, temperature=temperature, **kwargs)
            except TryAgain as e:
                if timeout is not None and time.time() > start + timeout:
                    raise

                util.log_info("Waiting for model to warm up", error=e)

    @classmethod
    async def acreate(cls, *args, **kwargs):
        start = time.time()

        # Count tokens in all messages
        total_tokens = sum(cls.count_tokens(
            message["content"]) for message in kwargs.get("messages", []))
        model = kwargs.pop(
            "model", cls.get_best_model_for_tokens(total_tokens))

        temperature = kwargs.pop("temperature", 0.7)
        timeout = kwargs.pop("timeout", None)

        while True:
            try:
                return await super().acreate(*args, model=model, temperature=temperature, **kwargs)
            except TryAgain as e:
                if timeout is not None and time.time() > start + timeout:
                    raise

                util.log_info("Waiting for model to warm up", error=e)
