import time

from openai import util
from openai.api_resources.abstract.engine_api_resource import EngineAPIResource
from openai.error import TryAgain


class Event(EngineAPIResource):
    engine_required = False
    OBJECT_NAME = "chat.completions"

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
        """
        Creates a new chat completion for the provided messages and parameters.

        See https://platform.openai.com/docs/api-reference/chat/create
        for a list of valid parameters.
        """
        start = time.time()

        # Set default values if not provided in kwargs
        model = kwargs.pop("model", "gpt-3.5-turbo")
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
        """
        Creates a new chat completion for the provided messages and parameters.

        See https://platform.openai.com/docs/api-reference/chat/create
        for a list of valid parameters.
        """
        start = time.time()

        # Set default values if not provided in kwargs
        model = kwargs.pop("model", "gpt-3.5-turbo")
        temperature = kwargs.pop("temperature", 0.7)
        timeout = kwargs.pop("timeout", None)

        while True:
            try:
                return await super().acreate(*args, model=model, temperature=temperature, **kwargs)
            except TryAgain as e:
                if timeout is not None and time.time() > start + timeout:
                    raise

                util.log_info("Waiting for model to warm up", error=e)
