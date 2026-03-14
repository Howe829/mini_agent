from openai import OpenAI
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from dataclasses import dataclass


@dataclass
class OpenAiLike:
    client: OpenAI
    tools: list[dict]

    def chat(self, messages: list[dict], model: str) -> ChatCompletionMessage:
        response = self.client.chat.completions.create(
            model=model, messages=messages, tools=self.tools
        )
        message = response.choices[0].message
        return message

    def stream_chat(self, messages: list[dict], model: str):
        response = self.client.chat.completions.create(
            model=model, messages=messages, tools=self.tools, stream=True
        )
        return response
