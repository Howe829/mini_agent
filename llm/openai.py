from typing import Iterable
from openai import OpenAI, AsyncOpenAI
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionFunctionToolParam,
)
from dataclasses import dataclass


@dataclass
class OpenAiLike:
    client: OpenAI
    tools: list[ChatCompletionFunctionToolParam]

    def chat(
        self, messages: Iterable[ChatCompletionMessageParam], model: str
    ) -> ChatCompletionMessage:
        response = self.client.chat.completions.create(
            model=model, messages=messages, tools=self.tools
        )
        message = response.choices[0].message
        return message

    def stream_chat(self, messages: Iterable[ChatCompletionMessageParam], model: str):
        response = self.client.chat.completions.create(
            model=model, messages=messages, tools=self.tools, stream=True
        )
        return response


@dataclass
class AsyncOpenAiLike:
    client: AsyncOpenAI
    tools: list[ChatCompletionFunctionToolParam]

    async def chat(
        self, messages: Iterable[ChatCompletionMessageParam], model: str
    ) -> ChatCompletionMessage:
        response = await self.client.chat.completions.create(
            model=model, messages=messages, tools=self.tools
        )
        message = response.choices[0].message
        return message

    async def stream_chat(
        self, messages: Iterable[ChatCompletionMessageParam], model: str
    ):
        response = await self.client.chat.completions.create(
            model=model, messages=messages, tools=self.tools, stream=True
        )
        return response
