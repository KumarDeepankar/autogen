from autogen_core import AgentId, MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler
from dataclasses import dataclass


@dataclass
class TextMessage:
    content: str
    source: str


@dataclass
class ImageMessage:
    url: str
    source: str


class RoutedBySenderAgent(RoutedAgent):
    @message_handler(match=lambda msg, ctx: msg.source.startswith("user1"))  # type: ignore
    async def on_user1_message(self, message: TextMessage, ctx: MessageContext) -> None:
        print(f"Hello from user 1 handler, {message.source}, you said {message.content}!")

    @message_handler(match=lambda msg, ctx: msg.source.startswith("user2"))  # type: ignore
    async def on_user2_message(self, message: TextMessage, ctx: MessageContext) -> None:
        print(f"Hello from user 2 handler, {message.source}, you said {message.content}!")

    @message_handler(match=lambda msg, ctx: msg.source.startswith("user2"))  # type: ignore
    async def on_image_message(self, message: ImageMessage, ctx: MessageContext) -> None:
        print(f"Hello, {message.source}, you sent me {message.url}!")


import asyncio


async def main():
    runtime = SingleThreadedAgentRuntime()
    await RoutedBySenderAgent.register(runtime, "my_agent", lambda: RoutedBySenderAgent("Routed by sender agent"))
    runtime.start()
    agent_id = AgentId("my_agent", "default")

    await runtime.send_message(TextMessage(content="Hello, World!", source="user1-test"), agent_id)
    await runtime.send_message(TextMessage(content="Hello, World!", source="user2-test"), agent_id)
    await runtime.send_message(ImageMessage(url="https://example.com/image.jpg", source="user1-test"), agent_id)
    await runtime.send_message(ImageMessage(url="https://example.com/image.jpg", source="user2-test"), agent_id)
    await runtime.stop_when_idle()


# Run the asynchronous main function
asyncio.run(main())
