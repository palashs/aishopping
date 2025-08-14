import json
from typing import List, Tuple

from autogen_core import (
    FunctionCall,
    MessageContext,
    RoutedAgent,
    TopicId,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    SystemMessage
)
from autogen_core.tools import Tool
from models import UserTask,AgentResponse
import asyncio
from logger import logger_instance

# Logger context for the application
logger_ctx = logger_instance

class AIAgent(RoutedAgent):
    def __init__(
        self,
        description: str,
        system_message: SystemMessage,
        model_client: ChatCompletionClient,
        tools: List[Tool],
        delegate_tools: List[Tool],
        agent_topic_type: str,
        user_topic_type: str,
        response_queue : asyncio.Queue[str | object]
    ) -> None:
        super().__init__(description)
        self._system_message = system_message
        self._model_client = model_client
        self._tools = dict([(tool.name, tool) for tool in tools])
        self._tool_schema = [tool.schema for tool in tools]
        self._delegate_tools = dict([(tool.name, tool) for tool in delegate_tools])
        self._delegate_tool_schema = [tool.schema for tool in delegate_tools]
        self._agent_topic_type = agent_topic_type
        self._user_topic_type = user_topic_type
        self._response_queue = response_queue

    @message_handler
    async def handle_task(self, message: UserTask, ctx: MessageContext) -> None:
        logger_ctx.info(f"{self.id.type} response started")
        # Start streaming LLM responses
        llm_stream = self._model_client.create_stream(
            messages=[self._system_message] + message.context,
            tools=self._tool_schema + self._delegate_tool_schema,
           # json_output = True,
            cancellation_token=ctx.cancellation_token
        )
        final_response = None
        async for chunk in llm_stream:
            if isinstance(chunk, str):
                await self._response_queue.put({'type': "string", 'message': chunk})
            else:
                final_response = chunk
        assert final_response is not None, "No response from model"
        logger_ctx.info(f"{self.id.type} response received")
        logger_ctx.debug(f"{self.id.type}: {final_response.content}")
        # Process the LLM result.
        while isinstance(final_response.content, list) and all(isinstance(m, FunctionCall) for m in final_response.content):
            tool_call_results: List[FunctionExecutionResult] = []
            delegate_targets: List[Tuple[str, UserTask]] = []
            # Process each function call.
            for call in final_response.content:
                arguments = json.loads(call.arguments)
                await self._response_queue.put({"type":"function","message":f"Executing {call.name}"})
                if call.name in self._tools:
                    # Execute the tool directly.
                    result = await self._tools[call.name].run_json(arguments, ctx.cancellation_token)
                    result_as_str = self._tools[call.name].return_value_as_string(result)
                    tool_call_results.append(
                        FunctionExecutionResult(call_id=call.id, content=result_as_str, is_error=False, name=call.name)
                    )
                elif call.name in self._delegate_tools:
                    # Execute the tool to get the delegate agent's topic type.
                    result = await self._delegate_tools[call.name].run_json(arguments, ctx.cancellation_token)
                    topic_type = self._delegate_tools[call.name].return_value_as_string(result)
                    # Create the context for the delegate agent, including the function call and the result.
                    delegate_messages = list(message.context) + [
                        AssistantMessage(content=[call], source=self.id.type),
                        FunctionExecutionResultMessage(
                            content=[
                                FunctionExecutionResult(
                                    call_id=call.id,
                                    content=f"Transferred to {topic_type}. Adopt persona immediately.",
                                    is_error=False,
                                    name=call.name,
                                )
                            ]
                        ),
                    ]
                    delegate_targets.append((topic_type, UserTask(context=delegate_messages)))
                else:
                    logger_ctx.error(f"Unknown tool: {call.name}")
                    raise ValueError(f"Unknown tool: {call.name}")
            if len(delegate_targets) > 0:
                # Delegate the task to other agents by publishing messages to the corresponding topics.
                for topic_type, task in delegate_targets:
                    logger_ctx.info(f"{self.id.type}: Delegating to {topic_type}")
                    await self._response_queue.put({"type":"function","message":f"You are now talking to {topic_type}"})
                    await self.publish_message(task, topic_id=TopicId(topic_type, source=self.id.key))
            if len(tool_call_results) > 0:
                logger_ctx.debug(f"{self.id.type}: Tool call results: {tool_call_results}")
                # Make another LLM call with the results.
                message.context.extend([
                    AssistantMessage(content=final_response.content, source=self.id.type),
                    FunctionExecutionResultMessage(content=tool_call_results),
                ])
                llm_stream = self._model_client.create_stream(
                    messages=[self._system_message] + message.context,
                    tools=self._tool_schema + self._delegate_tool_schema,
                    #json_output = True,
                    cancellation_token=ctx.cancellation_token
                )
                final_response = None
                async for chunk in llm_stream:
                    if isinstance(chunk, str):
                        await self._response_queue.put({'type': 'string', 'message': chunk})
                    else:
                        final_response = chunk
                assert final_response is not None, "No response from model"
                logger_ctx.info(f"{self.id.type} response received after tool execution")
                logger_ctx.debug(f"{self.id.type}: {final_response.content}")
            else:
                # The task has been delegated, so we are done.
                return
        # The task has been completed, publish the final result.
        assert isinstance(final_response.content, str)
        message.context.append(AssistantMessage(content=final_response.content, source=self.id.type))
        await self.publish_message(
            AgentResponse(context=message.context, reply_to_topic_type=self._agent_topic_type),
            topic_id=TopicId(self._user_topic_type, source=self.id.key),
        )
