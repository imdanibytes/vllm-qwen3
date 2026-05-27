# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.tool_parsers.qwen3xml_tool_parser import Qwen3XMLToolParser


def test_required_tool_choice_uses_configured_parser_for_xml_tool_calls():
    request = ChatCompletionRequest(
        messages=[],
        model="test-model",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "send_message",
                    "description": "Send a message to the user.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                        },
                        "required": ["text"],
                    },
                },
            }
        ],
        tool_choice="required",
    )
    content = (
        "<tool_call>\n"
        "<function=send_message>\n"
        "<parameter=text>message delivered</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    function_calls, remaining_content = OpenAIServing._parse_tool_calls_from_content(
        request=request,
        tokenizer=object(),
        enable_auto_tools=True,
        tool_parser_cls=Qwen3XMLToolParser,
        content=content,
    )

    assert function_calls is not None
    assert len(function_calls) == 1
    assert function_calls[0].name == "send_message"
    assert json.loads(function_calls[0].arguments) == {"text": "message delivered"}
    assert remaining_content is None
