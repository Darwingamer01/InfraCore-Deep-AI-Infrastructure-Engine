"""Test suite for ReActAgent, PromptBuilder, ToolRegistry."""

import json
from unittest.mock import AsyncMock

import pytest

from src.infracore.agents.prompt_builder import ParsedAction, PromptBuilder, Step
from src.infracore.agents.react_agent import AgentConfig, ReActAgent
from src.infracore.agents.tools import (
    CalculatorTool,
    CurrentTimeTool,
    ToolError,
    ToolRegistry,
    WordCountTool,
)
from src.infracore.inference.backend_base import GenerationResult


# ============================================================================
# PromptBuilder Tests (1-7)
# ============================================================================


def test_prompt_builder_system_prompt_contains_tools():
    """Test 1: build_system_prompt() contains tool names and descriptions."""
    builder = PromptBuilder()
    tools = [WordCountTool(), CalculatorTool(), CurrentTimeTool()]

    prompt = builder.build_system_prompt(tools)

    assert "word_count" in prompt
    assert "Count words, characters, or sentences" in prompt
    assert "calculator" in prompt
    assert "Evaluate mathematical expressions" in prompt
    assert "current_time" in prompt


def test_prompt_builder_user_prompt_empty_scratchpad():
    """Test 2: build_user_prompt() with empty scratchpad contains Question and Thought."""
    builder = PromptBuilder()
    query = "What is 2 + 2?"
    scratchpad = []

    prompt = builder.build_user_prompt(query, scratchpad)

    assert "Question: What is 2 + 2?" in prompt
    assert "Thought:" in prompt
    assert "Action:" not in prompt  # No previous steps


def test_prompt_builder_user_prompt_with_scratchpad():
    """Test 3: build_user_prompt() with 1 step → scratchpad formatted correctly."""
    builder = PromptBuilder()
    query = "What is 2 + 2?"
    step = Step(
        thought="I need to use the calculator",
        action_name="calculator",
        action_input={"expression": "2 + 2"},
        observation="Result: 4",
    )
    scratchpad = [step]

    prompt = builder.build_user_prompt(query, scratchpad)

    assert "Question: What is 2 + 2?" in prompt
    assert "Thought: I need to use the calculator" in prompt
    assert "Action: calculator" in prompt
    assert '"expression": "2 + 2"' in prompt
    assert "Observation: Result: 4" in prompt


def test_prompt_builder_parse_valid_action():
    """Test 4: parse_llm_output() with valid Thought/Action/Action Input → ParsedAction fields correct."""
    builder = PromptBuilder()
    llm_output = """Thought: I need to calculate something
Action: calculator
Action Input: {"expression": "15 * 7 + 42"}"""

    parsed = builder.parse_llm_output(llm_output)

    assert parsed.thought == "I need to calculate something"
    assert parsed.action_name == "calculator"
    assert parsed.action_input == {"expression": "15 * 7 + 42"}
    assert parsed.is_final is False


def test_prompt_builder_parse_final_answer():
    """Test 5: parse_llm_output() with Final Answer → is_final=True, final_answer extracted."""
    builder = PromptBuilder()
    llm_output = """Thought: I have the answer
Final Answer: The answer is 147"""

    parsed = builder.parse_llm_output(llm_output)

    assert parsed.is_final is True
    assert parsed.final_answer == "The answer is 147"


def test_prompt_builder_parse_invalid_json_fallback():
    """Test 6: parse_llm_output() with invalid JSON Action Input → fallback to {"input": raw_string}."""
    builder = PromptBuilder()
    llm_output = """Thought: Use word_count
Action: word_count
Action Input: {text: "hello world"}"""  # Invalid JSON

    parsed = builder.parse_llm_output(llm_output)

    assert parsed.action_name == "word_count"
    assert "input" in parsed.action_input  # Fallback structure


def test_prompt_builder_parse_garbage_returns_final():
    """Test 7: parse_llm_output() with garbage text → returns is_final=True (safe fallback)."""
    builder = PromptBuilder()
    llm_output = "blah blah blah random text"

    parsed = builder.parse_llm_output(llm_output)

    assert parsed.is_final is True  # Safe fallback


# ============================================================================
# ToolRegistry Tests (8-10)
# ============================================================================


def test_tool_registry_register_and_get():
    """Test 8: register() + get() returns correct tool."""
    registry = ToolRegistry()
    tool = WordCountTool()

    registry.register(tool)
    retrieved = registry.get("word_count")

    assert retrieved is tool


def test_tool_registry_get_unknown_returns_none():
    """Test 9: get() with unknown name returns None."""
    registry = ToolRegistry()

    result = registry.get("unknown_tool")

    assert result is None


def test_tool_registry_list_tools():
    """Test 10: list_tools() returns string containing all tool names."""
    registry = ToolRegistry()
    registry.register(WordCountTool())
    registry.register(CalculatorTool())

    tools_str = registry.list_tools()

    assert "word_count" in tools_str
    assert "calculator" in tools_str
    assert "Count words" in tools_str


# ============================================================================
# CalculatorTool Tests (11-12)
# ============================================================================


@pytest.mark.asyncio
async def test_calculator_operator_precedence():
    """Test 11: 2 + 2 * 3 → Result: 8 (operator precedence)."""
    calculator = CalculatorTool()

    result = await calculator.call(expression="2 + 2 * 3")

    assert result.success is True
    assert "Result: 8" in result.output


@pytest.mark.asyncio
async def test_calculator_rejects_unsafe_import():
    """Test 12: 'import os' → ToolError raised (unsafe)."""
    calculator = CalculatorTool()

    result = await calculator.call(expression="import os")

    assert result.success is False
    assert "Unsafe" in result.error or "restricted" in result.error.lower()


# ============================================================================
# ReActAgent Tests (13-17)
# ============================================================================


@pytest.mark.asyncio
async def test_agent_stops_on_final_answer_immediately():
    """Test 13: Agent stops when LLM returns Final Answer on first step."""
    config = AgentConfig(max_steps=10, verbose=False)
    llm = AsyncMock()
    llm.chat = AsyncMock(
        return_value=GenerationResult(
            text="Thought: I have the answer\nFinal Answer: The answer is 42",
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
            latency_ms=100.0,
            model="test",
            finish_reason="stop",
        )
    )

    tools = [WordCountTool()]
    agent = ReActAgent(llm=llm, tools=tools, config=config)

    result = await agent.run("What is the answer?")

    assert result.answer == "The answer is 42"
    assert result.total_steps == 1
    assert result.success is True


@pytest.mark.asyncio
async def test_agent_calls_tool_and_continues():
    """Test 14: Agent calls tool on step 1, returns Final Answer on step 2."""
    config = AgentConfig(max_steps=10, verbose=False)

    # Mock LLM: step 1 returns action, step 2 returns final answer
    llm = AsyncMock()
    llm.chat = AsyncMock(
        side_effect=[
            GenerationResult(
                text='Thought: Count words\nAction: word_count\nAction Input: {"text": "hello world"}',
                prompt_tokens=50,
                completion_tokens=20,
                total_tokens=70,
                latency_ms=100.0,
                model="test",
                finish_reason="stop",
            ),
            GenerationResult(
                text="Thought: Got result\nFinal Answer: The text has 2 words",
                prompt_tokens=60,
                completion_tokens=15,
                total_tokens=75,
                latency_ms=100.0,
                model="test",
                finish_reason="stop",
            ),
        ]
    )

    tools = [WordCountTool()]
    agent = ReActAgent(llm=llm, tools=tools, config=config)

    result = await agent.run("How many words in 'hello world'?")

    assert result.success is True
    assert result.total_steps == 2
    assert len(result.steps) == 1  # One tool call


@pytest.mark.asyncio
async def test_agent_returns_failure_after_max_steps():
    """Test 15: Agent returns success=False after max_steps with no Final Answer."""
    config = AgentConfig(max_steps=2, verbose=False)

    # Mock LLM always returns action (never final answer)
    llm = AsyncMock()
    llm.chat = AsyncMock(
        return_value=GenerationResult(
            text='Thought: Action\nAction: calculator\nAction Input: {"expression": "1+1"}',
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            latency_ms=100.0,
            model="test",
            finish_reason="stop",
        )
    )

    tools = [CalculatorTool()]
    agent = ReActAgent(llm=llm, tools=tools, config=config)

    result = await agent.run("Calculate something")

    assert result.success is False
    assert result.total_steps == 2
    assert "Max steps reached" in result.answer or "exceeded" in result.error.lower()


@pytest.mark.asyncio
async def test_agent_handles_unknown_tool():
    """Test 16: Agent handles unknown tool gracefully (observation = Error message)."""
    config = AgentConfig(max_steps=10, verbose=False)

    llm = AsyncMock()
    llm.chat = AsyncMock(
        side_effect=[
            GenerationResult(
                text='Thought: Use tool\nAction: unknown_tool\nAction Input: {}',
                prompt_tokens=50,
                completion_tokens=20,
                total_tokens=70,
                latency_ms=100.0,
                model="test",
                finish_reason="stop",
            ),
            GenerationResult(
                text="Thought: Got error\nFinal Answer: Tool not found",
                prompt_tokens=60,
                completion_tokens=15,
                total_tokens=75,
                latency_ms=100.0,
                model="test",
                finish_reason="stop",
            ),
        ]
    )

    tools = [WordCountTool()]
    agent = ReActAgent(llm=llm, tools=tools, config=config)

    result = await agent.run("Test unknown tool")

    assert result.success is True
    assert len(result.steps) >= 1
    # First step should have error observation
    assert "Error" in result.steps[0].observation or "not found" in result.steps[0].observation


@pytest.mark.asyncio
async def test_agent_result_contains_correct_step_count():
    """Test 17: AgentResult contains correct step count."""
    config = AgentConfig(max_steps=10, verbose=False)

    llm = AsyncMock()
    llm.chat = AsyncMock(
        side_effect=[
            GenerationResult(
                text='Thought: Step 1\nAction: word_count\nAction Input: {"text": "hello"}',
                prompt_tokens=50,
                completion_tokens=20,
                total_tokens=70,
                latency_ms=100.0,
                model="test",
                finish_reason="stop",
            ),
            GenerationResult(
                text='Thought: Step 2\nAction: calculator\nAction Input: {"expression": "2+2"}',
                prompt_tokens=60,
                completion_tokens=20,
                total_tokens=80,
                latency_ms=100.0,
                model="test",
                finish_reason="stop",
            ),
            GenerationResult(
                text="Thought: Done\nFinal Answer: Done",
                prompt_tokens=70,
                completion_tokens=15,
                total_tokens=85,
                latency_ms=100.0,
                model="test",
                finish_reason="stop",
            ),
        ]
    )

    tools = [WordCountTool(), CalculatorTool()]
    agent = ReActAgent(llm=llm, tools=tools, config=config)

    result = await agent.run("Test multi-step")

    assert result.total_steps == 3
    assert len(result.steps) == 2  # Two tool calls before final answer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
