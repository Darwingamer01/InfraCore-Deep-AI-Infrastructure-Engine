"""ReAct prompt builder for LLM-based reasoning."""

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from src.infracore.agents.tools import BaseTool


@dataclass
class Step:
    """Single step in agent reasoning."""

    thought: str
    action_name: str
    action_input: dict
    observation: str


@dataclass
class ParsedAction:
    """Parsed LLM output."""

    thought: str
    action_name: Optional[str] = None
    action_input: Optional[dict] = None
    is_final: bool = False
    final_answer: Optional[str] = None


class PromptBuilder:
    """Build and parse ReAct prompts."""

    def build_system_prompt(self, tools: List[BaseTool]) -> str:
        """Build system prompt with tools and ReAct instructions."""
        tools_section = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])

        system_prompt = f"""You are a helpful AI assistant that solves problems step by step using the ReAct (Reason + Act) framework.

You have access to the following tools:
{tools_section}

Use the following format for your response:

Thought: Do I need to use a tool? Yes, and if so which one?
Action: the action to take, should be one of [{', '.join([t.name for t in tools])}]
Action Input: the input to the action, as a JSON object with tool-specific parameters
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: Do I now have enough information to answer the question?
Final Answer: the final answer to the original input question

When you have enough information, respond with "Final Answer:" followed by your answer.

Important:
- Always think before using a tool
- Use JSON format for Action Input
- Stop when you have a complete answer
- Your answer should directly address the original question"""

        return system_prompt

    def build_user_prompt(self, query: str, scratchpad: List[Step]) -> str:
        """Build user prompt with query and scratchpad."""
        scratchpad_text = ""

        for step in scratchpad:
            scratchpad_text += f"Thought: {step.thought}\n"
            scratchpad_text += f"Action: {step.action_name}\n"
            scratchpad_text += f"Action Input: {json.dumps(step.action_input)}\n"
            scratchpad_text += f"Observation: {step.observation}\n"

        if scratchpad_text:
            user_prompt = f"Question: {query}\n\n{scratchpad_text}Thought:"
        else:
            user_prompt = f"Question: {query}\n\nThought:"

        return user_prompt

    def parse_llm_output(self, text: str) -> ParsedAction:
        """Parse LLM output into structured action."""
        # Extract Thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else "No thought"

        # Check for Final Answer
        final_answer_match = re.search(
            r"Final Answer:\s*(.+?)$", text, re.DOTALL | re.MULTILINE
        )
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            return ParsedAction(
                thought=thought,
                is_final=True,
                final_answer=final_answer,
            )

        # Extract Action
        action_match = re.search(r"Action:\s*(\w+)", text)
        if not action_match:
            # No action found, treat as final
            return ParsedAction(
                thought=thought,
                is_final=True,
                final_answer="Unable to parse action from response",
            )

        action_name = action_match.group(1)

        # Extract Action Input
        action_input = {}
        action_input_match = re.search(
            r"Action Input:\s*(\{.+?\})", text, re.DOTALL
        )

        if action_input_match:
            try:
                action_input = json.loads(action_input_match.group(1))
            except json.JSONDecodeError:
                # Fallback: treat as raw string input
                raw_input = action_input_match.group(1).strip()
                action_input = {"input": raw_input}
        else:
            # No JSON found, use empty dict
            action_input = {}

        return ParsedAction(
            thought=thought,
            action_name=action_name,
            action_input=action_input,
            is_final=False,
        )
