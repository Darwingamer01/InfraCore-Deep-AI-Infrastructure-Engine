"""ReActAgent - Core agent loop implementing Reason + Act pattern."""

import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, List, Optional

import structlog
from prometheus_client import Counter as PrometheusCounter
from prometheus_client import Histogram
from pydantic import BaseModel, ConfigDict, Field

from src.infracore.agents.prompt_builder import PromptBuilder, Step
from src.infracore.agents.tools import BaseTool, ToolRegistry
from src.infracore.inference.backend_base import BaseInferenceBackend, GenerationResult

logger = structlog.get_logger()


class AgentConfig(BaseModel):
    """Agent configuration."""

    model_config = ConfigDict(frozen=True)

    max_steps: int = Field(default=10, description="Maximum reasoning steps")
    verbose: bool = Field(default=True, description="Log each step")
    early_stop_on_final: bool = Field(default=True, description="Stop immediately on Final Answer")


@dataclass
class AgentResult:
    """Result of agent execution."""

    answer: str
    steps: List[Step] = field(default_factory=list)
    total_steps: int = 0
    success: bool = True
    error: Optional[str] = None
    latency_ms: float = 0.0


class ReActAgent:
    """ReAct (Reason + Act) agent for multi-step reasoning."""

    def __init__(
        self,
        llm: BaseInferenceBackend,
        tools: List[BaseTool],
        config: AgentConfig,
    ):
        self.llm = llm
        self.config = config
        self.prompt_builder = PromptBuilder()

        # Build tool registry
        self.tool_registry = ToolRegistry()
        for tool in tools:
            self.tool_registry.register(tool)

        # Build system prompt
        self.system_prompt = self.prompt_builder.build_system_prompt(tools)

        # Prometheus metrics
        self._runs_counter = PrometheusCounter(
            "agent_runs_total",
            "Total agent runs",
            labelnames=["success"],
        )
        self._latency_histogram = Histogram(
            "agent_latency_seconds",
            "Agent latency",
        )
        self._steps_histogram = Histogram(
            "agent_steps_per_run",
            "Steps per agent run",
        )

    async def run(self, query: str) -> AgentResult:
        """Run the agent to completion."""
        start_time = time.time()
        scratchpad: List[Step] = []

        try:
            for step_num in range(1, self.config.max_steps + 1):
                # THINK: Build prompt and call LLM
                user_prompt = self.prompt_builder.build_user_prompt(query, scratchpad)
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                generation = await self.llm.chat(messages)
                llm_output = generation.text

                if self.config.verbose:
                    logger.info(
                        "agent.think",
                        step=step_num,
                        llm_output_preview=llm_output[:200],
                    )

                # PARSE: Extract thought, action, action_input
                parsed = self.prompt_builder.parse_llm_output(llm_output)

                # If Final Answer, return
                if parsed.is_final:
                    latency_ms = (time.time() - start_time) * 1000

                    result = AgentResult(
                        answer=parsed.final_answer or "",
                        steps=scratchpad,
                        total_steps=step_num,
                        success=True,
                        latency_ms=latency_ms,
                    )

                    self._runs_counter.labels(success="true").inc()
                    self._latency_histogram.observe(latency_ms / 1000)
                    self._steps_histogram.observe(step_num)

                    if self.config.verbose:
                        logger.info(
                            "agent.finished",
                            step=step_num,
                            answer_preview=parsed.final_answer[:100] if parsed.final_answer else "",
                        )

                    return result

                # ACT: Execute tool
                tool_name = parsed.action_name
                tool = self.tool_registry.get(tool_name) if tool_name else None

                if not tool:
                    observation = (
                        f"Error: tool '{tool_name}' not found. Available tools: "
                        f"{', '.join(self.tool_registry.tools.keys())}"
                    )
                else:
                    try:
                        tool_result = await tool.call(**parsed.action_input)
                        observation = (
                            tool_result.output
                            if tool_result.success
                            else f"Error: {tool_result.error}"
                        )
                    except Exception as e:
                        observation = f"Error: {str(e)}"

                # OBSERVE: Record step
                step = Step(
                    thought=parsed.thought,
                    action_name=tool_name or "unknown",
                    action_input=parsed.action_input or {},
                    observation=observation,
                )
                scratchpad.append(step)

                if self.config.verbose:
                    obs_preview = observation[:100] if len(observation) > 100 else observation
                    logger.info(
                        "agent.act",
                        step=step_num,
                        tool=tool_name,
                        observation_preview=obs_preview,
                    )

            # Max steps reached without Final Answer
            latency_ms = (time.time() - start_time) * 1000

            result = AgentResult(
                answer="Max steps reached without answer",
                steps=scratchpad,
                total_steps=self.config.max_steps,
                success=False,
                error="Max steps exceeded",
                latency_ms=latency_ms,
            )

            self._runs_counter.labels(success="false").inc()
            self._latency_histogram.observe(latency_ms / 1000)
            self._steps_histogram.observe(self.config.max_steps)

            logger.warning("agent.max_steps_exceeded")
            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            result = AgentResult(
                answer="",
                steps=scratchpad,
                total_steps=len(scratchpad),
                success=False,
                error=str(e),
                latency_ms=latency_ms,
            )

            self._runs_counter.labels(success="false").inc()
            self._latency_histogram.observe(latency_ms / 1000)

            logger.error("agent.error", error=str(e))
            return result

    async def run_stream(self, query: str) -> AsyncGenerator[str, None]:
        """Stream agent steps as they happen."""
        scratchpad: List[Step] = []

        try:
            for step_num in range(1, self.config.max_steps + 1):
                # THINK: Build prompt and call LLM
                user_prompt = self.prompt_builder.build_user_prompt(query, scratchpad)
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                generation = await self.llm.chat(messages)
                llm_output = generation.text

                # PARSE
                parsed = self.prompt_builder.parse_llm_output(llm_output)

                # If Final Answer, yield and return
                if parsed.is_final:
                    yield f"Final Answer: {parsed.final_answer}\n"
                    return

                # ACT: Execute tool
                tool_name = parsed.action_name
                tool = self.tool_registry.get(tool_name) if tool_name else None

                if not tool:
                    observation = (
                        f"Error: tool '{tool_name}' not found. Available tools: "
                        f"{', '.join(self.tool_registry.tools.keys())}"
                    )
                else:
                    try:
                        tool_result = await tool.call(**parsed.action_input)
                        observation = (
                            tool_result.output
                            if tool_result.success
                            else f"Error: {tool_result.error}"
                        )
                    except Exception as e:
                        observation = f"Error: {str(e)}"

                # OBSERVE: Record and yield
                step = Step(
                    thought=parsed.thought,
                    action_name=tool_name or "unknown",
                    action_input=parsed.action_input or {},
                    observation=observation,
                )
                scratchpad.append(step)

                yield f"Step {step_num} | Action: {tool_name} | Observation: {observation[:100]}\n"

        except Exception as e:
            yield f"Error: {str(e)}\n"
