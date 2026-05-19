"""
BaseAgent – Abstract interface for agentic loops.

ReAct pattern with typed tool registry. Pure async.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field


class AgentConfig(BaseModel):
    """Pydantic config for agents."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(..., description="LLM model ID")
    max_steps: int = Field(default=10, description="Max reasoning steps")
    temperature: float = Field(default=0.7, description="Sampling temperature")


@dataclass
class AgentResult:
    """Result of agent execution."""

    final_answer: str
    steps: List[Dict[str, Any]]
    total_tokens: int


class BaseTool(ABC):
    """Base class for agent tools."""

    name: str
    description: str

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """Execute the tool with given arguments."""
        pass


class BaseAgent(ABC):
    """
    Abstract base class for agentic loops.

    Subclasses must implement:
    - run(query: str) -> AgentResult
    """

    def __init__(self, config: AgentConfig, tools: List[BaseTool]):
        self.config = config
        self.tools = {tool.name: tool for tool in tools}

    @abstractmethod
    async def run(self, query: str) -> AgentResult:
        """
        Run agent reasoning loop.

        Args:
            query: User query

        Returns:
            AgentResult with final answer + execution trace
        """
        pass
