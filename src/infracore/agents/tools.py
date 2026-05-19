"""Tool registry and built-in tools for ReAct agent."""

import ast
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import structlog

from src.infracore.retrieval.base import BaseRetriever

logger = structlog.get_logger()


class ToolError(Exception):
    """Raised when a tool fails."""

    pass


class ToolResult:
    """Result of a tool execution."""

    def __init__(
        self,
        output: str,
        success: bool = True,
        error: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.output = output
        self.success = success
        self.error = error
        self.meta = meta or {}


class BaseTool:
    """Abstract base for all tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    async def call(self, **kwargs) -> ToolResult:
        """Execute the tool. Must be async."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement call()")


class RAGSearchTool(BaseTool):
    """Search the knowledge base using retrieval."""

    def __init__(self, retriever: BaseRetriever):
        super().__init__(
            name="rag_search",
            description="Search the knowledge base for relevant information",
        )
        self.retriever = retriever

    async def call(self, query: str, top_k: int = 3, **kwargs) -> ToolResult:
        """Search and format results."""
        try:
            results = await self.retriever.retrieve(query, top_k=top_k)

            # Format as numbered list
            output_lines = []
            for i, result in enumerate(results, 1):
                score = getattr(result, "score", 0.0)
                text = getattr(result, "text", getattr(result, "chunk", {}).get("text", ""))
                output_lines.append(f"{i}. [score={score:.2f}] {text}")

            output = "\n".join(output_lines) if output_lines else "No results found."

            logger.info("rag_search.executed", query_len=len(query), result_count=len(results))
            return ToolResult(output=output, success=True)

        except Exception as e:
            error_msg = f"RAG search failed: {str(e)}"
            logger.error("rag_search.failed", error=str(e))
            return ToolResult(output="", success=False, error=error_msg)


class CalculatorTool(BaseTool):
    """Evaluate mathematical expressions safely."""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Evaluate mathematical expressions safely",
        )
        # Allowed operators
        self.allowed_ops = {
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.Mod,
            ast.FloorDiv,
            ast.USub,
            ast.UAdd,
        }

    async def call(self, expression: str, **kwargs) -> ToolResult:
        """Evaluate a mathematical expression."""
        try:
            # Safety check: reject unsafe tokens
            unsafe_tokens = ["import", "exec", "eval", "__", "open", "input", "os", "sys"]
            if any(token in expression.lower() for token in unsafe_tokens):
                raise ToolError(f"Unsafe expression: contains restricted token")

            # Parse expression
            tree = ast.parse(expression, mode="eval")

            # Validate nodes are safe
            self._validate_ast(tree.body)

            # Evaluate safely using compile/eval with restricted namespace
            code = compile(tree, "<string>", "eval")
            result = eval(code, {"__builtins__": {}}, {})

            output = f"Result: {result}"
            logger.info("calculator.executed", expression=expression, result=result)
            return ToolResult(output=output, success=True)

        except ToolError as e:
            logger.error("calculator.unsafe", expression=expression)
            return ToolResult(output="", success=False, error=str(e))
        except Exception as e:
            error_msg = f"Calculation failed: {str(e)}"
            logger.error("calculator.failed", error=str(e))
            return ToolResult(output="", success=False, error=error_msg)

    def _validate_ast(self, node: ast.AST) -> None:
        """Recursively validate AST contains only safe operations."""
        if isinstance(node, ast.Constant):
            return  # Numbers, strings are OK
        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in self.allowed_ops:
                raise ToolError(f"Unsafe operation: {type(node.op).__name__}")
            self._validate_ast(node.operand)
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in self.allowed_ops:
                raise ToolError(f"Unsafe operation: {type(node.op).__name__}")
            self._validate_ast(node.left)
            self._validate_ast(node.right)
        elif isinstance(node, (ast.Call, ast.Attribute, ast.Name, ast.Import)):
            raise ToolError(f"Unsafe node type: {type(node).__name__}")
        else:
            # Allow Expr wrapper
            for child in ast.iter_child_nodes(node):
                self._validate_ast(child)


class WordCountTool(BaseTool):
    """Count words, characters, or sentences."""

    def __init__(self):
        super().__init__(
            name="word_count",
            description="Count words, characters, or sentences in text",
        )

    async def call(
        self, text: str, count_type: str = "words", **kwargs
    ) -> ToolResult:
        """Count words, chars, or sentences."""
        try:
            if count_type == "words":
                count = len(text.split())
                output = f"Word count: {count}"
            elif count_type == "chars":
                count = len(text)
                output = f"Character count: {count}"
            elif count_type == "sentences":
                # Simple sentence splitting on . ! ?
                sentences = re.split(r"[.!?]+", text)
                count = len([s for s in sentences if s.strip()])
                output = f"Sentence count: {count}"
            else:
                raise ValueError(f"Unknown count_type: {count_type}")

            logger.info("word_count.executed", count_type=count_type, count=count)
            return ToolResult(output=output, success=True)

        except Exception as e:
            error_msg = f"Word count failed: {str(e)}"
            logger.error("word_count.failed", error=str(e))
            return ToolResult(output="", success=False, error=error_msg)


class CurrentTimeTool(BaseTool):
    """Get the current date and time."""

    def __init__(self):
        super().__init__(
            name="current_time",
            description="Get the current date and time",
        )

    async def call(self, timezone: str = "UTC", **kwargs) -> ToolResult:
        """Get current time with timezone."""
        try:
            now = datetime.now()
            # Format: YYYY-MM-DD HH:MM:SS UTC
            output = f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} {timezone}"

            logger.info("current_time.executed", timezone=timezone, timestamp=str(now))
            return ToolResult(output=output, success=True)

        except Exception as e:
            error_msg = f"Time retrieval failed: {str(e)}"
            logger.error("current_time.failed", error=str(e))
            return ToolResult(output="", success=False, error=error_msg)


class ToolRegistry:
    """Registry for tools."""

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool
        logger.info("tool.registered", name=tool.name)

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> str:
        """Get formatted list of tool names and descriptions."""
        lines = []
        for name, tool in self.tools.items():
            lines.append(f"- {name}: {tool.description}")
        return "\n".join(lines)
