#!/usr/bin/env python
"""Smoke test for ReActAgent with real Ollama backend."""

import asyncio

from src.infracore.agents.react_agent import AgentConfig, ReActAgent
from src.infracore.agents.tools import CalculatorTool, CurrentTimeTool, WordCountTool
from src.infracore.inference.ollama_backend import OllamaBackend, OllamaConfig


async def main():
    """Run agent smoke test."""
    print("🚀 Starting ReActAgent smoke test with Ollama backend...\n")

    # Initialize LLM backend
    llm_config = OllamaConfig(
        model="llama3.2:1b",
        max_tokens=200,
        temperature=0.1,
        base_url="http://localhost:11434",
    )
    llm = OllamaBackend(llm_config)

    # Check availability
    available = await llm.is_available()
    if not available:
        print("❌ Ollama server not available at http://localhost:11434")
        print("   Start with: ollama serve &")
        print("   Then: ollama pull llama3.2:1b")
        return

    print("✅ Ollama backend available\n")

    # Initialize tools
    tools = [CalculatorTool(), WordCountTool(), CurrentTimeTool()]

    # Create agent
    config = AgentConfig(max_steps=5, verbose=True)
    agent = ReActAgent(llm=llm, tools=tools, config=config)

    # Run query
    query = "What is 15 * 7 + 42?"
    print(f"❓ Query: {query}\n")
    print("=" * 60)

    result = await agent.run(query)

    print("=" * 60)
    print(f"\n📋 Result Summary:")
    print(f"   Answer: {result.answer}")
    print(f"   Steps: {result.total_steps}")
    print(f"   Success: {result.success}")
    print(f"   Latency: {result.latency_ms:.0f}ms\n")

    if result.steps:
        print("📝 Step Details:")
        for i, step in enumerate(result.steps, 1):
            print(f"\n   Step {i}:")
            print(f"      Thought: {step.thought[:80]}...")
            print(f"      Action: {step.action_name}")
            print(f"      Input: {step.action_input}")
            print(f"      Observation: {step.observation[:80]}...")

    # Expected: agent should calculate 15 * 7 + 42 = 147
    if result.success and "147" in result.answer:
        print("\n✅ TEST PASSED: Agent correctly calculated the answer!")
    elif result.success:
        print(f"\n⚠️  TEST COMPLETED: Agent finished but answer may not be exact: {result.answer}")
    else:
        print(f"\n❌ TEST FAILED: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
