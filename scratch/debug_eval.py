import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.document_agent import structured_llm

async def main():
    prompt = "Analyze this: Policy Number: POL-123"
    try:
        res = await structured_llm.ainvoke(prompt)
        print("Success:", res)
    except Exception as e:
        print("Error:", repr(e))

asyncio.run(main())
