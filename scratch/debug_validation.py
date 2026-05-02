import json
import asyncio
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.document_agent import structured_llm, validate_extracted_data

async def main():
    data = json.load(open("data/eval_dataset.json"))
    
    # Find a valid case
    case = next(c for c in data if c["scenario_type"] == "valid")
    
    prompt = f"Analyze extracted claim text.\nExtracted Text: {case['input']['extracted_text']}\nDamage Summary: "
    print("Calling LLM...")
    res = await structured_llm.ainvoke(prompt)
    print("LLM Result:", res)
    
    missing = validate_extracted_data(res)
    print("Missing:", missing)

asyncio.run(main())
