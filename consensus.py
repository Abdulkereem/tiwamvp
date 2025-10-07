from models import call_gemini_judge

async def verify_and_merge(outputs: dict, evidence: list, prompt: str) -> dict:
    """
    Core consensus logic. 
    - If models agree, return the common output.
    - If they disagree, use the Gemini Judge to make a final decision.
    """
    gpt_output = outputs.get("gpt")
    deepseek_output = outputs.get("deepseek")

    # For now, we'll use a simple majority or fallback to the judge.
    # A more sophisticated method could involve semantic similarity checks.
    if gpt_output == deepseek_output:
        return {"final_output": gpt_output, "consensus_method": "unanimous_agreement"}
    else:
        # Let Gemini decide
        final_answer = await call_gemini_judge(gpt_output, deepseek_output, prompt)
        return {"final_output": final_answer, "consensus_method": "gemini_judge_arbitration"}
