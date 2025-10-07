import asyncio
from typing import List, Dict

# Import the judge function
from models import call_gemini_judge

async def verify_and_merge(outputs: Dict[str, str], evidence: List[str], prompt: str) -> Dict:
    """
    Verifies and merges the outputs from different models using Gemini as a judge.
    
    Args:
        outputs (Dict[str, str]): A dictionary where keys are model names and values are their outputs.
                                  Expected keys: 'gpt' and 'deepseek'.
        evidence (List[str]): A list of evidence snippets. In this strategy, the raw
                              model outputs are used as sources.
        prompt (str): The original user prompt.
        
    Returns:
        Dict: A dictionary containing the final output from the judge, a confidence score,
              the original model outputs as sources, and the per-model breakdown.
    """
    
    gpt_output = outputs.get("gpt", "No output from GPT.")
    deepseek_output = outputs.get("deepseek", "No output from Deepseek.")

    # Use Gemini to arbitrate between the two models
    final_output = await call_gemini_judge(
        gpt_output=gpt_output,
        deepseek_output=deepseek_output,
        prompt=prompt
    )
    
    # Check if the judge call failed and returned a fallback
    if final_output == gpt_output:
        confidence = 0.5 # Lower confidence if judge failed
    else:
        confidence = 0.9 # High confidence since it was judged by a powerful model

    # Return the final structure
    return {
        "final_output": final_output,
        "confidence": confidence,
        "sources": [f"GPT: {gpt_output}", f"Deepseek: {deepseek_output}"],
        "per_model": outputs,
        "consensus_method": "gemini_judge"
    }
