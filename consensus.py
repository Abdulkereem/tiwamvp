from typing import List, Dict

def verify_and_merge(outputs: Dict[str, str], evidence: List[str]) -> Dict:
    """
    Verifies and merges the outputs from different models.
    
    Args:
        outputs (Dict[str, str]): A dictionary where keys are model names and values are their outputs.
        evidence (List[str]): A list of evidence snippets from the retrieval model.
        
    Returns:
        Dict: A dictionary containing the merged output, confidence score, and sources.
    """
    
    # For this MVP, we'll use a simplified consensus mechanism.
    # We'll consider the response from GPT as the base and check for semantic similarity.
    
    base_output = outputs.get("gpt", "")
    
    # Simple confidence score based on the number of models that agree with the base output
    agreement_count = 0
    for model_output in outputs.values():
        if model_output.startswith(base_output[:10]):  # Simplified similarity check
            agreement_count += 1
            
    confidence = (0.3 + 0.2 * agreement_count + 0.2 * (1 if evidence else 0))
    
    # For the MVP, we will just return the base output as the final merged output.
    # A more advanced implementation would involve more sophisticated merging logic.
    merged_output = base_output
    
    return {
        "final_output": merged_output,
        "confidence": confidence,
        "sources": evidence,
        "per_model": outputs
    }

def compute_confidence(models: List[str], evidence: List[str]) -> float:
    """
    Computes a confidence score based on model agreement and retrieval evidence.
    
    Args:
        models (List[str]): A list of models that produced an output.
        evidence (List[str]): A list of evidence snippets.
        
    Returns:
        float: A confidence score between 0 and 1.
    """
    
    # Simplified confidence calculation
    num_agree = len(models)
    evidence_score = 1 if evidence else 0
    
    confidence = 0.3 + 0.2 * num_agree + 0.2 * evidence_score
    
    return min(1.0, confidence) # Ensure confidence is not more than 1.0
