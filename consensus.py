import asyncio
from models import call_gemini_judge
from sentence_transformers import SentenceTransformer, util

# Load sentence embedding model once globally
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

async def encode_outputs(outputs):
    """Asynchronously encode outputs into embeddings."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, similarity_model.encode, outputs, True)

async def compute_consensus(outputs: dict):
    """Compute semantic consensus asynchronously."""
    model_outputs = list(outputs.values())
    model_names = list(outputs.keys())

    # Encode asynchronously
    embeddings = await encode_outputs(model_outputs)
    sim_matrix = util.pytorch_cos_sim(embeddings, embeddings)

    # Find the most semantically central output
    avg_sims = sim_matrix.mean(dim=1)
    top_idx = avg_sims.argmax().item()

    return {
        "output": model_outputs[top_idx],
        "confidence": avg_sims[top_idx].item(),
        "source_model": model_names[top_idx]
    }

async def verify_and_merge(outputs: dict, evidence: list, prompt: str) -> dict:
    """
    Async-parallel TIWA consensus engine:
    - Computes semantic agreement across multiple models.
    - If confidence < threshold, invokes Gemini Judge concurrently.
    """
    consensus_task = asyncio.create_task(compute_consensus(outputs))
    consensus = await consensus_task

    if consensus["confidence"] >= 0.85:
        # Consensus strong enough, no need for arbitration
        return {
            "final_output": consensus["output"],
            "consensus_method": "semantic_agreement",
            "confidence": consensus["confidence"],
            "source_model": consensus["source_model"]
        }

    # Run Gemini Judge arbitration concurrently with re-checks or evidence synthesis
    arbitration_task = asyncio.create_task(
        call_gemini_judge(list(outputs.values()), evidence, prompt)
    )

    final_answer = await arbitration_task

    return {
        "final_output": final_answer,
        "consensus_method": "gemini_judge_arbitration",
        "confidence": consensus["confidence"],
        "source_model": consensus["source_model"]
    }
