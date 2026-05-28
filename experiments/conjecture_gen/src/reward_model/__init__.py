from .signals import compute_connectivity_score, compute_generality_score
from .surprise import train_gnn, compute_surprise_score, GraphAutoEncoder

def compute_interestingness(kg, model: GraphAutoEncoder, source_id: str, target_id: str, edge_type) -> float:
    """Computes the aggregated interestingness score."""
    conn = compute_connectivity_score(kg, source_id, target_id)
    gen = compute_generality_score(kg, source_id, target_id, edge_type)
    surp = compute_surprise_score(kg, model, source_id, target_id)
    
    # Simple linear combination with equal weights for scaffolding
    return (conn + gen + surp) / 3.0
