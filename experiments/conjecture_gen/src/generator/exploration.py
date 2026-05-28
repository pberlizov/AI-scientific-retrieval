import torch
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from knowledge_graph.graph import KnowledgeGraph
from reward_model.surprise import kg_to_pyg_data, GraphAutoEncoder

def compute_curiosity_bonus(kg: KnowledgeGraph, model: GraphAutoEncoder, source_id: str, target_id: str) -> float:
    """
    Computes epistemic uncertainty (curiosity). 
    Bonus is maximized when P(edge) is near 0.5.
    """
    data, node_mapping = kg_to_pyg_data(kg)
    
    if source_id not in node_mapping or target_id not in node_mapping:
        return 0.0
        
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        edge = torch.tensor([[node_mapping[source_id]], [node_mapping[target_id]]], dtype=torch.long)
        pred_logit = model.decode(z, edge)
        prob = torch.sigmoid(pred_logit).item()
        
    # Uncertainty is maxed at p=0.5. We can use a simple parabola: 1 - 4*(p - 0.5)^2
    # Yields 1.0 at p=0.5, and 0.0 at p=0.0 and p=1.0.
    uncertainty = 1.0 - 4.0 * ((prob - 0.5) ** 2)
    
    # Scale bonus to be max 0.3 so it doesn't overwhelm the structural signals
    return max(0.0, float(uncertainty) * 0.3)
