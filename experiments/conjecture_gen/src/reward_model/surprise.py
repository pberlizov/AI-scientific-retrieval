import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
from typing import Dict, Tuple

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from knowledge_graph.graph import KnowledgeGraph

class GraphAutoEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphAutoEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_index):
        # Inner product decoder
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_index)

def kg_to_pyg_data(kg: KnowledgeGraph) -> Tuple[Data, Dict[str, int]]:
    """Converts the NetworkX KnowledgeGraph to PyTorch Geometric Data."""
    node_mapping = {node_id: i for i, node_id in enumerate(kg.graph.nodes())}
    
    # Create simple one-hot features for Node Types
    num_nodes = len(node_mapping)
    num_features = 5 # OBJECT, THEOREM, CONJECTURE, QUANTITY, TECHNIQUE
    x = torch.zeros((num_nodes, num_features))
    
    # A bit hacky: just assign a one-hot based on Enum value
    for node_id, data in kg.graph.nodes(data=True):
        idx = node_mapping[node_id]
        n_type = data.get('type')
        if n_type:
            # -1 because Enum starts at 1
            x[idx, n_type.value - 1] = 1.0 
            
    edges = list(kg.graph.edges())
    edge_index = torch.tensor([
        [node_mapping[u] for u, v in edges],
        [node_mapping[v] for u, v in edges]
    ], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index)
    return data, node_mapping

def train_gnn(kg: KnowledgeGraph, epochs=100) -> GraphAutoEncoder:
    """Trains a simple GAE on the current Knowledge Graph to learn link probabilities."""
    data, _ = kg_to_pyg_data(kg)
    
    model = GraphAutoEncoder(in_channels=5, hidden_channels=16, out_channels=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Create positive edges (existing) and negative edges (non-existing)
    # For a tiny graph, we can just use all non-edges
    import itertools
    all_possible_edges = set(itertools.product(range(data.num_nodes), repeat=2))
    existing_edges = set(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
    negative_edges = list(all_possible_edges - existing_edges)
    
    # Sample negative edges to balance classes
    import random
    if len(negative_edges) > len(existing_edges):
        negative_edges = random.sample(negative_edges, len(existing_edges))
        
    if not negative_edges:
        # Fully connected graph
        return model

    neg_edge_index = torch.tensor(negative_edges, dtype=torch.long).t()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        
        pos_pred = model.decode(z, data.edge_index)
        neg_pred = model.decode(z, neg_edge_index)
        
        # Binary Cross Entropy
        pos_loss = -torch.log(torch.sigmoid(pos_pred) + 1e-15).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(neg_pred) + 1e-15).mean()
        loss = pos_loss + neg_loss
        
        loss.backward()
        optimizer.step()
        
    return model

def compute_surprise_score(kg: KnowledgeGraph, model: GraphAutoEncoder, source_id: str, target_id: str) -> float:
    """Computes Surprise Score: 1.0 - P(edge existing)."""
    data, node_mapping = kg_to_pyg_data(kg)
    
    if source_id not in node_mapping or target_id not in node_mapping:
        return 1.0 # Completely unseen nodes are very surprising
        
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        edge = torch.tensor([[node_mapping[source_id]], [node_mapping[target_id]]], dtype=torch.long)
        pred_logit = model.decode(z, edge)
        prob = torch.sigmoid(pred_logit).item()
        
    # Surprise is the inverse of probability
    return 1.0 - prob

if __name__ == "__main__":
    from knowledge_graph.seed import generate_seed_graph
    
    kg = generate_seed_graph()
    model = train_gnn(kg)
    
    # Check probability of an existing edge (should be low surprise)
    surprise_existing = compute_surprise_score(kg, model, "thm:turan", "thm:mantel")
    print(f"Surprise for existing edge (Turan -> Mantel): {surprise_existing:.4f}")
    
    # Check probability of a random non-existing edge (should be high surprise)
    surprise_fake = compute_surprise_score(kg, model, "quant:edge_count", "obj:complete_graph")
    print(f"Surprise for fake edge (Edge Count -> Complete Graph): {surprise_fake:.4f}")
