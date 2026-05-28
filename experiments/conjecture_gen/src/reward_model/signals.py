import networkx as nx
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from knowledge_graph.graph import KnowledgeGraph, EdgeType

def compute_connectivity_score(kg: KnowledgeGraph, source_id: str, target_id: str) -> float:
    """
    Computes the connectivity score for a proposed edge between source_id and target_id.
    
    A high score is awarded if the edge bridges two currently distant or disconnected 
    regions of the graph. We measure this using the shortest path distance in the 
    undirected version of the graph *before* the edge is added.
    """
    if source_id not in kg.graph or target_id not in kg.graph:
        return 0.0

    undirected_graph = kg.graph.to_undirected()
    
    try:
        # Calculate shortest path length without considering edge weights
        distance = nx.shortest_path_length(undirected_graph, source=source_id, target=target_id)
        
        # If distance is 1, they are already connected. 
        # If distance is > 1, they are bridging distant nodes.
        if distance == 1:
            return 0.1 # Very low score for redundant connections
        
        # Logarithmic scaling to reward distant connections without blowing up
        return float(min(distance, 5)) / 5.0 
        
    except nx.NetworkXNoPath:
        # They are completely disconnected! Bridging them is highly interesting.
        return 1.0


def compute_generality_score(kg: KnowledgeGraph, source_id: str, target_id: str, edge_type: EdgeType) -> float:
    """
    Computes the generality score for a proposed edge.
    
    If the edge is a GENERALIZES edge (source_id GENERALIZES target_id), the score 
    scales with the number of theorems/objects that target_id already generalizes.
    Finding a master theorem that generalizes an already general theorem is highly rewarded.
    """
    if edge_type != EdgeType.GENERALIZES:
        return 0.0
        
    if target_id not in kg.graph:
        return 0.0
        
    # Extract a subgraph of only GENERALIZES edges
    generalizes_edges = [
        (u, v) for u, v, data in kg.graph.edges(data=True) 
        if data.get('etype') == EdgeType.GENERALIZES
    ]
    
    gen_subgraph = nx.DiGraph()
    gen_subgraph.add_nodes_from(kg.graph.nodes())
    gen_subgraph.add_edges_from(generalizes_edges)
    
    # We want to find all nodes that are reachable from target_id in the GENERALIZES subgraph.
    # Since generalization flows from general -> specific (A generalizes B), 
    # we count descendants of target_id.
    try:
        descendants = nx.descendants(gen_subgraph, target_id)
        num_special_cases = len(descendants)
        
        # Base score of 0.5 just for generalizing something.
        # Add 0.1 for every additional thing it recursively generalizes.
        return min(0.5 + (num_special_cases * 0.1), 1.0)
        
    except nx.NetworkXError:
        return 0.0

if __name__ == "__main__":
    from knowledge_graph.seed import generate_seed_graph
    
    print("Initializing Seed Graph...")
    kg = generate_seed_graph()
    
    print("\n--- Testing Connectivity Score ---")
    # Test 1: Connect two completely disconnected objects (e.g. if we add a new dummy node)
    kg.add_node("obj:dummy", node_type=None)
    score1 = compute_connectivity_score(kg, "obj:triangle", "obj:dummy")
    print(f"Connectivity between Triangle and Dummy (Disconnected): {score1}")
    
    # Test 2: Connect Triangle directly to Edge Count
    score2 = compute_connectivity_score(kg, "obj:triangle", "quant:edge_count")
    print(f"Connectivity between Triangle and Edge Count (Distance 2): {score2}")
    
    print("\n--- Testing Generality Score ---")
    # Test 3: Imagine a new super theorem generalizing Turan
    kg.add_node("thm:super_turan", node_type=None)
    gen_score = compute_generality_score(kg, "thm:super_turan", "thm:turan", EdgeType.GENERALIZES)
    # Turan generalises Mantel, so Turan has 1 descendant. 
    # Base 0.5 + (1 * 0.1) = 0.6
    print(f"Generality Score for generalizing Turan's Theorem: {gen_score}")
