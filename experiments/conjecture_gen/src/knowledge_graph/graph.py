import networkx as nx
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Tuple

class NodeType(Enum):
    OBJECT = auto()        # e.g., "Complete Graph", "Turán Graph"
    THEOREM = auto()       # e.g., "Mantel's Theorem"
    CONJECTURE = auto()    # e.g., "Erdős-Faber-Lovász conjecture"
    QUANTITY = auto()      # e.g., "Edge Count", "Clique Number"
    TECHNIQUE = auto()     # e.g., "Probabilistic Method", "Induction"

class EdgeType(Enum):
    IMPLIES = auto()           # Theorem A implies Theorem B
    GENERALIZES = auto()       # Theorem A generalizes Theorem B
    TECHNIQUE_USED = auto()    # Theorem A proved using Technique B
    COUNTEREXAMPLE = auto()    # Object A is counterexample to Conjecture B
    BOUNDS = auto()            # Theorem A provides bound for Quantity B
    HAS_CONDITION = auto()     # Theorem A requires Object/Property B
    SPECIAL_CASE = auto()      # Object A is a special case of Object B

class KnowledgeGraph:
    def __init__(self):
        # We use a MultiDiGraph because multiple different typed edges 
        # can exist between the same two nodes.
        self.graph = nx.MultiDiGraph()

    def add_node(self, node_id: str, node_type: NodeType, properties: Dict[str, Any] = None):
        """Adds a typed node to the knowledge graph."""
        if properties is None:
            properties = {}
        
        self.graph.add_node(
            node_id, 
            type=node_type, 
            **properties
        )

    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType, properties: Dict[str, Any] = None):
        """Adds a typed directed edge between two nodes."""
        if properties is None:
            properties = {}
            
        if source_id not in self.graph or target_id not in self.graph:
            raise ValueError(f"Both source '{source_id}' and target '{target_id}' must exist.")
            
        self.graph.add_edge(
            source_id,
            target_id,
            etype=edge_type,
            **properties
        )

    def get_node(self, node_id: str) -> Dict[str, Any]:
        return self.graph.nodes[node_id]
        
    def get_edges(self, source_id: Optional[str] = None, target_id: Optional[str] = None) -> List[Tuple]:
        """Retrieve edges, optionally filtered by source or target."""
        if source_id and target_id:
            return list(self.graph.edges(source_id, target_id, data=True))
        elif source_id:
            return list(self.graph.out_edges(source_id, data=True))
        elif target_id:
            return list(self.graph.in_edges(target_id, data=True))
        else:
            return list(self.graph.edges(data=True))

    def summary(self) -> str:
        """Returns a string summary of the graph's size and composition."""
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        
        node_counts = {}
        for _, data in self.graph.nodes(data=True):
            n_type = data.get('type')
            if n_type:
                node_counts[n_type.name] = node_counts.get(n_type.name, 0) + 1
                
        edge_counts = {}
        for _, _, data in self.graph.edges(data=True):
            e_type = data.get('etype')
            if e_type:
                edge_counts[e_type.name] = edge_counts.get(e_type.name, 0) + 1

        summary_lines = [
            f"Knowledge Graph Summary:",
            f"Total Nodes: {num_nodes}",
            f"Total Edges: {num_edges}",
            "--- Node Types ---"
        ]
        for nt, count in node_counts.items():
            summary_lines.append(f"  {nt}: {count}")
            
        summary_lines.append("--- Edge Types ---")
        for et, count in edge_counts.items():
            summary_lines.append(f"  {et}: {count}")
            
        return "\n".join(summary_lines)
