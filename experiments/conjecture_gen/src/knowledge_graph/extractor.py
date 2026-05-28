import os
import re
from typing import List, Dict, Any, Tuple
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from knowledge_graph.graph import KnowledgeGraph, NodeType, EdgeType

class RobustLeanParser:
    """A parser that handles real Lean 4 syntax from Mathlib4."""
    
    def __init__(self):
        # This matches optional docstrings, optional attributes/modifiers, 
        # the keyword, the name, and the signature up to := or 'where'
        # It's heuristic but much better than before.
        self.decl_pattern = re.compile(
            r'(?:/--\s*(.*?)\s*-/\s*)?'                 # Optional docstring
            r'(?:@\[.*?\]\s*)*'                         # Optional attributes like @[simp]
            r'(?:(?:protected|private|noncomputable)\s+)*' # Optional modifiers
            r'(def|theorem|lemma|abbrev|structure|class)\s+' # Keyword
            r'([a-zA-Z0-9_]+)'                          # Name
            r'(.*?)'                                    # Signature
            r'(?::=|where)',                            # End of signature
            re.DOTALL | re.MULTILINE
        )

    def parse_string(self, content: str, filename: str = "unknown") -> List[Dict[str, str]]:
        blocks = []
        for match in self.decl_pattern.finditer(content):
            docstring, keyword, name, signature = match.groups()
            
            node_type = "theorem" if keyword in ["theorem", "lemma"] else "def"
            
            blocks.append({
                "type": node_type,
                "name": name.strip(),
                "docstring": docstring.strip() if docstring else "",
                "signature": signature.strip(),
                "filename": filename
            })
            
        return blocks

    def parse_file(self, filepath: str) -> List[Dict[str, str]]:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        filename = os.path.basename(filepath)
        return self.parse_string(content, filename)


class HeuristicEdgeClassifier:
    """Uses syntactic and import analysis to find real edges without an LLM."""
    
    def __init__(self):
        pass

    def classify(self, block: Dict[str, str], existing_nodes: Dict[str, str]) -> List[Tuple[str, EdgeType]]:
        """
        existing_nodes is a dict mapping node_name to node_id.
        """
        edges = []
        signature = block['signature']
        
        # 1. Reference Edges: If a node's name appears in the signature, it uses it.
        # This gives us HAS_CONDITION, BOUNDS, or simply a USES relationship.
        # For simplicity, we'll map object references to HAS_CONDITION
        # and theorem references (rare in signatures, but possible) to USES.
        
        # Tokenize signature roughly to avoid substring matches (e.g., 'graph' matching 'complete_graph')
        tokens = set(re.findall(r'[a-zA-Z0-9_]+', signature))
        
        for node_name, node_id in existing_nodes.items():
            if node_name == block['name']:
                continue
                
            if node_name in tokens:
                if node_id.startswith("obj:"):
                    edges.append((node_id, EdgeType.HAS_CONDITION))
                else:
                    edges.append((node_id, EdgeType.TECHNIQUE_USED)) # Hack for general usage
                    
        return edges

def build_graph_from_mathlib(data_dir: str) -> KnowledgeGraph:
    kg = KnowledgeGraph()
    parser = RobustLeanParser()
    classifier = HeuristicEdgeClassifier()
    
    all_blocks = []
    print(f"Parsing files in {data_dir}...")
    for filename in os.listdir(data_dir):
        if filename.endswith(".lean"):
            filepath = os.path.join(data_dir, filename)
            all_blocks.extend(parser.parse_file(filepath))
            
    print(f"Extracted {len(all_blocks)} declarations.")
            
    # Add all nodes first
    existing_nodes = {} # map name -> node_id
    for block in all_blocks:
        node_id = f"obj:{block['name']}" if block['type'] == "def" else f"thm:{block['name']}"
        node_type = NodeType.OBJECT if block['type'] == "def" else NodeType.THEOREM
        
        kg.add_node(node_id, node_type, {
            "name": block['name'], 
            "docstring": block['docstring'],
            "filename": block['filename']
        })
        existing_nodes[block['name']] = node_id
        
    # Classify edges
    print("Classifying edges based on signature references...")
    edge_count = 0
    for block in all_blocks:
        source_id = f"obj:{block['name']}" if block['type'] == "def" else f"thm:{block['name']}"
        edges = classifier.classify(block, existing_nodes)
        for target_id, edge_type in edges:
            kg.add_edge(source_id, target_id, edge_type)
            edge_count += 1
                
    print(f"Created {edge_count} edges.")
    return kg

def extract_from_text(text: str, existing_kg: KnowledgeGraph) -> List[Tuple[str, str, EdgeType]]:
    """
    Parses a generated Lean 4 string, identifies what nodes it defines, 
    and returns a list of proposed structural edges.
    """
    parser = RobustLeanParser()
    classifier = HeuristicEdgeClassifier()
    
    blocks = parser.parse_string(text, "generated.lean")
    proposed_edges = []
    
    # Map name to ID for the classifier
    existing_nodes = {}
    for node_id, data in existing_kg.graph.nodes(data=True):
        existing_nodes[data.get('name', node_id)] = node_id
    
    for block in blocks:
        source_id = f"obj:{block['name']}" if block['type'] == "def" else f"thm:{block['name']}"
        classified_edges = classifier.classify(block, existing_nodes)
        
        for target_id, edge_type in classified_edges:
            proposed_edges.append((source_id, target_id, edge_type))
            
    return proposed_edges

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'mathlib4')
    kg = build_graph_from_mathlib(data_dir)
    print("\nExtraction Complete. Resulting Graph:")
    print(kg.summary())
