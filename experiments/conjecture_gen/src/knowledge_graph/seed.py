"""Seed knowledge graph for extremal graph theory.

Covers the main theorem landscape: Turán-type results, Ramsey theory,
chromatic graph theory, matching/connectivity, and key techniques.
Edges are typed (see EdgeType) so the reward model can reason about structure.
"""

from .graph import KnowledgeGraph, NodeType, EdgeType


def generate_seed_graph() -> KnowledgeGraph:
    kg = KnowledgeGraph()

    # ─── Objects ──────────────────────────────────────────────────────────────
    _objs = [
        ("obj:graph",            "Graph",                    "A set of vertices and edges."),
        ("obj:simple_graph",     "Simple Graph",             "Graph with no loops or multi-edges."),
        ("obj:complete_graph",   "Complete Graph K_n",       "Every pair of distinct vertices is connected."),
        ("obj:bipartite_graph",  "Bipartite Graph",          "Vertices partitioned into two sets; edges only cross between sets."),
        ("obj:complete_bipartite","Complete Bipartite K_{s,t}","All edges between two parts of size s and t."),
        ("obj:cycle",            "Cycle C_n",                "A closed path on n vertices."),
        ("obj:even_cycle",       "Even Cycle C_{2k}",        "A cycle of even length 2k."),
        ("obj:path",             "Path P_k",                 "A simple path on k vertices."),
        ("obj:tree",             "Tree",                     "A connected acyclic graph."),
        ("obj:triangle",         "Triangle K_3",             "Complete graph on 3 vertices."),
        ("obj:turan_graph",      "Turán Graph T(n,r)",       "Complete r-partite graph with parts as equal as possible."),
        ("obj:regular_graph",    "d-Regular Graph",          "Every vertex has degree exactly d."),
        ("obj:planar_graph",     "Planar Graph",             "Embeddable in the plane without edge crossings."),
        ("obj:hamiltonian_graph","Hamiltonian Graph",        "Contains a cycle visiting every vertex exactly once."),
        ("obj:eulerian_graph",   "Eulerian Graph",           "Contains a closed walk traversing every edge exactly once."),
        ("obj:expander_graph",   "Expander Graph",           "Sparse graph with strong connectivity / large spectral gap."),
        ("obj:mycielski_graph",  "Mycielski Graph",          "Triangle-free graph with arbitrarily large chromatic number."),
        ("obj:petersen_graph",   "Petersen Graph",           "Cubic graph on 10 vertices; archetypal counterexample/extremal object."),
        ("obj:windmill_graph",   "Windmill Graph",           "K_1 joined to t disjoint triangles; characterizes friendship graphs."),
    ]
    for nid, name, desc in _objs:
        kg.add_node(nid, NodeType.OBJECT, {"name": name, "description": desc})

    # ─── Quantities ───────────────────────────────────────────────────────────
    _quants = [
        ("quant:edge_count",        "Number of Edges e(G)",       "Total edges in G."),
        ("quant:vertex_count",      "Number of Vertices n",       "Total vertices in G."),
        ("quant:clique_number",     "Clique Number ω(G)",         "Size of largest clique."),
        ("quant:chromatic_number",  "Chromatic Number χ(G)",      "Minimum colors to properly color vertices."),
        ("quant:edge_chromatic",    "Edge Chromatic Number χ'(G)","Minimum colors to properly color edges (= chromatic index)."),
        ("quant:independence_number","Independence Number α(G)",  "Size of largest independent set."),
        ("quant:max_degree",        "Maximum Degree Δ(G)",        "Largest vertex degree."),
        ("quant:min_degree",        "Minimum Degree δ(G)",        "Smallest vertex degree."),
        ("quant:girth",             "Girth g(G)",                 "Length of shortest cycle."),
        ("quant:diameter",          "Diameter diam(G)",           "Maximum shortest-path distance."),
        ("quant:connectivity",      "Vertex Connectivity κ(G)",   "Min vertices to remove to disconnect G."),
        ("quant:treewidth",         "Treewidth tw(G)",            "Width of optimal tree decomposition."),
        ("quant:extremal_number",   "Extremal Number ex(n,H)",    "Max edges in n-vertex H-free graph."),
        ("quant:ramsey_number",     "Ramsey Number R(r,s)",       "Min n: any 2-coloring of K_n has red K_r or blue K_s."),
        ("quant:zarankiewicz",      "Zarankiewicz Number z(n;s,t)","Max edges in n×n 0/1 matrix with no s×t all-1 submatrix."),
    ]
    for nid, name, desc in _quants:
        kg.add_node(nid, NodeType.QUANTITY, {"name": name, "description": desc})

    # ─── Techniques ───────────────────────────────────────────────────────────
    _techs = [
        ("tech:probabilistic",    "Probabilistic Method",       "Existence proofs via random constructions (Erdős)."),
        ("tech:double_counting",  "Double Counting",            "Count the same quantity two ways to derive equalities/inequalities."),
        ("tech:linear_algebra",   "Linear Algebra Method",      "Use rank, eigenvalues, or polynomial arguments on adjacency/Laplacian."),
        ("tech:regularity_lemma", "Szemerédi Regularity Lemma", "Decompose dense graphs into pseudorandom bipartite pieces."),
        ("tech:flag_algebras",    "Flag Algebra Method",        "Razborov's SDP-based framework for extremal density problems."),
        ("tech:discharging",      "Discharging Method",         "Local redistribution argument; key in four-color theorem proof."),
        ("tech:tensor_product",   "Tensor Product / Zig-Zag",   "Graph product constructions for expanders and extremal graphs."),
        ("tech:induction",        "Induction",                  "Structural or strong induction on graph parameters."),
        ("tech:lovász_theta",     "Lovász Theta Function",      "SDP bound on independence/clique numbers; bridges graph theory and semidefinite programming."),
    ]
    for nid, name, desc in _techs:
        kg.add_node(nid, NodeType.TECHNIQUE, {"name": name, "description": desc})

    # ─── Theorems ─────────────────────────────────────────────────────────────
    _thms = [
        ("thm:mantel",
         "Mantel's Theorem",
         "Triangle-free graph on n vertices has at most ⌊n²/4⌋ edges."),
        ("thm:turan",
         "Turán's Theorem",
         "K_{r+1}-free graph on n vertices has at most e(T(n,r)) edges, achieved uniquely by T(n,r)."),
        ("thm:erdos_stone",
         "Erdős–Stone Theorem",
         "ex(n,H) = (1 - 1/(χ(H)-1)) · n²/2 + o(n²) for any graph H with χ(H) ≥ 2."),
        ("thm:kovari_sos_turan",
         "Kővári–Sós–Turán Theorem",
         "K_{s,t}-free graph on n vertices has at most ½(t-1)^{1/s} n^{2-1/s} + (s-1)n/2 edges."),
        ("thm:bondy_simonovits",
         "Bondy–Simonovits Theorem",
         "ex(n, C_{2k}) = Θ(n^{1+1/k}); even-cycle extremal numbers have a tight algebraic form."),
        ("thm:ramsey_existence",
         "Ramsey's Theorem",
         "For all r,s ≥ 1, R(r,s) is finite; every sufficiently large complete graph contains a monochromatic K_r or K_s."),
        ("thm:ramsey_r33",
         "R(3,3) = 6",
         "The smallest n for which every 2-coloring of K_n contains a monochromatic triangle is 6."),
        ("thm:brooks",
         "Brooks' Theorem",
         "χ(G) ≤ Δ(G) for any connected graph G that is not K_n or an odd cycle."),
        ("thm:vizing",
         "Vizing's Theorem",
         "Δ(G) ≤ χ'(G) ≤ Δ(G)+1 for any simple graph G."),
        ("thm:konig_edge",
         "König's Theorem (Edge Coloring)",
         "χ'(G) = Δ(G) for every bipartite graph G."),
        ("thm:konig_matching",
         "König's Theorem (Matching)",
         "In bipartite graphs, maximum matching size = minimum vertex cover size."),
        ("thm:hall",
         "Hall's Marriage Theorem",
         "A bipartite graph has a perfect matching iff for every subset S of one part, |N(S)| ≥ |S|."),
        ("thm:menger",
         "Menger's Theorem",
         "Max number of vertex-disjoint s-t paths equals min size of s-t vertex cut."),
        ("thm:max_flow_min_cut",
         "Max-Flow Min-Cut Theorem",
         "In a flow network, maximum flow value equals minimum cut capacity."),
        ("thm:dirac",
         "Dirac's Theorem",
         "If δ(G) ≥ n/2 for n ≥ 3, then G is Hamiltonian."),
        ("thm:ore",
         "Ore's Theorem",
         "If deg(u)+deg(v) ≥ n for all non-adjacent u,v, then G is Hamiltonian."),
        ("thm:four_color",
         "Four Color Theorem",
         "Every planar graph is 4-colorable."),
        ("thm:friendship",
         "Friendship Theorem",
         "If every pair of vertices has exactly one common neighbor, then G is a windmill graph (K_1 joined to disjoint triangles)."),
        ("thm:erdos_gallai",
         "Erdős–Gallai Theorem (Paths)",
         "ex(n, P_{k+1}) = (k-1)n/2; the extremal graph is a disjoint union of complete graphs K_k."),
        ("thm:kruskal_katona",
         "Kruskal–Katona Theorem",
         "Bounds the minimum shadow size of a family of k-sets of given size; fundamental in extremal set theory."),
        ("thm:erdos_ko_rado",
         "Erdős–Ko–Rado Theorem",
         "A maximal intersecting family of k-subsets of [n] (n ≥ 2k) has at most C(n-1, k-1) members."),
        ("thm:expander_mixing",
         "Expander Mixing Lemma",
         "For a d-regular graph with second eigenvalue λ, edges between sets S,T satisfy |e(S,T) - d|S||T|/n| ≤ λ√(|S||T|)."),
        ("thm:lovász_theta_bound",
         "Lovász Sandwich Theorem",
         "ω(G) ≤ ϑ(Ḡ) ≤ χ(G) and ω(Ḡ) ≤ ϑ(G) ≤ χ(Ḡ), where ϑ is the Lovász theta function."),
        ("thm:szemeredi_regularity",
         "Szemerédi Regularity Lemma",
         "Every dense graph admits an ε-regular partition into a bounded number of nearly equal parts."),
        ("thm:szemeredi_ap",
         "Szemerédi's Theorem (Arithmetic Progressions)",
         "Every subset of integers with positive upper density contains arbitrarily long arithmetic progressions."),
        ("thm:ramseys_multiplicity",
         "Goodman's Formula",
         "The number of monochromatic triangles in a 2-coloring of K_n is at least n(n-1)(n-5)/24."),
    ]
    for nid, name, stmt in _thms:
        kg.add_node(nid, NodeType.THEOREM, {"name": name, "statement": stmt})

    # ─── Open Conjectures ─────────────────────────────────────────────────────
    _conjs = [
        ("conj:erdos_simonovits",
         "Erdős–Simonovits Conjecture (Exact Turán)",
         "For every graph H with χ(H)=r+1 ≥ 3, ex(n,H) = e(T(n,r)) for all large n (exact, not just asymptotic)."),
        ("conj:hadwiger",
         "Hadwiger's Conjecture",
         "Every K_{t+1}-minor-free graph has χ(G) ≤ t."),
        ("conj:erdos_hajnal",
         "Erdős–Hajnal Conjecture",
         "For every graph H, graphs with no induced copy of H have a clique or independent set of size n^{ε(H)}."),
        ("conj:zarankiewicz_tight",
         "Zarankiewicz Problem (K_{2,2})",
         "ex(n, K_{2,2}) = Θ(n^{3/2}); the exact constant and extremal graphs are not fully characterized."),
        ("conj:ramsey_lower",
         "Erdős Ramsey Lower Bound Conjecture",
         "R(k,k) ≥ 2^{k/2+o(k)}; the probabilistic lower bound is tight up to the exponent constant."),
        ("conj:bipartite_turan",
         "Bipartite Turán Conjecture",
         "For bipartite H, ex(n,H) = O(n^{2-1/s}) where s is the smaller part of a balanced bipartite subgraph of H."),
        ("conj:linear_arboricity",
         "Linear Arboricity Conjecture",
         "Every d-regular graph can be decomposed into ⌈(d+1)/2⌉ linear forests."),
    ]
    for nid, name, stmt in _conjs:
        kg.add_node(nid, NodeType.CONJECTURE, {"name": name, "statement": stmt})

    # ─── Edges ────────────────────────────────────────────────────────────────

    # Object hierarchy
    kg.add_edge("obj:triangle",        "obj:complete_graph",   EdgeType.SPECIAL_CASE,  {"note": "K_3"})
    kg.add_edge("obj:complete_bipartite","obj:bipartite_graph", EdgeType.SPECIAL_CASE)
    kg.add_edge("obj:bipartite_graph", "obj:graph",            EdgeType.SPECIAL_CASE)
    kg.add_edge("obj:tree",            "obj:bipartite_graph",  EdgeType.SPECIAL_CASE,  {"note": "Trees are bipartite"})
    kg.add_edge("obj:cycle",           "obj:graph",            EdgeType.SPECIAL_CASE)
    kg.add_edge("obj:even_cycle",      "obj:cycle",            EdgeType.SPECIAL_CASE)
    kg.add_edge("obj:turan_graph",     "obj:complete_graph",   EdgeType.SPECIAL_CASE,  {"note": "T(n,1) = K_n"})
    kg.add_edge("obj:turan_graph",     "obj:complete_bipartite",EdgeType.SPECIAL_CASE, {"note": "T(n,2) = K_{n/2,n/2}"})
    kg.add_edge("obj:petersen_graph",  "obj:regular_graph",    EdgeType.SPECIAL_CASE,  {"note": "3-regular"})

    # Turán-type theorem relationships
    kg.add_edge("thm:turan",        "thm:mantel",           EdgeType.GENERALIZES,   {"note": "r=2 gives Mantel"})
    kg.add_edge("thm:erdos_stone",  "thm:turan",            EdgeType.GENERALIZES,   {"note": "asymptotic; χ(H)=r+1 → Turán density"})
    kg.add_edge("conj:erdos_simonovits","thm:erdos_stone",  EdgeType.GENERALIZES,   {"note": "exact form of the asymptotic"})
    kg.add_edge("thm:kovari_sos_turan","thm:mantel",        EdgeType.GENERALIZES,   {"note": "s=t=2 gives bipartite Mantel"})
    kg.add_edge("thm:bondy_simonovits","thm:kovari_sos_turan",EdgeType.GENERALIZES, {"note": "even cycles C_{2k}; KST gives C_4 case"})
    kg.add_edge("conj:bipartite_turan","thm:kovari_sos_turan",EdgeType.GENERALIZES)

    # Bounds on edge count
    kg.add_edge("thm:turan",            "quant:edge_count",      EdgeType.BOUNDS, {"type": "upper", "condition": "K_{r+1}-free"})
    kg.add_edge("thm:mantel",           "quant:edge_count",      EdgeType.BOUNDS, {"type": "upper", "condition": "triangle-free"})
    kg.add_edge("thm:kovari_sos_turan", "quant:edge_count",      EdgeType.BOUNDS, {"type": "upper", "condition": "K_{s,t}-free"})
    kg.add_edge("thm:bondy_simonovits", "quant:extremal_number", EdgeType.BOUNDS, {"type": "exact order", "condition": "C_{2k}-free"})
    kg.add_edge("thm:erdos_gallai",     "quant:extremal_number", EdgeType.BOUNDS, {"type": "exact", "condition": "P_{k+1}-free"})
    kg.add_edge("thm:expander_mixing",  "quant:edge_count",      EdgeType.BOUNDS, {"type": "discrepancy"})

    # Conditions
    kg.add_edge("thm:mantel",           "obj:triangle",          EdgeType.HAS_CONDITION, {"role": "excluded subgraph"})
    kg.add_edge("thm:turan",            "obj:complete_graph",    EdgeType.HAS_CONDITION, {"role": "excluded subgraph"})
    kg.add_edge("thm:kovari_sos_turan", "obj:complete_bipartite",EdgeType.HAS_CONDITION, {"role": "excluded subgraph"})
    kg.add_edge("thm:bondy_simonovits", "obj:even_cycle",        EdgeType.HAS_CONDITION, {"role": "excluded subgraph"})
    kg.add_edge("thm:brooks",           "quant:max_degree",      EdgeType.HAS_CONDITION)
    kg.add_edge("thm:dirac",            "quant:min_degree",      EdgeType.HAS_CONDITION, {"threshold": "n/2"})
    kg.add_edge("thm:ore",              "quant:min_degree",      EdgeType.HAS_CONDITION, {"condition": "degree-sum"})
    kg.add_edge("thm:hall",             "obj:bipartite_graph",   EdgeType.HAS_CONDITION)

    # Chromatic number bounds
    kg.add_edge("thm:brooks",           "quant:chromatic_number",EdgeType.BOUNDS, {"type": "upper"})
    kg.add_edge("thm:four_color",       "quant:chromatic_number",EdgeType.BOUNDS, {"type": "upper", "condition": "planar"})
    kg.add_edge("thm:lovász_theta_bound","quant:chromatic_number",EdgeType.BOUNDS, {"type": "lower", "via": "theta function"})
    kg.add_edge("conj:hadwiger",        "quant:chromatic_number",EdgeType.BOUNDS, {"type": "upper", "via": "minor"})
    kg.add_edge("thm:erdos_stone",      "quant:chromatic_number",EdgeType.HAS_CONDITION, {"note": "density determined by χ(H)"})

    # Edge coloring
    kg.add_edge("thm:vizing",           "quant:edge_chromatic",  EdgeType.BOUNDS, {"type": "upper+lower"})
    kg.add_edge("thm:konig_edge",       "quant:edge_chromatic",  EdgeType.BOUNDS, {"type": "exact", "condition": "bipartite"})
    kg.add_edge("thm:konig_edge",       "thm:vizing",            EdgeType.SPECIAL_CASE, {"note": "tight for bipartite"})

    # Matching and connectivity
    kg.add_edge("thm:hall",             "thm:konig_matching",    EdgeType.IMPLIES, {"note": "Hall → König bipartite matching"})
    kg.add_edge("thm:menger",           "thm:max_flow_min_cut",  EdgeType.IMPLIES, {"note": "Menger is the graph-theoretic analogue"})
    kg.add_edge("thm:konig_matching",   "quant:independence_number",EdgeType.BOUNDS, {"note": "bipartite: α(G)+ν(G)=n"})

    # Hamiltonicity
    kg.add_edge("thm:dirac",            "obj:hamiltonian_graph", EdgeType.IMPLIES)
    kg.add_edge("thm:ore",              "obj:hamiltonian_graph", EdgeType.IMPLIES)
    kg.add_edge("thm:ore",              "thm:dirac",             EdgeType.GENERALIZES)

    # Ramsey theory
    kg.add_edge("thm:ramsey_existence", "quant:ramsey_number",   EdgeType.BOUNDS, {"type": "finiteness"})
    kg.add_edge("thm:ramsey_r33",       "quant:ramsey_number",   EdgeType.BOUNDS, {"type": "exact", "R": "R(3,3)=6"})
    kg.add_edge("thm:ramsey_r33",       "thm:ramsey_existence",  EdgeType.SPECIAL_CASE, {"note": "r=s=3"})
    kg.add_edge("conj:ramsey_lower",    "quant:ramsey_number",   EdgeType.BOUNDS, {"type": "lower"})
    kg.add_edge("thm:ramseys_multiplicity","thm:ramsey_r33",     EdgeType.IMPLIES, {"note": "Goodman formula gives count"})

    # Regularity and density
    kg.add_edge("thm:szemeredi_regularity","thm:erdos_stone",    EdgeType.TECHNIQUE_USED)
    kg.add_edge("thm:szemeredi_regularity","thm:bondy_simonovits",EdgeType.TECHNIQUE_USED)
    kg.add_edge("thm:szemeredi_ap",     "thm:szemeredi_regularity",EdgeType.TECHNIQUE_USED)

    # Extremal set theory
    kg.add_edge("thm:kruskal_katona",   "thm:erdos_ko_rado",     EdgeType.TECHNIQUE_USED)

    # Special graph properties
    kg.add_edge("thm:friendship",       "obj:windmill_graph",    EdgeType.IMPLIES, {"note": "characterizes friendship graphs"})
    kg.add_edge("obj:mycielski_graph",  "quant:chromatic_number",EdgeType.BOUNDS,  {"note": "χ can be large while triangle-free"})
    kg.add_edge("obj:mycielski_graph",  "obj:triangle",          EdgeType.COUNTEREXAMPLE, {"note": "no triangle but arbitrary χ"})

    # Lovász theta
    kg.add_edge("thm:lovász_theta_bound","quant:clique_number",  EdgeType.BOUNDS, {"type": "sandwiched"})
    kg.add_edge("thm:lovász_theta_bound","quant:independence_number",EdgeType.BOUNDS)
    kg.add_edge("tech:lovász_theta",    "thm:lovász_theta_bound",EdgeType.TECHNIQUE_USED)

    # Techniques → Theorems
    kg.add_edge("tech:probabilistic",   "thm:ramsey_existence",  EdgeType.TECHNIQUE_USED)
    kg.add_edge("tech:probabilistic",   "conj:ramsey_lower",     EdgeType.TECHNIQUE_USED)
    kg.add_edge("tech:double_counting", "thm:mantel",            EdgeType.TECHNIQUE_USED)
    kg.add_edge("tech:double_counting", "thm:kovari_sos_turan",  EdgeType.TECHNIQUE_USED)
    kg.add_edge("tech:linear_algebra",  "thm:expander_mixing",   EdgeType.TECHNIQUE_USED)
    kg.add_edge("tech:linear_algebra",  "thm:friendship",        EdgeType.TECHNIQUE_USED)
    kg.add_edge("tech:regularity_lemma","thm:szemeredi_regularity",EdgeType.TECHNIQUE_USED)
    kg.add_edge("tech:flag_algebras",   "thm:ramseys_multiplicity",EdgeType.TECHNIQUE_USED)
    kg.add_edge("tech:discharging",     "thm:four_color",        EdgeType.TECHNIQUE_USED)
    kg.add_edge("tech:induction",       "thm:ramsey_existence",  EdgeType.TECHNIQUE_USED)

    # Hadwiger and Erdős–Hajnal open problems
    kg.add_edge("conj:hadwiger",        "thm:four_color",        EdgeType.IMPLIES, {"note": "Hadwiger for t=4 implies 4CT"})
    kg.add_edge("conj:erdos_hajnal",    "quant:clique_number",   EdgeType.BOUNDS,  {"type": "lower", "condition": "H-free induced"})
    kg.add_edge("conj:erdos_hajnal",    "quant:independence_number",EdgeType.BOUNDS, {"type": "lower"})

    return kg


if __name__ == "__main__":
    kg = generate_seed_graph()
    print(kg.summary())
