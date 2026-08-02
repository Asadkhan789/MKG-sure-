from tvqa_mkg_rag.mkg_sure.config import PathConfig, SelectionConfig
from tvqa_mkg_rag.mkg_sure.retrieval import generate_candidate_paths, select_paths
from tvqa_mkg_rag.mkg_sure.types import GraphEdge, GraphNode


def test_generates_one_and_two_hop_paths() -> None:
    nodes = {
        "a": GraphNode("a", "v1", "entity", "Alice"),
        "b": GraphNode("b", "v1", "event", "opens door"),
        "c": GraphNode("c", "v1", "entity", "room"),
    }
    first = GraphEdge("e1", "v1", "a", "performs", "b", "vis", 0.9, ["s1"])
    second = GraphEdge("e2", "v1", "b", "occurs_in", "c", "vis", 0.8, ["s2"])
    paths = generate_candidate_paths(
        "v1",
        nodes,
        [first, second],
        [(first, 1.0), (second, 0.8)],
        PathConfig(seed_edges=1, beam_width=5, max_path_length=2, max_candidates=16),
    )
    assert {path.hop_count for path in paths} == {1, 2}


def test_selector_respects_budget_and_prefers_gain() -> None:
    nodes = {
        "a": GraphNode("a", "v1", "entity", "Alice"),
        "b": GraphNode("b", "v1", "event", "opens door"),
        "c": GraphNode("c", "v1", "entity", "room"),
    }
    first = GraphEdge("e1", "v1", "a", "performs", "b", "vis", 0.9, ["s1"])
    second = GraphEdge("e2", "v1", "b", "occurs_in", "c", "vis", 0.8, ["s2"])
    paths = generate_candidate_paths(
        "v1",
        nodes,
        [first, second],
        [(first, 1.0), (second, 0.8)],
        PathConfig(seed_edges=1, beam_width=5, max_path_length=2, max_candidates=16),
    )
    for path in paths:
        path.utility = 0.8 if path.hop_count == 2 else 0.2
        path.grounding = 0.8
        path.matched_anchors = ["Alice"]
        path.text_token_ids = [f"t:{path.path_id}:0", f"t:{path.path_id}:1"]
    selected, trace = select_paths(
        paths,
        ["Alice", "room"],
        SelectionConfig(evidence_budget=2, grounding_weight=0.35, coverage_weight=0.3, redundancy_weight=0.25),
    )
    assert len(selected) == 1
    assert selected[0].hop_count == 2
    assert trace[0].marginal_cost == 2
