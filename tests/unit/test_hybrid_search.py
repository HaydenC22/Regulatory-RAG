from services.retrieval.hybrid import RRF_K, reciprocal_rank_fusion
from services.schemas import ChunkMetadata, RetrievedChunk


def make_chunk(chunk_id: str, score: float) -> RetrievedChunk:
    metadata = ChunkMetadata(
        chunk_id=chunk_id,
        doc_id="DOC",
        doc_title="Doc",
        doc_type="notice",
        page_number=1,
        source_url="https://example.com",
        chunk_index=0,
        token_count=10,
    )
    return RetrievedChunk(metadata=metadata, text="text", score=score)


def test_rrf_favors_chunk_ranked_highly_in_both_lists():
    dense = [make_chunk("A", 0.9), make_chunk("B", 0.8), make_chunk("C", 0.7)]
    bm25 = [make_chunk("B", 5.0), make_chunk("A", 4.0), make_chunk("C", 1.0)]

    fused = reciprocal_rank_fusion([dense, bm25])
    fused_ids = [c.metadata.chunk_id for c in fused]

    # A is rank1+rank2, B is rank2+rank1 -> tied for top two, both ahead of C (rank3+rank3)
    assert set(fused_ids[:2]) == {"A", "B"}
    assert fused_ids[2] == "C"


def test_rrf_score_matches_formula():
    dense = [make_chunk("A", 0.9)]
    bm25: list[RetrievedChunk] = []

    fused = reciprocal_rank_fusion([dense, bm25])
    assert len(fused) == 1
    expected = 1.0 / (RRF_K + 1)
    assert abs(fused[0].score - expected) < 1e-9


def test_rrf_chunk_only_in_one_list_still_included():
    dense = [make_chunk("A", 0.9), make_chunk("B", 0.8)]
    bm25 = [make_chunk("C", 5.0)]

    fused = reciprocal_rank_fusion([dense, bm25])
    fused_ids = {c.metadata.chunk_id for c in fused}
    assert fused_ids == {"A", "B", "C"}


def test_rrf_empty_lists_produce_empty_result():
    assert reciprocal_rank_fusion([[], []]) == []
