#!/usr/bin/env python3
"""
End-to-end test for the PubMed-backed Phase 2.

Runs PaperSearcher (ESearch + EFetch) on a synthetic Phase1 extraction built
around a well-known paper ("Attention Is All You Need" as a familiarity
anchor; the queries themselves search for prior work on that topic), then
runs the REAL Phase2 postprocess to produce final/citation_index.json +
TopK files — proving Phase III/IV compatibility without modification.

Usage:
    cd OpenNovelty
    python scripts/test_pubmed_phase2.py [--keep]

Requires NCBI_API_KEY (set in .env or environment).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paper_novelty_pipeline.models import (  # noqa: E402
    ContributionClaim,
    CoreTask,
    ExtractedContent,
)
from paper_novelty_pipeline.phases.phase2.searching import PaperSearcher  # noqa: E402
from paper_novelty_pipeline.phases.phase2.postprocess import (  # noqa: E402
    Phase2Processor,
)


def build_fake_extraction() -> ExtractedContent:
    """Synthetic Phase1 output about transformer-based machine translation."""
    return ExtractedContent(
        core_task=CoreTask(
            text="neural machine translation with transformer architectures",
            query_variants=[
                "neural machine translation transformer",
                "transformer sequence transduction attention",
            ],
        ),
        contributions=[
            ContributionClaim(
                id="contribution_1",
                name="Multi-head self-attention for translation",
                author_claim_text="We rely exclusively on attention, removing recurrence and convolutions.",
                description="Self-attention mechanism replacing recurrent layers in sequence transduction.",
                prior_work_query="self-attention neural machine translation",
            ),
            ContributionClaim(
                id="contribution_2",
                name="Scaled dot-product attention",
                author_claim_text="We scale dot products by sqrt(d_k) for stable gradients.",
                description="Scaling trick for dot-product attention.",
                # no prior_work_query/variants -> exercises query synthesis path
            ),
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keep", action="store_true", help="keep the test output directory"
    )
    parser.add_argument(
        "--out", default="test_output_pubmed", help="output directory (default test_output_pubmed)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("test_pubmed_phase2")

    out_dir = PROJECT_ROOT / args.out
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # Need a fake phase1/paper.json for postprocess self-filter + index 0
    phase1_dir = out_dir / "phase1"
    phase1_dir.mkdir()
    (phase1_dir / "paper.json").write_text(
        json.dumps(
            {
                "paper_id": "test-paper",
                "title": "Attention Is All You Need",
                "authors": ["Vaswani Ashish"],
                "venue": "NeurIPS",
                "year": 2017,
                "abstract": "We propose the Transformer.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    log.info("=== Phase 2 search (PubMed ESearch + EFetch) ===")
    searcher = PaperSearcher(concurrency=1)
    stats = searcher.search_all(build_fake_extraction(), out_dir / "phase2")
    log.info("search stats: %s", json.dumps(stats, ensure_ascii=False))

    log.info("=== Phase 2 postprocess (real postprocess.py, unmodified) ===")
    processor = Phase2Processor(phase2_dir=out_dir / "phase2")
    pp_stats = processor.process()
    log.info("postprocess stats: %s", json.dumps(pp_stats, ensure_ascii=False, indent=2))

    # ---- validation ----
    final_dir = out_dir / "phase2" / "final"
    ci_path = final_dir / "citation_index.json"
    assert ci_path.exists(), "citation_index.json was not produced"
    ci = json.loads(ci_path.read_text(encoding="utf-8"))
    assert ci.get("count") == len(ci.get("items", []))
    assert ci["items"][0]["roles"][0]["type"] == "original_paper"
    n_candidates = ci["count"] - 1
    assert n_candidates > 0, "no candidate papers made it through the filter"

    # every candidate must carry full PubMed metadata used by Phase III/IV
    for item in ci["items"][1:]:
        assert item["title"], f"missing title: {item}"
        assert item["abstract"] is not None, f"missing abstract: {item['title']}"
        assert item["venue"], f"missing venue: {item['title']}"
        assert item["year"], f"missing year: {item['title']}"
        assert item["paper_id"].startswith("PMID:"), f"bad paper_id: {item['paper_id']}"
        assert item["relevance_score"] is not None

    topk_core = final_dir / f"core_task_perfect_top{processor.topk_core_task}.json"
    topk_c1 = final_dir / "contribution_1_perfect_top10.json"
    assert topk_core.exists(), "core_task TopK file missing"
    assert topk_c1.exists(), "contribution_1 TopK file missing"
    for meta in ("contribution_mapping.json", "index.json", "contributions_index_top10.json", "stats.json"):
        assert (final_dir / meta).exists(), f"{meta} missing"

    log.info("PASS: %d candidates in citation_index; TopK + metadata files all present.", n_candidates)
    sample = ci["items"][1]
    log.info("Sample candidate: [%s] %s (%s, %s)",
             sample["paper_id"], sample["title"][:70], sample["venue"][:40], sample["year"])

    if not args.keep:
        shutil.rmtree(out_dir)
        log.info("cleaned up %s (use --keep to inspect outputs)", out_dir)
    else:
        log.info("outputs kept at %s", out_dir)

    print("\n== test_pubmed_phase2 PASS ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
