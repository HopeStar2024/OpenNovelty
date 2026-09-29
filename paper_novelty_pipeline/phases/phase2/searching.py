"""
Phase 2: Paper Searching — PubMed E-utilities backend.

This module is a drop-in replacement for the original Wispaper-based Phase 2
(which was never publicly released — the whole module used to raise
`NotImplementedError`). The external interface (`PaperSearcher`,
`run_phase2_search`) is unchanged; only the internal implementation differs:

  Original (Wispaper, unavailable)          This implementation (PubMed)
  ----------------------------------------  ----------------------------------------
  OAuth2 + SSE streaming search             ESearch (relevance-sorted PMIDs)
  Server-side LLM verification events       Local keyword-overlap heuristic
                                            producing the same verdict shape
  Raw SSE events saved to raw_responses/    Synthesized "onAgentEnd/verification"
                                            events in the exact same shape,
                                            so postprocess.py stays untouched

Output contract (unchanged, consumed by phases/phase2/postprocess.py):
  Writes `phase2/raw_responses/raw_{scope}_{qhash}.json` per query, where the
  file contains a list of pseudo-SSE events:
      [{"event": "onAgentEnd", "name": "verification",
        "data": {"metadata": {...paper fields...},
                 "content": "<verdict JSON>"}}, ...]
  `metadata` carries title/authors/venue/year/doi/url/pdf_url/abstract/...,
  `content` is a JSON string with `criteria_assessment` used by postprocess to
  compute the `perfect` quality flags.

Phase 1 input (confirmed from models.py / llm_extractors.py):
  - core_task.query_variants (variants[0] == original text); searched directly
  - contributions[i].id == "contribution_i"; queries = prior_work_query +
    query_variants; if both empty, 2-3 [tiab]-style variants are synthesized.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from paper_novelty_pipeline.models import ExtractedContent
from paper_novelty_pipeline.services.pubmed_client import (
    PubMedClient,
    PubMedError,
)

logger = logging.getLogger(__name__)

# English stopwords for keyword-based query synthesis / relevance scoring.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "in", "on", "to", "with",
    "by", "via", "using", "use", "used", "is", "are", "was", "were", "be",
    "been", "this", "that", "these", "those", "we", "our", "its", "it",
    "as", "at", "from", "into", "than", "then", "which", "who", "whom",
    "based", "novel", "new", "study", "approach", "method", "methods",
    "propose", "proposed", "propose", "paper", "work", "propose", "also",
    "such", "can", "could", "may", "might", "will", "would", "has", "have",
    "had", "not", "no", "but", "their", "them", "they", "between", "among",
    "more", "most", "less", "least", "very", "both", "each", "all", "any",
}


def _keywords(text: str, limit: int = 6) -> List[str]:
    """Extract lowercase keywords from free text (stopwords removed)."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())
    seen: List[str] = []
    for w in words:
        if w in _STOPWORDS and w not in seen:
            continue
        if w not in seen:
            seen.append(w)
    return seen[:limit]


def _synthesize_queries(text: str) -> List[str]:
    """Build 2-3 semantic-equivalent PubMed queries when Phase I gave none.

    Strategy: keyword extraction + [tiab] field qualification, with an
    OR-broadened second variant for recall.
    """
    kws = _keywords(text, limit=6)
    if not kws:
        return []
    tight = " AND ".join(f"{k}[tiab]" for k in kws[:4])
    loose = " OR ".join(f"{k}[tiab]" for k in kws[:4])
    queries = [tight]
    if len(kws) >= 2:
        queries.append(f"({loose}) AND {kws[0]}[tiab]")
    return queries


def _relevance_verdict(query: str, title: str, abstract: str) -> Dict[str, Any]:
    """Local replacement for Wispaper's server-side LLM verification.

    Computes the fraction of distinct query keywords found in
    title+abstract, and emits the same `criteria_assessment` verdict shape
    postprocess.py understands:
      ratio >= 0.30 -> "support"     (flags.perfect)
      ratio >= 0.15 -> "somewhat_support" (flags.partial)
      else          -> "no"
    """
    kws = _keywords(query, limit=10)
    if not kws:
        assessment = "support"  # cannot judge; keep the record
    else:
        hay = f"{title} {abstract}".lower()
        hits = sum(1 for k in kws if k in hay)
        ratio = hits / len(kws)
        if ratio >= 0.30:
            assessment = "support"
        elif ratio >= 0.15:
            assessment = "somewhat_support"
        else:
            assessment = "no"
    return {"criteria_assessment": [{"assessment": assessment, "type": "topic"}]}


class PaperSearcher:
    """Phase 2: search for related papers via PubMed E-utilities.

    Replaces the unavailable Wispaper API. Interface is identical to the
    original stub: `search_all(extracted, out_dir) -> stats dict`.
    """

    def __init__(self, concurrency: Optional[int] = None):
        # `concurrency` kept for signature compatibility; PubMed rate limits
        # make serial execution the correct behavior (limiter inside the client).
        if concurrency and concurrency > 1:
            logger.info(
                "PaperSearcher(PubMed): concurrency=%d ignored; PubMed rate "
                "limits enforce serial requests.", concurrency,
            )
        self.client = PubMedClient()

    # ------------------------------------------------------------------ #

    def _build_queries(self, extracted: ExtractedContent) -> Dict[str, List[str]]:
        """scope -> ordered list of PubMed queries for that scope."""
        queries: Dict[str, List[str]] = {}

        # Core task: use Phase I variants verbatim when available
        core_variants = [q for q in (extracted.core_task.query_variants or []) if q.strip()]
        if not core_variants and extracted.core_task.text.strip():
            core_variants = _synthesize_queries(extracted.core_task.text)
        if core_variants:
            queries["core_task"] = core_variants

        # Contributions: prior_work_query + variants, else synthesize
        for i, contrib in enumerate(extracted.contributions or [], start=1):
            scope = getattr(contrib, "id", None) or f"contribution_{i}"
            # postprocess filename regex requires contribution_\d+
            if not re.match(r"^contribution_\d+$", scope):
                scope = f"contribution_{i}"
            qs = [contrib.prior_work_query] + [
                q for q in (contrib.query_variants or []) if q.strip()
            ]
            qs = [q.strip() for q in qs if q and q.strip()]
            if not qs:
                base = f"{contrib.name}. {contrib.description or contrib.author_claim_text}"
                qs = _synthesize_queries(base)
            if qs:
                queries[scope] = qs

        return queries

    # ------------------------------------------------------------------ #

    def _search_scope(
        self,
        scope: str,
        queries: List[str],
        raw_responses_dir: Path,
    ) -> Dict[str, Any]:
        """Run all queries for one scope, write one raw_*.json per query."""
        total_pmids: List[str] = []
        seen: set = set()
        n_failed = 0

        for query in queries:
            qhash = hashlib.md5(query.encode("utf-8")).hexdigest()[:10]
            fname = f"raw_{scope}_{qhash}.json"
            try:
                pmids = self.client.esearch(query)
            except PubMedError as e:
                # Single-query failure: log, write empty file, keep going.
                logger.error("Scope %s query failed (%r): %s", scope, query[:60], e)
                n_failed += 1
                events: List[Dict[str, Any]] = []
                self._write_events(raw_responses_dir / fname, query, events)
                continue

            if not pmids:
                logger.info("Scope %s query returned 0 PMIDs: %r", scope, query[:60])
                self._write_events(raw_responses_dir / fname, query, [])
                continue

            try:
                articles = self.client.efetch_articles(pmids)
            except PubMedError as e:
                logger.error("Scope %s efetch failed for %r: %s", scope, query[:60], e)
                n_failed += 1
                self._write_events(raw_responses_dir / fname, query, [])
                continue

            # Relevance score from ESearch rank position (descending).
            events = []
            for rank, art in enumerate(articles):
                art = dict(art)
                art["relevance_score"] = round(1.0 - rank * (0.9 / max(len(articles), 1)), 4)
                verdict = _relevance_verdict(query, art.get("title", ""), art.get("abstract", ""))
                events.append(
                    {
                        "event": "onAgentEnd",
                        "name": "verification",
                        "data": {
                            "metadata": art,
                            "content": json.dumps(verdict, ensure_ascii=False),
                        },
                    }
                )
                pmid = (art.get("paper_id") or "").replace("PMID:", "")
                if pmid and pmid not in seen:
                    seen.add(pmid)
                    total_pmids.append(pmid)

            self._write_events(raw_responses_dir / fname, query, events)
            logger.info("Scope %s: %d articles for query %r", scope, len(articles), query[:60])

        return {
            "scope": scope,
            "queries": len(queries),
            "failed_queries": n_failed,
            "unique_articles": len(total_pmids),
        }

    @staticmethod
    def _write_events(path: Path, query: str, events: List[Dict[str, Any]]) -> None:
        """Write one raw_responses file in the (Wispaper-compatible) event format."""
        path.parent.mkdir(parents=True, exist_ok=True)
        # postprocess.py expects the file to BE the event list (json.load -> list).
        with path.open("w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ #

    def search_all(
        self,
        extracted: ExtractedContent,
        out_dir: Path,
    ) -> Dict[str, Any]:
        """Execute all searches for a paper (interface unchanged vs. original)."""
        raw_dir = Path(out_dir) / "raw_responses"
        raw_dir.mkdir(parents=True, exist_ok=True)

        scope_queries = self._build_queries(extracted)
        if not scope_queries:
            logger.warning("Phase2: no searchable queries could be built from Phase1 output.")
            return {"total_queries": 0, "failed": 0, "scopes": []}

        # Connectivity guard: fail loudly and early if PubMed is unreachable,
        # so the user knows to check network / API key before burning time.
        if not self.client.health_check():
            raise PubMedError(
                "Phase2 search failed: PubMed E-utilities is unreachable. "
                "Check your network connection and NCBI_API_KEY."
            )

        total = failed = 0
        scope_stats = []
        for scope, queries in scope_queries.items():
            stats = self._search_scope(scope, queries, raw_dir)
            total += stats["queries"]
            failed += stats["failed_queries"]
            scope_stats.append(stats)

        return {
            "backend": "pubmed",
            "total_queries": total,
            "failed": failed,
            "scopes": scope_stats,
        }


def run_phase2_search(
    extracted: ExtractedContent,
    out_dir: Path,
    concurrency: Optional[int] = None,
) -> Dict[str, Any]:
    """Run Phase2 search (API calls only). Interface unchanged vs. original."""
    searcher = PaperSearcher(concurrency=concurrency)
    return searcher.search_all(extracted, out_dir)
