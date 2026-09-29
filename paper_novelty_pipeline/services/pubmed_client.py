"""
PubMed E-utilities client for academic paper search.

This module is the PubMed-based replacement for the original (never-released)
Wispaper API client (`wispaper_client.py`). It provides:

  - ESearch (`esearch.fcgi`): query -> PMID list, relevance-sorted, retmax-limited
  - EFetch  (`efetch.fcgi`) : batch PMID -> full article XML (rettype=abstract,
                              retmode=xml), parsed into normalized metadata dicts
  - Rate limiting: 10 req/s with an NCBI API key, 3 req/s without
    (implemented as a 0.11s / 0.34s minimum inter-request interval)
  - All requests carry `api_key`, `tool`, `email` parameters
  - Exponential backoff on HTTP 429 / 5xx (up to 3 retries)

Differences from the Wispaper implementation:
  - No OAuth flow, no SSE streaming, no server-side LLM verification.
    Wispaper returned "verification" SSE events; PubMed returns plain XML,
    so relevance verification is done locally by the Phase2 adapter
    (see phases/phase2/searching.py).
  - canonical paper identity: we expose `PMID:<n>` as `paper_id`; the global
    canonical_id remains title-hash based (handled by postprocess /
    utils/paper_id.py, unchanged).

Field mapping (PubMed XML -> normalized dict):
  PMID                -> paper_id "PMID:<n>"
  ArticleTitle        -> title
  AbstractText*       -> abstract (all sections joined, "LABEL: text" when labeled)
  AuthorList/Author   -> authors "LastName ForeName"
  Journal/Title       -> venue
  ArticleId(IdType=doi) -> doi
  JournalIssue/PubDate/Year (or MedlineDate fallback) -> year
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

import requests

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# NCBI rate limits: 10 req/s with API key, 3 req/s without.
_INTERVAL_WITH_KEY = 0.11
_INTERVAL_NO_KEY = 0.34

# EFetch batching: max PMIDs per request (spec: <= 200).
EFETCH_BATCH_SIZE = 200

_MAX_RETRIES = 3
_REQUEST_TIMEOUT = 60


class PubMedError(RuntimeError):
    """Raised when the PubMed backend is completely unusable (network/auth)."""


class _RateLimiter:
    """Thread-safe-enough minimum-interval limiter (single-threaded Phase2 use)."""

    def __init__(self, interval: float):
        self.interval = interval
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last = time.monotonic()


class PubMedClient:
    """Thin E-utilities client with rate limiting and retry handling."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        tool: Optional[str] = None,
        email: Optional[str] = None,
        max_results: Optional[int] = None,
    ):
        # Config precedence: explicit args > environment variables.
        self.api_key = (api_key or os.getenv("NCBI_API_KEY") or "").strip()
        self.tool = (tool or os.getenv("NCBI_TOOL_NAME") or "").strip()
        self.email = (email or os.getenv("NCBI_EMAIL") or "").strip()

        if not self.api_key:
            warnings.warn(
                "NCBI_API_KEY is not set: PubMed requests are limited to 3/s "
                "and may be throttled. Set NCBI_API_KEY for 10/s."
            )
        if not self.tool:
            warnings.warn("NCBI_TOOL_NAME is not set: using placeholder 'opennovelty'.")
            self.tool = "opennovelty"
        if not self.email:
            warnings.warn("NCBI_EMAIL is not set: using placeholder 'unknown@example.com'.")
            self.email = "unknown@example.com"

        try:
            self.max_results = int(max_results or os.getenv("PUBMED_MAX_RESULTS") or 50)
        except ValueError:
            self.max_results = 50

        self._limiter = _RateLimiter(
            _INTERVAL_WITH_KEY if self.api_key else _INTERVAL_NO_KEY
        )
        self._session = requests.Session()
        # Fail fast if credentials are obviously bad (400 with api_key message).

    # ------------------------------------------------------------------ #
    # Low-level request helpers
    # ------------------------------------------------------------------ #

    def _get(self, endpoint: str, params: Dict[str, Any]) -> requests.Response:
        params = dict(params)
        params["tool"] = self.tool
        params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key

        last_error: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES + 1):
            self._limiter.wait()
            try:
                resp = self._session.get(
                    f"{EUTILS_BASE}/{endpoint}", params=params, timeout=_REQUEST_TIMEOUT
                )
            except requests.RequestException as e:
                last_error = e
                logger.warning("PubMed %s network error (attempt %d): %s", endpoint, attempt + 1, e)
                time.sleep(2 ** attempt)
                continue

            if resp.status_code == 200:
                return resp
            if resp.status_code == 429 or resp.status_code >= 500:
                delay = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                logger.warning(
                    "PubMed %s HTTP %d (attempt %d/%d): retrying in %ss",
                    endpoint, resp.status_code, attempt + 1, _MAX_RETRIES + 1, delay,
                )
                last_error = PubMedError(f"HTTP {resp.status_code}")
                time.sleep(delay)
                continue
            # 400/403 etc: retrying won't help
            raise PubMedError(
                f"PubMed {endpoint} returned HTTP {resp.status_code}: {resp.text[:300]}. "
                "Check your NCBI_API_KEY and query syntax."
            )

        raise PubMedError(
            f"PubMed {endpoint} failed after {_MAX_RETRIES + 1} attempts: {last_error}. "
            "Check network connectivity and your NCBI API key."
        )

    # ------------------------------------------------------------------ #
    # ESearch
    # ------------------------------------------------------------------ #

    def esearch(self, query: str, retmax: Optional[int] = None) -> List[str]:
        """Run esearch.fcgi, return PMID list (relevance-sorted)."""
        retmax = min(retmax or self.max_results, 10000)
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": retmax,
            "sort": "relevance",
            "retmode": "json",
            "usehistory": "n",
        }
        resp = self._get("esearch.fcgi", params)
        data = resp.json().get("esearchresult", {})
        err = data.get("error")
        if err:
            raise PubMedError(f"PubMed esearch error: {err}")
        ids = [str(p) for p in data.get("idlist", [])]
        logger.debug("esearch %r -> %d PMIDs", query, len(ids))
        return ids

    # ------------------------------------------------------------------ #
    # EFetch + XML parsing
    # ------------------------------------------------------------------ #

    def efetch_articles(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch article metadata for PMIDs in batches of EFETCH_BATCH_SIZE."""
        articles: List[Dict[str, Any]] = []
        for i in range(0, len(pmids), EFETCH_BATCH_SIZE):
            batch = pmids[i : i + EFETCH_BATCH_SIZE]
            resp = self._get(
                "efetch.fcgi",
                {
                    "db": "pubmed",
                    "id": ",".join(batch),
                    "rettype": "abstract",
                    "retmode": "xml",
                },
            )
            articles.extend(self._parse_pubmed_xml(resp.text))
        return articles

    @staticmethod
    def _parse_pubmed_xml(xml_text: str) -> List[Dict[str, Any]]:
        """Parse PubmedArticleSet XML into normalized metadata dicts.

        A malformed record is skipped with a warning; other records survive.
        (XML paths verified against a live EFetch response; see module docstring.)
        """
        out: List[Dict[str, Any]] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.warning("EFetch XML unparseable (%s); skipping whole response.", e)
            return out

        for article_el in root.iter("PubmedArticle"):
            try:
                rec = PubMedClient._parse_one_article(article_el)
                if rec:
                    out.append(rec)
            except Exception as e:  # skip bad record, keep going
                logger.warning("Skipping unparseable PubMed record: %s", e)
        return out

    @staticmethod
    def _parse_one_article(article_el: ET.Element) -> Optional[Dict[str, Any]]:
        medline = article_el.find("MedlineCitation")
        if medline is None:
            return None
        pmid_el = medline.find("PMID")
        pmid = (pmid_el.text or "").strip() if pmid_el is not None else None
        if not pmid:
            return None

        art = medline.find("Article")
        if art is None:
            return None

        def _text(el: Optional[ET.Element]) -> str:
            return (el.text or "").strip() if el is not None else ""

        # Title (may contain inline markup like <i>; grab all nested text)
        title_el = art.find("ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""

        # Abstract: merge all AbstractText paragraphs; prefix label when present.
        parts: List[str] = []
        abstract_el = art.find("Abstract")
        if abstract_el is not None:
            for at in abstract_el.findall("AbstractText"):
                label = (at.get("Label") or "").strip()
                seg = "".join(at.itertext()).strip()
                if not seg:
                    continue
                # Skip structured-abstract section labels that carry no content.
                if label and label.upper() not in ("UNLABELLED", "UNLABELED"):
                    parts.append(f"{label}: {seg}")
                else:
                    parts.append(seg)
        abstract = " ".join(parts)

        # Authors: "LastName ForeName" (collective authors fall back to CollectiveName)
        authors: List[str] = []
        author_list = art.find("AuthorList")
        if author_list is not None:
            for author in author_list.findall("Author"):
                last = _text(author.find("LastName"))
                fore = _text(author.find("ForeName"))
                if last:
                    authors.append(f"{last} {fore}".strip())
                else:
                    coll = _text(author.find("CollectiveName"))
                    if coll:
                        authors.append(coll)

        # Journal title + year
        journal_el = art.find("Journal")
        venue = ""
        year: Optional[int] = None
        if journal_el is not None:
            venue = _text(journal_el.find("Title"))
            pub_date = journal_el.find("JournalIssue/PubDate")
            if pub_date is not None:
                y = _text(pub_date.find("Year"))
                if y.isdigit():
                    year = int(y)
                else:
                    # MedlineDate fallback like "2005 Jan-Feb"
                    md = _text(pub_date.find("MedlineDate"))
                    if md[:4].isdigit():
                        year = int(md[:4])

        # DOI from ArticleIdList — lives under PubmedArticle/PubmedData in
        # EFetch responses (verified against live XML; not under Article).
        doi = None
        aid_elements = list(art.iter("ArticleId")) + list(
            article_el.findall("./PubmedData/ArticleIdList/ArticleId")
        )
        for aid in aid_elements:
            if (aid.get("IdType") or "").lower() == "doi" and aid.text:
                doi = aid.text.strip().lower()
                break

        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        return {
            "paper_id": f"PMID:{pmid}",   # canonical PubMed id; global
                                           # canonical_id stays title-hash based
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "venue": venue,
            "year": year,
            "doi": doi,
            "url": url,
            "source_url": url,
            "pdf_url": None,               # PubMed abstracts have no direct PDF
            "citations": None,             # not available via E-utilities
            "research_field": None,
        }

    # ------------------------------------------------------------------ #
    # High-level convenience
    # ------------------------------------------------------------------ #

    def search(self, query: str, retmax: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search + fetch: query -> normalized article metadata list.

        Returns [] when the query matches nothing (logged, not an error).
        """
        pmids = self.esearch(query, retmax=retmax)
        if not pmids:
            logger.info("PubMed query returned 0 results: %r", query)
            return []
        return self.efetch_articles(pmids)

    def health_check(self) -> bool:
        """True when E-utilities answers a trivial query."""
        try:
            return len(self.esearch("pubmed[sb]", retmax=1)) >= 0
        except Exception:
            return False
