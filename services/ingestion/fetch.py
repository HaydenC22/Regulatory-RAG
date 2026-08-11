"""Downloads corpus PDFs from the official URLs in corpus/manifest.yaml and
verifies them against corpus/manifest.lock.yaml, updating the lock on first fetch."""

import hashlib
import logging
from pathlib import Path

import requests
import yaml

from services.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MAS's media/document CDN serves an HTML "Maintenance" interstitial (HTTP 200,
# text/html) instead of the PDF when the request doesn't look like a browser —
# a bare custom User-Agent with no Referer gets blocked even though the same
# URL resolves fine in an actual browser. A realistic UA + the document's own
# mas.gov.sg page as Referer reliably gets the real application/pdf response.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def load_manifest(manifest_path: str) -> list[dict]:
    with open(manifest_path, encoding="utf-8") as f:
        return yaml.safe_load(f)["documents"]


def load_lock(lock_path: Path) -> dict:
    if lock_path.exists():
        with open(lock_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_lock(lock_path: Path, lock: dict) -> None:
    with open(lock_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(lock, f, sort_keys=True)


def sha256_of(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def fetch_document(doc: dict, raw_dir: Path, lock: dict) -> bool:
    doc_id = doc["doc_id"]
    pdf_url = doc.get("pdf_url")
    if not pdf_url:
        logger.warning("Skipping %s: no pdf_url in manifest (see page_url=%s)", doc_id, doc.get("page_url"))
        return False

    dest = raw_dir / f"{doc_id}.pdf"
    headers = {
        "User-Agent": BROWSER_USER_AGENT,
        "Accept": "application/pdf,*/*",
        "Referer": doc.get("page_url") or pdf_url,
    }
    try:
        resp = requests.get(pdf_url, timeout=30, headers=headers)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Failed to fetch %s from %s: %s", doc_id, pdf_url, exc)
        return False

    content = resp.content
    content_type = resp.headers.get("content-type", "")
    if not content.startswith(b"%PDF-") or "html" in content_type.lower():
        logger.error(
            "Fetch for %s returned non-PDF content (content-type=%r, first bytes=%r) — "
            "likely blocked/redirected rather than a real failure. Not saving.",
            doc_id,
            content_type,
            content[:80],
        )
        return False

    digest = sha256_of(content)

    prior = lock.get(doc_id)
    if prior and prior["sha256"] != digest:
        logger.warning(
            "%s: downloaded content hash changed since last fetch (%s -> %s). "
            "MAS may have revised this document — review before re-ingesting.",
            doc_id,
            prior["sha256"][:12],
            digest[:12],
        )

    dest.write_bytes(content)
    lock[doc_id] = {"sha256": digest, "source_url": pdf_url, "bytes": len(content)}
    logger.info("Fetched %s (%d bytes, sha256=%s)", doc_id, len(content), digest[:12])
    return True


def main() -> None:
    settings = get_settings()
    raw_dir = Path(settings.corpus_raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    lock_path = Path(settings.corpus_manifest_path).parent / "manifest.lock.yaml"

    documents = load_manifest(settings.corpus_manifest_path)
    lock = load_lock(lock_path)

    ok, failed = 0, []
    for doc in documents:
        if fetch_document(doc, raw_dir, lock):
            ok += 1
        else:
            failed.append(doc["doc_id"])

    save_lock(lock_path, lock)
    logger.info("Fetched %d/%d documents.", ok, len(documents))
    if failed:
        logger.warning("Documents needing manual attention: %s", failed)


if __name__ == "__main__":
    main()
