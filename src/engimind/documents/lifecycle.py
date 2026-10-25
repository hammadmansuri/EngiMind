from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4


class DocumentVersionRegistry:
    """Tracks a document by filename and keeps the newest version active."""

    def __init__(self, path: str | Path = ".document_registry.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _save(self, payload: dict) -> None:
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _document_id(self, document_name: str) -> str:
        normalized = (document_name or "untitled").strip().lower()
        return sha256(normalized.encode("utf-8")).hexdigest()[:32]

    def _make_version(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}-{uuid4().hex[:8]}"

    def register_version(
        self,
        *,
        document_name: str,
        file_hash: str,
        collection_name: str,
        pages: int,
        chunks: int,
        version: str | None = None,
        active: bool = True,
    ) -> dict:
        document_id = self._document_id(document_name)
        payload = self._load()
        document = payload.setdefault(
            document_id,
            {
                "document_id": document_id,
                "document_name": document_name,
                "versions": [],
            },
        )

        version_name = version or self._make_version()
        entry = {
            "version": version_name,
            "file_hash": file_hash,
            "collection_name": collection_name,
            "pages": int(pages),
            "chunks": int(chunks),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "active": bool(active),
        }

        if active:
            for current in document["versions"]:
                current["active"] = False

        document["versions"].append(entry)
        self._save(payload)
        return self.latest_active(document_name) or entry

    def list_versions(self, document_name: str) -> list[dict]:
        document_id = self._document_id(document_name)
        payload = self._load()
        document = payload.get(document_id)
        if not document:
            return []
        return document.get("versions", [])

    def latest_active(self, document_name: str) -> dict | None:
        versions = self.list_versions(document_name)
        for entry in reversed(versions):
            if entry.get("active"):
                return entry | {"document_id": self._document_id(document_name), "document_name": document_name}
        return None if not versions else versions[-1] | {"document_id": self._document_id(document_name), "document_name": document_name}

    def list_documents(self) -> list[dict]:
        payload = self._load()
        documents: list[dict] = []
        for doc in payload.values():
            versions = doc.get("versions", [])
            active = next((v for v in reversed(versions) if v.get("active")), None)
            documents.append(
                {
                    "document_id": doc["document_id"],
                    "document_name": doc["document_name"],
                    "version_count": len(versions),
                    "latest_active": active,
                }
            )
        return sorted(documents, key=lambda item: item["document_name"].lower())
