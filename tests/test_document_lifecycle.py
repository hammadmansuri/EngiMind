import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from engimind.api.main import app
from engimind.documents.lifecycle import DocumentVersionRegistry
from engimind.jobs.indexing_queue import IndexingQueue


class DocumentLifecycleTests(unittest.TestCase):
    def test_registry_tracks_versions_and_latest_active_record(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry = DocumentVersionRegistry(path=Path(tmp_dir) / "registry.json")

            first = registry.register_version(
                document_name="policy.pdf",
                file_hash="abc123",
                collection_name="pdf_abc123",
                pages=10,
                chunks=40,
            )
            second = registry.register_version(
                document_name="policy.pdf",
                file_hash="def456",
                collection_name="pdf_def456",
                pages=11,
                chunks=42,
            )

            self.assertEqual(first["document_id"], second["document_id"])
            self.assertEqual(len(registry.list_versions("policy.pdf")), 2)
            self.assertEqual(registry.latest_active("policy.pdf")["collection_name"], "pdf_def456")

    def test_health_endpoint_is_available(self):
        client = TestClient(app)
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")

    def test_api_requires_key_when_configured(self):
        original = os.getenv("ENGIMIND_API_KEY")
        os.environ["ENGIMIND_API_KEY"] = "super-secret"
        try:
            client = TestClient(app)
            response = client.get("/documents", headers={"X-API-Key": "wrong"})
            self.assertEqual(response.status_code, 401)
            response_ok = client.get("/documents", headers={"X-API-Key": "super-secret"})
            self.assertEqual(response_ok.status_code, 200)
        finally:
            if original is None:
                os.environ.pop("ENGIMIND_API_KEY", None)
            else:
                os.environ["ENGIMIND_API_KEY"] = original

    def test_queue_tracks_job_status(self):
        queue = IndexingQueue()
        job = queue.enqueue(document_name="demo.pdf", file_bytes=b"%PDF-1.4", version="v1")
        self.assertEqual(job["status"], "queued")
        self.assertIn(job["job_id"], queue.jobs)

    def test_queue_processes_jobs_in_background(self):
        processed: dict[str, str] = {}

        def processor(job):
            processed[job.job_id] = job.document_name
            return {"processed": True}

        queue = IndexingQueue(processor=processor)
        job = queue.enqueue(document_name="background.pdf", file_bytes=b"%PDF-1.4", version="v1")

        deadline = time.time() + 2
        while time.time() < deadline:
            status = queue.get_status(job["job_id"])
            if status["status"] == "done":
                break
            time.sleep(0.05)

        self.assertEqual(queue.get_status(job["job_id"])["status"], "done")
        self.assertEqual(processed[job["job_id"]], "background.pdf")


if __name__ == "__main__":
    unittest.main()
