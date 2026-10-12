import io
import json
import os
from pathlib import Path
from unittest.mock import patch

from django.conf import settings
from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import RequestFactory, TestCase, override_settings
from django.urls import reverse
from rest_framework.exceptions import ValidationError
from pypdf import PdfWriter

from agent_api.views import UploadPDFView, resolve_safe_doc_path


def create_minimal_pdf_bytes():
    """Generates a valid in-memory 1-page PDF file."""
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class BaseSecurityTestCase(TestCase):
    def setUp(self):
        super().setUp()
        cache.clear()

    def tearDown(self):
        cache.clear()
        super().tearDown()


class PathTraversalSecurityTestCase(BaseSecurityTestCase):
    def setUp(self):
        super().setUp()
        self.upload_url = reverse('upload-pdf')
        self.valid_pdf_content = create_minimal_pdf_bytes()
        self.rf = RequestFactory()

    def test_resolve_safe_doc_path_rejects_all_traversal_attempts(self):
        traversal_attempts = [
            '../../secret.pdf',
            '..\\..\\secret.pdf',
            '/etc/passwd.pdf',
            'C:\\Windows\\System32\\calc.pdf',
            'sub/folder/doc.pdf',
            'sub\\folder\\doc.pdf',
            'valid.pdf\x00.exe',
            '..',
            '/',
            '\\',
        ]
        for attempt in traversal_attempts:
            with self.subTest(attempt=attempt):
                with self.assertRaises(ValidationError):
                    resolve_safe_doc_path(attempt)

    def test_upload_path_traversal_dot_dot_rejected(self):
        request = self.rf.post(self.upload_url)
        file = SimpleUploadedFile('test.pdf', self.valid_pdf_content, content_type='application/pdf')
        file._name = '../../traversal_test.pdf'
        request.FILES['file'] = file
        response = UploadPDFView.as_view()(request)
        self.assertEqual(response.status_code, 400)
        self.assertIn('Path traversal', response.data.get('error', ''))
        # Ensure file was not created outside storage dir
        forbidden_path = Path(settings.BASE_DIR) / 'traversal_test.pdf'
        self.assertFalse(forbidden_path.exists())

    def test_upload_path_traversal_windows_backslash_rejected(self):
        request = self.rf.post(self.upload_url)
        file = SimpleUploadedFile('test.pdf', self.valid_pdf_content, content_type='application/pdf')
        file._name = '..\\..\\win_traversal.pdf'
        request.FILES['file'] = file
        response = UploadPDFView.as_view()(request)
        self.assertEqual(response.status_code, 400)
        self.assertIn('Path traversal', response.data.get('error', ''))

    def test_upload_nested_path_rejected(self):
        request = self.rf.post(self.upload_url)
        file = SimpleUploadedFile('test.pdf', self.valid_pdf_content, content_type='application/pdf')
        file._name = 'subdir/test.pdf'
        request.FILES['file'] = file
        response = UploadPDFView.as_view()(request)
        self.assertEqual(response.status_code, 400)

    def test_upload_null_byte_rejected(self):
        request = self.rf.post(self.upload_url)
        file = SimpleUploadedFile('test.pdf', self.valid_pdf_content, content_type='application/pdf')
        file._name = 'test.pdf\x00.exe'
        request.FILES['file'] = file
        response = UploadPDFView.as_view()(request)
        self.assertEqual(response.status_code, 400)

    def test_delete_path_traversal_rejected(self):
        response = self.client.delete(
            self.upload_url,
            data=json.dumps({'filename': '../../core_project/settings.py'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        # Verify settings.py was NOT deleted
        self.assertTrue((Path(settings.BASE_DIR) / 'core_project' / 'settings.py').exists())

    def test_delete_backslash_traversal_rejected(self):
        response = self.client.delete(
            self.upload_url,
            data=json.dumps({'filename': '..\\..\\manage.py'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        self.assertTrue((Path(settings.BASE_DIR) / 'manage.py').exists())


class InvalidPDFSecurityTestCase(BaseSecurityTestCase):
    def setUp(self):
        super().setUp()
        self.upload_url = reverse('upload-pdf')

    def test_upload_fake_pdf_text_content_rejected(self):
        fake_content = b"This is plain text with a .pdf extension, not an actual PDF."
        file = SimpleUploadedFile('fake.pdf', fake_content, content_type='application/pdf')
        response = self.client.post(self.upload_url, {'file': file})
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json())

    def test_upload_corrupted_pdf_structure_rejected(self):
        corrupted_content = b"%PDF-1.4\ncorrupted bytes that are definitely not parseable pdf objects..."
        file = SimpleUploadedFile('corrupted.pdf', corrupted_content, content_type='application/pdf')
        response = self.client.post(self.upload_url, {'file': file})
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json())

    def test_upload_empty_file_rejected(self):
        file = SimpleUploadedFile('empty.pdf', b'', content_type='application/pdf')
        response = self.client.post(self.upload_url, {'file': file})
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json())

    def test_upload_non_pdf_extension_rejected(self):
        valid_pdf = create_minimal_pdf_bytes()
        file = SimpleUploadedFile('document.docx', valid_pdf, content_type='application/pdf')
        response = self.client.post(self.upload_url, {'file': file})
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json())

    def test_upload_pdf_with_no_extractable_text_returns_400(self):
        valid_blank_pdf = create_minimal_pdf_bytes()
        file = SimpleUploadedFile('blank.pdf', valid_blank_pdf, content_type='application/pdf')
        response = self.client.post(self.upload_url, {'file': file})
        self.assertEqual(response.status_code, 400)
        self.assertIn('no extractable text', response.json().get('error', '').lower())


class OversizedUploadSecurityTestCase(BaseSecurityTestCase):
    def setUp(self):
        super().setUp()
        self.upload_url = reverse('upload-pdf')

    def test_upload_exceeds_max_size_limit_rejected(self):
        valid_pdf = create_minimal_pdf_bytes()
        # Set max upload size to 10 bytes (smaller than minimal valid pdf)
        with override_settings(MAX_UPLOAD_SIZE_BYTES=10):
            file = SimpleUploadedFile('oversized.pdf', valid_pdf, content_type='application/pdf')
            response = self.client.post(self.upload_url, {'file': file})
            self.assertEqual(response.status_code, 400)
            self.assertIn('exceeds the maximum allowed limit', response.json().get('error', ''))


class DuplicateFilenameSecurityTestCase(BaseSecurityTestCase):
    def setUp(self):
        super().setUp()
        self.upload_url = reverse('upload-pdf')
        self.valid_pdf = create_minimal_pdf_bytes()

    @patch('agent_api.views.ingest_single_pdf', return_value=1)
    def test_duplicate_filename_rejected_409_conflict(self, mock_ingest):
        filename = 'unique_contract.pdf'
        target_file = Path(settings.DOCS_STORAGE_DIR) / filename
        if target_file.exists():
            target_file.unlink()

        try:
            # First upload succeeds
            file1 = SimpleUploadedFile(filename, self.valid_pdf, content_type='application/pdf')
            resp1 = self.client.post(self.upload_url, {'file': file1})
            self.assertEqual(resp1.status_code, 200)

            # Second upload with same name is blocked with 409 Conflict
            file2 = SimpleUploadedFile(filename, self.valid_pdf, content_type='application/pdf')
            resp2 = self.client.post(self.upload_url, {'file': file2})
            self.assertEqual(resp2.status_code, 409)
            self.assertIn('already exists', resp2.json().get('error', ''))
        finally:
            if target_file.exists():
                target_file.unlink()


class AuthenticationSecurityTestCase(BaseSecurityTestCase):
    def setUp(self):
        super().setUp()
        self.query_url = reverse('query-rag')
        self.upload_url = reverse('upload-pdf')

    @override_settings(API_REQUIRE_AUTH=True, API_SECRET_KEY='secure-test-token-123')
    def test_unauthorized_requests_rejected_when_auth_required(self):
        # Query endpoint without auth
        resp_query = self.client.post(
            self.query_url,
            data=json.dumps({'query': 'test query'}),
            content_type='application/json'
        )
        self.assertEqual(resp_query.status_code, 403)

        # Upload endpoint without auth
        resp_upload = self.client.get(self.upload_url)
        self.assertEqual(resp_upload.status_code, 403)

        # Delete endpoint without auth
        resp_delete = self.client.delete(
            self.upload_url,
            data=json.dumps({'filename': 'doc.pdf'}),
            content_type='application/json'
        )
        self.assertEqual(resp_delete.status_code, 403)

    @override_settings(API_REQUIRE_AUTH=True, API_SECRET_KEY='secure-test-token-123')
    @patch('agent_api.views.query_rag', return_value={'answer': 'ok', 'sources': []})
    def test_authorized_request_succeeds_with_api_key_header(self, mock_query):
        resp = self.client.post(
            self.query_url,
            data=json.dumps({'query': 'test query'}),
            content_type='application/json',
            HTTP_X_API_KEY='secure-test-token-123'
        )
        self.assertEqual(resp.status_code, 200)

    @override_settings(API_REQUIRE_AUTH=False)
    @patch('agent_api.views.query_rag', return_value={'answer': 'ok', 'sources': []})
    def test_public_access_allowed_when_auth_not_required(self, mock_query):
        resp = self.client.post(
            self.query_url,
            data=json.dumps({'query': 'test query'}),
            content_type='application/json'
        )
        self.assertEqual(resp.status_code, 200)


class XSSPayloadSecurityTestCase(BaseSecurityTestCase):
    def setUp(self):
        super().setUp()
        self.upload_url = reverse('upload-pdf')
        self.chat_url = reverse('chat-interface')
        self.valid_pdf = create_minimal_pdf_bytes()
        self.rf = RequestFactory()

    def test_malicious_script_tags_in_filename_rejected(self):
        request = self.rf.post(self.upload_url)
        file = SimpleUploadedFile('test.pdf', self.valid_pdf, content_type='application/pdf')
        file._name = '<svg onload=alert(1)>.pdf'
        request.FILES['file'] = file
        response = UploadPDFView.as_view()(request)
        self.assertEqual(response.status_code, 400)
        self.assertIn('Invalid characters', response.data.get('error', ''))

    def test_malicious_script_closing_tag_with_slash_rejected(self):
        request = self.rf.post(self.upload_url)
        file = SimpleUploadedFile('test.pdf', self.valid_pdf, content_type='application/pdf')
        file._name = '<script>alert(1)</script>.pdf'
        request.FILES['file'] = file
        response = UploadPDFView.as_view()(request)
        self.assertEqual(response.status_code, 400)

    def test_malicious_img_onerror_in_filename_rejected(self):
        request = self.rf.post(self.upload_url)
        file = SimpleUploadedFile('test.pdf', self.valid_pdf, content_type='application/pdf')
        file._name = '"><img src=x onerror=alert(1)>.pdf'
        request.FILES['file'] = file
        response = UploadPDFView.as_view()(request)
        self.assertEqual(response.status_code, 400)
        self.assertIn('Invalid characters', response.data.get('error', ''))

    def test_chat_interface_includes_dompurify_sanitizer(self):
        response = self.client.get(self.chat_url)
        self.assertEqual(response.status_code, 200)
        content = response.content.decode('utf-8')
        # Ensure DOMPurify is included for client-side markdown sanitization
        self.assertIn('dompurify', content.lower())
        # Ensure safe CSRF token is present
        self.assertIn('csrfmiddlewaretoken', content)


class GenericErrorHandlingSecurityTestCase(BaseSecurityTestCase):
    def setUp(self):
        super().setUp()
        self.query_url = reverse('query-rag')
        self.upload_url = reverse('upload-pdf')
        self.valid_pdf = create_minimal_pdf_bytes()

    @patch('agent_api.views.query_rag', side_effect=RuntimeError("Sensitive DB leak: C:\\Secret\\Path\\api_key=XYZ"))
    def test_server_error_returns_generic_error_without_sensitive_info(self, mock_query):
        response = self.client.post(
            self.query_url,
            data=json.dumps({'query': 'trigger failure'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 500)
        data = response.json()
        error_msg = data.get('error', '')
        # Check that generic error is returned
        self.assertEqual(error_msg, "An internal error occurred while processing your query. Please try again later.")
        # Ensure NO filesystem paths or secret leaks are present
        self.assertNotIn('C:\\Secret', error_msg)
        self.assertNotIn('api_key=XYZ', error_msg)
        self.assertNotIn('Traceback', error_msg)

    @patch('agent_api.views.ingest_single_pdf', side_effect=RuntimeError("Internal Chroma store crash"))
    def test_ingest_failure_cleans_up_file_and_returns_generic_error(self, mock_ingest):
        filename = 'failed_ingest.pdf'
        target_path = Path(settings.DOCS_STORAGE_DIR) / filename
        file = SimpleUploadedFile(filename, self.valid_pdf, content_type='application/pdf')
        response = self.client.post(self.upload_url, {'file': file})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json().get('error'), "An error occurred while ingesting and processing the PDF.")
        # File must be cleaned up, not left orphaned on disk
        self.assertFalse(target_path.exists())


class RAGServiceTestCase(TestCase):
    @patch('agent_api.views.query_rag', return_value={
        "answer": "Automotive leasing revenue compared to energy generation...",
        "sources": [{"page_content": "Source text...", "metadata": {"source": "tsla.pdf"}}]
    })
    def test_query_revenue_comparison(self, mock_rag):
        url = reverse('query-rag')
        payload = {
            "query": "Compare the revenue generated from automotive leasing versus energy generation and storage."
        }
        
        response = self.client.post(
            url, 
            data=json.dumps(payload), 
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200, response.json().get('error', 'Unknown Error'))
        data = response.json()
        
        self.assertIn('answer', data)
        self.assertIn('sources', data)

    def test_query_rag_sanitizes_malformed_history_items(self):
        url = reverse('query-rag')
        payload = {
            "query": "Hello",
            "history": ["invalid string item", 123, {"role": "user", "content": "Valid"}]
        }
        with patch('agent_api.views.query_rag', return_value={"answer": "ok", "sources": []}) as mock_query:
            response = self.client.post(
                url, 
                data=json.dumps(payload), 
                content_type='application/json'
            )
            self.assertEqual(response.status_code, 200)
            # Verify only valid dict items were passed through
            mock_query.assert_called_once_with("Hello", [{'role': 'user', 'content': 'Valid'}])

    def test_delete_via_query_param(self):
        filename = 'query_param_delete.pdf'
        target_path = Path(settings.DOCS_STORAGE_DIR) / filename
        target_path.write_bytes(create_minimal_pdf_bytes())
        
        upload_url = reverse('upload-pdf')
        with patch('agent_api.views.remove_pdf', return_value=1):
            response = self.client.delete(f"{upload_url}?filename={filename}")
            self.assertEqual(response.status_code, 200)
            self.assertFalse(target_path.exists())

    def test_financial_calculator_and_tool_names(self):
        from agent_api.rag_service import financial_calculator, document_search
        self.assertEqual(document_search.name, "document_search")
        self.assertEqual(financial_calculator.invoke("10 + 5 * 2"), "20")
        self.assertEqual(financial_calculator.invoke("+5"), "5")
        # Exponent DoS protection
        self.assertIn("Exponent is too large", financial_calculator.invoke("2 ** 10000"))


def make_pdf_with_text(text: str = "Nexus document content for RAG testing.") -> bytes:
    escaped_text = text.replace("(", "").replace(")", "")
    s = f"BT\n/F1 12 Tf\n72 712 Td\n({escaped_text}) Tj\nET".encode('latin-1', errors='replace')
    return (
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length " + str(len(s)).encode() + b" >>\nstream\n" + s + b"\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000229 00000 n \n0000000324 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n393\n%%EOF"
    )


class ReliableIngestionAndDeletionTestCase(BaseSecurityTestCase):
    def setUp(self):
        super().setUp()
        import uuid
        import chromadb
        from langchain_core.embeddings import FakeEmbeddings
        from langchain_chroma import Chroma

        self.upload_url = reverse('upload-pdf')
        self.fake_embeddings = FakeEmbeddings(size=10)
        self.chroma_client = chromadb.Client()
        self.col_name = f"test_col_{uuid.uuid4().hex}"
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=self.col_name,
            embedding_function=self.fake_embeddings
        )
        self.pdf_content_v1 = make_pdf_with_text("Version 1 content: Alpha document qualitative report.")
        self.pdf_content_v2 = make_pdf_with_text("Version 2 updated content: Alpha document revised metrics.")
        self.storage_dir = Path(settings.DOCS_STORAGE_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def test_document_identity_and_deterministic_chunk_ids(self):
        from agent_api.rag_service import compute_doc_id, DocumentIdentity, ingest_single_pdf

        doc_id1 = compute_doc_id("test_report.pdf")
        doc_id2 = compute_doc_id("TEST_REPORT.PDF")
        doc_id3 = compute_doc_id("C:\\Windows\\Path\\test_report.pdf")
        self.assertEqual(doc_id1, doc_id2)
        self.assertEqual(doc_id1, doc_id3)
        self.assertTrue(doc_id1.startswith("doc_"))

        identity = DocumentIdentity.create("test_report.pdf", display_name="Display Name")
        self.assertEqual(identity.doc_id, doc_id1)
        self.assertEqual(identity.display_name, "Display Name")
        self.assertEqual(identity.storage_name, "test_report.pdf")
        self.assertEqual(identity.source, "data/raw_docs/test_report.pdf")

        # Ingest file and check assigned chunk IDs and metadata
        file_path = self.storage_dir / "test_report.pdf"
        file_path.write_bytes(self.pdf_content_v1)
        try:
            with patch('agent_api.rag_service.vector_store', self.vector_store):
                chunks_count, returned_doc_id = ingest_single_pdf(str(file_path))
                self.assertEqual(returned_doc_id, doc_id1)
                self.assertGreaterEqual(chunks_count, 1)

                stored = self.vector_store.get(where={"doc_id": doc_id1})
                self.assertEqual(len(stored['ids']), chunks_count)
                expected_first_id = f"{doc_id1}_chunk_0"
                self.assertIn(expected_first_id, stored['ids'])

                meta = stored['metadatas'][0]
                self.assertEqual(meta['doc_id'], doc_id1)
                self.assertEqual(meta['storage_name'], "test_report.pdf")
                self.assertEqual(meta['source'], "data/raw_docs/test_report.pdf")
                self.assertEqual(meta['chunk_index'], 0)
        finally:
            if file_path.exists():
                file_path.unlink()

    def test_ingestion_idempotency_repeated_runs_never_duplicate_vectors(self):
        from agent_api.rag_service import ingest_single_pdf

        file_path = self.storage_dir / "idempotent_doc.pdf"
        file_path.write_bytes(self.pdf_content_v1)
        try:
            with patch('agent_api.rag_service.vector_store', self.vector_store):
                # First run
                ingest_single_pdf(str(file_path))
                initial_count = len(self.vector_store.get()['ids'])
                self.assertGreater(initial_count, 0)

                # Second run (exact same document)
                ingest_single_pdf(str(file_path))
                second_count = len(self.vector_store.get()['ids'])
                self.assertEqual(initial_count, second_count, "Repeated ingestion must not duplicate chunks")

                # Third run
                ingest_single_pdf(str(file_path))
                third_count = len(self.vector_store.get()['ids'])
                self.assertEqual(initial_count, third_count, "Ingestion must remain strictly idempotent")
        finally:
            if file_path.exists():
                file_path.unlink()

    def test_duplicate_upload_without_replace_flag_returns_409(self):
        filename = "duplicate_test.pdf"
        file_path = self.storage_dir / filename
        file_path.write_bytes(self.pdf_content_v1)
        try:
            file_upload = SimpleUploadedFile(filename, self.pdf_content_v1, content_type='application/pdf')
            response = self.client.post(self.upload_url, {'file': file_upload})
            self.assertEqual(response.status_code, 409)
            self.assertIn("already exists", response.json().get('error', ''))
        finally:
            if file_path.exists():
                file_path.unlink()

    def test_duplicate_upload_with_replace_flag_updates_file_and_vectors(self):
        filename = "replaceable_doc.pdf"
        file_path = self.storage_dir / filename
        file_path.write_bytes(self.pdf_content_v1)
        try:
            with patch('agent_api.rag_service.vector_store', self.vector_store):
                # Ingest v1
                from agent_api.rag_service import ingest_single_pdf
                ingest_single_pdf(str(file_path), replace=False)

                v1_stored = self.vector_store.get()
                self.assertIn("Alpha document qualitative report", v1_stored['documents'][0])

                # Replace with v2 via API endpoint with replace=True
                file_upload_v2 = SimpleUploadedFile(filename, self.pdf_content_v2, content_type='application/pdf')
                response = self.client.post(f"{self.upload_url}?replace=true", {'file': file_upload_v2})
                self.assertEqual(response.status_code, 200)
                self.assertIn("Successfully replaced", response.json().get('message', ''))

                # Verify file on disk is updated to v2
                self.assertEqual(file_path.read_bytes(), self.pdf_content_v2)

                # Verify vectors in store updated to v2
                v2_stored = self.vector_store.get()
                self.assertIn("revised metrics", v2_stored['documents'][0])
                self.assertEqual(len(v2_stored['ids']), 1)
        finally:
            if file_path.exists():
                file_path.unlink()

    def test_replacement_failure_preserves_original_file_and_vectors(self):
        filename = "preserve_on_fail.pdf"
        file_path = self.storage_dir / filename
        file_path.write_bytes(self.pdf_content_v1)
        try:
            with patch('agent_api.rag_service.vector_store', self.vector_store):
                # Initial ingestion
                from agent_api.rag_service import ingest_single_pdf
                ingest_single_pdf(str(file_path))

                # Mock failure during replacement
                with patch('agent_api.views.ingest_single_pdf', side_effect=RuntimeError("Simulated embedding outage")):
                    file_upload_v2 = SimpleUploadedFile(filename, self.pdf_content_v2, content_type='application/pdf')
                    response = self.client.post(f"{self.upload_url}?replace=true", {'file': file_upload_v2})
                    self.assertEqual(response.status_code, 500)

                # Verify original file on disk is UNTOUCHED
                self.assertEqual(file_path.read_bytes(), self.pdf_content_v1)

                # Verify original vectors remain intact
                stored = self.vector_store.get()
                self.assertEqual(len(stored['ids']), 1)
                self.assertIn("Alpha document qualitative report", stored['documents'][0])

                # Verify no temporary files remain in storage directory
                temp_files = list(self.storage_dir.glob(".tmp_*"))
                self.assertEqual(len(temp_files), 0)
        finally:
            if file_path.exists():
                file_path.unlink()

    def test_failed_embedding_cleans_up_new_upload_and_partial_vectors(self):
        filename = "failed_new_upload.pdf"
        file_path = self.storage_dir / filename
        try:
            with patch('agent_api.rag_service.vector_store', self.vector_store):
                with patch('agent_api.rag_service._add_documents_with_retry', side_effect=RuntimeError("Chroma DB error")):
                    file_upload = SimpleUploadedFile(filename, self.pdf_content_v1, content_type='application/pdf')
                    response = self.client.post(self.upload_url, {'file': file_upload})
                    self.assertEqual(response.status_code, 500)

                # Verify file on disk was cleaned up
                self.assertFalse(file_path.exists())

                # Verify no partial chunks were left in Chroma
                from agent_api.rag_service import compute_doc_id
                doc_id = compute_doc_id(filename)
                stored = self.vector_store.get(where={"doc_id": doc_id})
                self.assertEqual(len(stored['ids']), 0)
        finally:
            if file_path.exists():
                file_path.unlink()

    def test_deletion_of_one_document_preserves_other_documents(self):
        file_a = self.storage_dir / "doc_a.pdf"
        file_b = self.storage_dir / "doc_b.pdf"
        file_a.write_bytes(make_pdf_with_text("Document A confidential information"))
        file_b.write_bytes(make_pdf_with_text("Document B financial statement"))
        try:
            with patch('agent_api.rag_service.vector_store', self.vector_store):
                from agent_api.rag_service import ingest_single_pdf, compute_doc_id
                ingest_single_pdf(str(file_a))
                ingest_single_pdf(str(file_b))

                doc_id_a = compute_doc_id("doc_a.pdf")
                doc_id_b = compute_doc_id("doc_b.pdf")

                # Both documents exist in vector store
                self.assertEqual(len(self.vector_store.get(where={"doc_id": doc_id_a})['ids']), 1)
                self.assertEqual(len(self.vector_store.get(where={"doc_id": doc_id_b})['ids']), 1)

                # Delete doc_a via DELETE endpoint
                del_response = self.client.delete(self.upload_url, data=json.dumps({'filename': 'doc_a.pdf'}), content_type='application/json')
                self.assertEqual(del_response.status_code, 200)

                # doc_a file and vectors removed
                self.assertFalse(file_a.exists())
                self.assertEqual(len(self.vector_store.get(where={"doc_id": doc_id_a})['ids']), 0)

                # doc_b file and vectors PRESERVED
                self.assertTrue(file_b.exists())
                self.assertEqual(len(self.vector_store.get(where={"doc_id": doc_id_b})['ids']), 1)
                b_stored = self.vector_store.get(where={"doc_id": doc_id_b})
                self.assertIn("Document B financial statement", b_stored['documents'][0])
        finally:
            if file_a.exists():
                file_a.unlink()
            if file_b.exists():
                file_b.unlink()

    def test_remove_pdf_handles_paths_separators_and_doc_id(self):
        file_path = self.storage_dir / "separator_test.pdf"
        file_path.write_bytes(self.pdf_content_v1)
        try:
            with patch('agent_api.rag_service.vector_store', self.vector_store):
                from agent_api.rag_service import ingest_single_pdf, remove_pdf, compute_doc_id
                ingest_single_pdf(str(file_path))
                doc_id = compute_doc_id("separator_test.pdf")

                # Test deletion with backslashes
                windows_path = f"C:\\custom\\path\\separator_test.pdf"
                removed = remove_pdf(file_path=windows_path)
                self.assertGreaterEqual(removed, 1)

                # Vector store now empty for this doc
                self.assertEqual(len(self.vector_store.get(where={"doc_id": doc_id})['ids']), 0)
        finally:
            if file_path.exists():
                file_path.unlink()

    def test_management_command_idempotence(self):
        import tempfile
        import shutil
        from django.core.management import call_command
        temp_dir = Path(tempfile.mkdtemp())
        file1 = temp_dir / "cmd_doc1.pdf"
        file2 = temp_dir / "cmd_doc2.pdf"
        file1.write_bytes(make_pdf_with_text("Command document one text"))
        file2.write_bytes(make_pdf_with_text("Command document two text"))
        try:
            with override_settings(DOCS_STORAGE_DIR=str(temp_dir)):
                with patch('agent_api.management.commands.ingest_docs.Chroma', return_value=self.vector_store), \
                     patch('agent_api.management.commands.ingest_docs.get_embeddings', return_value=self.fake_embeddings):
                    # First run
                    call_command('ingest_docs')
                    count_after_first = len(self.vector_store.get()['ids'])
                    self.assertEqual(count_after_first, 2)

                    # Second run must not increase count
                    call_command('ingest_docs')
                    count_after_second = len(self.vector_store.get()['ids'])
                    self.assertEqual(count_after_first, count_after_second, "Management command must be idempotent")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_management_command_records_failed_batches_and_raises_command_error(self):
        import tempfile
        import shutil
        from django.core.management import call_command
        from django.core.management.base import CommandError
        temp_dir = Path(tempfile.mkdtemp())
        file1 = temp_dir / "cmd_fail.pdf"
        file1.write_bytes(make_pdf_with_text("Command document text for batch failure"))
        try:
            mock_failing_vector_store = patch.object(
                self.vector_store,
                'add_documents',
                side_effect=RuntimeError("Simulated API failure on batch embedding")
            )
            with override_settings(DOCS_STORAGE_DIR=str(temp_dir)):
                with patch('agent_api.management.commands.ingest_docs.Chroma', return_value=self.vector_store), \
                     patch('agent_api.management.commands.ingest_docs.get_embeddings', return_value=self.fake_embeddings), \
                     mock_failing_vector_store:
                    with self.assertRaises(CommandError) as ctx:
                        call_command('ingest_docs')
                    self.assertIn("failed", str(ctx.exception).lower())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class LocalEmbeddingsAndRAGTestCase(TestCase):
    """
    Automated test suite verifying local Hugging Face BGE embeddings,
    vector storage in versioned Chroma collection, retrieval performance,
    Gemini response compatibility, and migration mechanics.
    """
    def setUp(self):
        super().setUp()
        cache.clear()
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        self.chroma_path = os.path.join(self.temp_dir, 'chroma_test_db')
        self.collection_name = 'test_pdf_documents_bge_v1'

    def tearDown(self):
        cache.clear()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def test_local_embeddings_initialization_without_api_key(self):
        """Verify BGE local embedding model initializes without requiring external API keys."""
        from agent_api.rag_service import get_embeddings, BGEHuggingFaceEmbeddings
        embeddings = get_embeddings()
        self.assertIsInstance(embeddings, BGEHuggingFaceEmbeddings)
        # Embedding text should not require any API keys
        vec = embeddings.embed_query("Test query for local embeddings")
        self.assertIsInstance(vec, list)
        self.assertEqual(len(vec), 384)

    def test_document_and_query_embeddings_use_same_model_and_normalization(self):
        """Verify document and query embeddings share the same model and 384 dimension."""
        from agent_api.rag_service import get_embeddings
        import math
        embeddings = get_embeddings()
        doc_vecs = embeddings.embed_documents(["Document passage about quarterly revenue."])
        query_vec = embeddings.embed_query("quarterly revenue")
        
        self.assertEqual(len(doc_vecs), 1)
        self.assertEqual(len(doc_vecs[0]), 384)
        self.assertEqual(len(query_vec), 384)

        # Verify L2 normalization (unit length norm == 1.0)
        doc_norm = math.sqrt(sum(x ** 2 for x in doc_vecs[0]))
        query_norm = math.sqrt(sum(x ** 2 for x in query_vec))
        self.assertAlmostEqual(doc_norm, 1.0, places=3)
        self.assertAlmostEqual(query_norm, 1.0, places=3)

    def test_pdf_ingestion_and_retrieval_in_versioned_collection(self):
        """Verify PDF ingestion stores chunks in the versioned collection with complete metadata."""
        from langchain_chroma import Chroma
        from agent_api.rag_service import get_embeddings, ingest_single_pdf, compute_doc_id
        
        test_vs = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.chroma_path,
            embedding_function=get_embeddings()
        )
        pdf_path = Path(self.temp_dir) / "sample_policy.pdf"
        pdf_path.write_bytes(make_pdf_with_text("Employees are eligible for 25 annual paid leave days."))
        
        try:
            with patch('agent_api.rag_service.vector_store', test_vs):
                result = ingest_single_pdf(str(pdf_path))
                self.assertGreaterEqual(result.chunks_count, 1)
                
                # Check data in Chroma
                doc_id = compute_doc_id("sample_policy.pdf")
                records = test_vs.get(where={"doc_id": doc_id})
                self.assertGreaterEqual(len(records['ids']), 1)
                self.assertEqual(records['metadatas'][0]['doc_id'], doc_id)
                self.assertEqual(records['metadatas'][0]['page'], 0)
                self.assertIn("sample_policy.pdf", records['metadatas'][0]['source'])
                
                # Retrieval probe
                retriever = test_vs.as_retriever(search_kwargs={"k": 1})
                retrieved_docs = retriever.invoke("How many annual leave days are provided?")
                self.assertGreaterEqual(len(retrieved_docs), 1)
                self.assertIn("paid leave days", retrieved_docs[0].page_content)
        finally:
            if pdf_path.exists():
                pdf_path.unlink()

    def test_bulk_and_upload_share_embedding_configuration(self):
        """Verify ingest_docs command and upload view share the same collection and embedding model."""
        from agent_api.rag_service import get_embeddings, get_vector_store
        from django.core.management import call_command

        raw_dir = Path(self.temp_dir) / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        pdf1 = raw_dir / "bulk1.pdf"
        pdf1.write_bytes(make_pdf_with_text("Bulk document content about cloud infrastructure."))
        
        with override_settings(
            DOCS_STORAGE_DIR=str(raw_dir),
            CHROMA_PATH=self.chroma_path,
            CHROMA_COLLECTION_NAME=self.collection_name
        ):
            # Run bulk ingestion command
            call_command('ingest_docs')
            
            # Query through vector store initialized with settings
            vs = get_vector_store(force_reload=True)
            res = vs.similarity_search("cloud infrastructure", k=1)
            self.assertGreaterEqual(len(res), 1)
            self.assertIn("cloud infrastructure", res[0].page_content)

    def test_query_rag_returns_sources_and_answer_with_mocked_llm(self):
        """Verify query_rag() calls retriever, invokes agent executor, and formats answer and sources."""
        from agent_api.rag_service import query_rag
        from langchain_core.documents import Document
        from unittest.mock import MagicMock

        mock_docs = [
            Document(page_content="Quarterly net revenue grew by 18 percent.", metadata={"source": "data/raw_docs/q3_report.pdf", "page": 1})
        ]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = mock_docs
        
        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {"output": "Revenue grew by 18% in Q3."}
        with patch('agent_api.rag_service.retriever', mock_retriever), \
             patch('agent_api.rag_service.agent_executor', mock_executor):
            
            res = query_rag("What was the revenue growth?", history=[{"role": "user", "content": "Hi"}])
            self.assertIn("answer", res)
            self.assertIn("sources", res)
            self.assertEqual(res["answer"], "Revenue grew by 18% in Q3.")
            self.assertEqual(len(res["sources"]), 1)
            self.assertEqual(res["sources"][0]["metadata"]["page"], 1)

    def test_migration_preserves_legacy_collection(self):
        """Verify migrate_embeddings copies/re-embeds documents without deleting legacy collection."""
        from django.core.management import call_command
        import chromadb
        from langchain_chroma import Chroma
        from agent_api.rag_service import get_embeddings

        class DummyLegacyEF:
            def __call__(self, input):
                return [[0.1] * 768 for _ in input]

        client = chromadb.PersistentClient(path=self.chroma_path)
        legacy_coll = client.create_collection("legacy_gemini_coll", embedding_function=DummyLegacyEF())
        legacy_coll.add(
            ids=["legacy_1"],
            documents=["Legacy document text preserved safely."],
            metadatas=[{"source": "data/raw_docs/legacy.pdf", "page": 0, "doc_id": "legacy_doc_1"}]
        )

        with override_settings(
            CHROMA_PATH=self.chroma_path,
            CHROMA_COLLECTION_NAME="migrated_bge_coll",
            DOCS_STORAGE_DIR=self.temp_dir
        ):
            call_command(
                'migrate_embeddings',
                source_collection='legacy_gemini_coll',
                target_collection='migrated_bge_coll'
            )

            # Check legacy collection is still untouched
            reloaded_legacy = client.get_collection("legacy_gemini_coll")
            self.assertEqual(reloaded_legacy.count(), 1)
            self.assertEqual(reloaded_legacy.get()['ids'], ['legacy_1'])

            # Check migrated collection contains the recovered/migrated text
            migrated_coll = client.get_collection("migrated_bge_coll")
            self.assertGreaterEqual(migrated_coll.count(), 1)

    def test_retrieval_evaluation_recall_at_3(self):
        """
        Retrieval Evaluation on a representative domain dataset.
        Evaluates Recall@3 on multi-domain document retrieval using local BGE embeddings.
        Target: Recall@3 >= 90%.
        """
        import logging
        from langchain_chroma import Chroma
        from agent_api.rag_service import get_embeddings

        test_coll_name = "retrieval_eval_collection"
        vs = Chroma(
            collection_name=test_coll_name,
            persist_directory=self.chroma_path,
            embedding_function=get_embeddings()
        )

        # Multi-domain corpus (10 distinct realistic business/technical documents)
        corpus = [
            ("doc_01", "Acme Corp reported Q3 net revenue of $142 million, representing a 22 percent increase year-over-year. Operating income was $31 million."),
            ("doc_02", "All remote employees are eligible for a $500 annual home office equipment reimbursement. Receipts must be submitted by December 15th."),
            ("doc_03", "Antigravity API requires SHA-256 HMAC request signing with timestamp headers within a 300 second drift tolerance window."),
            ("doc_04", "The data retention policy mandates that customer transaction records must be archived in immutable storage for seven years."),
            ("doc_05", "Kubernetes pod autoscaling triggers at 75% average CPU utilization, provisioning up to a maximum of 20 replica pods per cluster."),
            ("doc_06", "The medical leave benefit provides full salary continuation for up to 12 weeks upon submission of medical certification."),
            ("doc_07", "Database backups are executed every 6 hours with automated cross-region replication to us-west-2 and 30-day retention."),
            ("doc_08", "Customer support Tier 1 response SLA is 15 minutes for critical severity incidents and 2 hours for standard tickets."),
            ("doc_09", "The enterprise cybersecurity protocol enforces multi-factor authentication (MFA) with FIDO2 hardware security keys for all staff."),
            ("doc_10", "Nexus RAG system utilizes BAAI/bge-small-en-v1.5 local embeddings with 384 dimensions and normalized cosine vector similarity.")
        ]

        doc_texts = [text for _, text in corpus]
        doc_metas = [{"doc_id": did, "title": f"Doc {did}"} for did, _ in corpus]
        doc_ids = [did for did, _ in corpus]
        vs.add_texts(texts=doc_texts, metadatas=doc_metas, ids=doc_ids)

        # 10 test queries mapped to expected ground truth doc_ids
        eval_queries = [
            ("What was Acme Corp Q3 financial revenue and operating income?", "doc_01"),
            ("How much can remote workers expense for home office desk equipment?", "doc_02"),
            ("What cryptographic algorithm is required for API request signatures and timestamp tolerance?", "doc_03"),
            ("How many years must customer financial transaction records be retained under compliance policy?", "doc_04"),
            ("What is the CPU threshold that triggers Kubernetes horizontal pod autoscaling replicas?", "doc_05"),
            ("How many weeks of full salary are covered under the medical leave benefit policy?", "doc_06"),
            ("How frequently are automated database backups performed and replicated across regions?", "doc_07"),
            ("What is the Tier 1 incident SLA response time for critical priority customer support tickets?", "doc_08"),
            ("Which hardware security token is required for employee multi-factor authentication?", "doc_09"),
            ("What local Hugging Face embedding model and vector dimension does the Nexus RAG pipeline use?", "doc_10")
        ]

        hits_at_3 = 0
        total_queries = len(eval_queries)

        for query_text, expected_doc_id in eval_queries:
            results = vs.similarity_search(query_text, k=3)
            retrieved_ids = [doc.metadata.get("doc_id") for doc in results]
            if expected_doc_id in retrieved_ids:
                hits_at_3 += 1

        recall_at_3 = hits_at_3 / total_queries
        print(f"\n[EVALUATION] Measured Retrieval Recall@3: {recall_at_3 * 100:.1f}% ({hits_at_3}/{total_queries} hits)")

        self.assertGreaterEqual(recall_at_3, 0.90, f"Expected Recall@3 >= 0.90, got {recall_at_3}")


class OllamaIntegrationTestCase(TestCase):
    """
    Automated test suite verifying the complete replacement of Gemini with Ollama (qwen2.5:3b),
    ensuring no Gemini imports or active calls remain, and verifying RAG responses and citations.
    """
    def setUp(self):
        super().setUp()
        cache.clear()

    def test_ollama_llm_initialization_with_settings(self):
        """Verify get_llm() initializes ChatOllama using configured model and base URL."""
        from agent_api.rag_service import get_llm
        from langchain_ollama import ChatOllama

        with override_settings(OLLAMA_MODEL_NAME='qwen2.5:3b', OLLAMA_BASE_URL='http://127.0.0.1:11434'):
            llm_instance = get_llm()
            self.assertIsInstance(llm_instance, ChatOllama)
            self.assertEqual(llm_instance.model, 'qwen2.5:3b')
            self.assertEqual(llm_instance.base_url, 'http://127.0.0.1:11434')

    def test_no_gemini_dependencies_or_imports_remain(self):
        """Verify langchain_google_genai is completely removed and not present in rag_service or settings."""
        import sys
        import agent_api.rag_service as rag_module

        # Ensure langchain_google_genai is not imported in rag_service
        self.assertFalse(hasattr(rag_module, 'ChatGoogleGenerativeAI'))
        self.assertFalse(hasattr(rag_module, 'GoogleGenerativeAIEmbeddings'))
        self.assertNotIn('langchain_google_genai', sys.modules)

        # Verify neither rag_service nor settings require GOOGLE_API_KEY
        self.assertFalse(hasattr(settings, 'GOOGLE_API_KEY'))

    def test_query_rag_returns_answer_and_preserves_citations(self):
        """Verify query_rag() with Ollama agent formats answer and returns document source citations."""
        from agent_api.rag_service import query_rag
        from langchain_core.documents import Document
        from unittest.mock import MagicMock

        mock_docs = [
            Document(page_content="Net income grew to $2.1 billion in fiscal 2025.", metadata={"source": "data/raw_docs/annual_report.pdf", "page": 4})
        ]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = mock_docs

        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {"output": "Net income reached $2.1B according to the 2025 annual report."}

        with patch('agent_api.rag_service.retriever', mock_retriever), \
             patch('agent_api.rag_service.agent_executor', mock_executor):
            result = query_rag("What was the annual net income?")
            self.assertIn("answer", result)
            self.assertIn("sources", result)
            self.assertEqual(result["answer"], "Net income reached $2.1B according to the 2025 annual report.")
            self.assertEqual(len(result["sources"]), 1)
            self.assertEqual(result["sources"][0]["metadata"]["page"], 4)
            self.assertIn("annual_report.pdf", result["sources"][0]["metadata"]["source"])

    def test_live_ollama_invocation_if_server_available(self):
        """Test live Ollama model invocation if local server is active, otherwise skip gracefully."""
        import urllib.request
        try:
            with urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=2) as resp:
                is_available = resp.status == 200
        except Exception:
            is_available = False

        if not is_available:
            self.skipTest("Local Ollama server is not running on http://127.0.0.1:11434")

        from agent_api.rag_service import get_llm
        llm = get_llm()
        resp = llm.invoke("Respond with the single word: PONG")
        self.assertIsInstance(resp.content, str)
        self.assertGreater(len(resp.content), 0)


class ContextSufficiencyAndResilienceTestCase(TestCase):
    """
    Comprehensive test suite verifying:
    - Context-sufficiency workflow (PDF vs Web vs Unverified)
    - Structured citations distinguishing document evidence from web evidence
    - Bulk ingestion chunk preservation on failed batch
    - Verified indexing before obsolete chunk pruning
    - Unavailable service handling (Ollama outages, missing models, missing Serper key, web errors)
    - Zero fallback to cloud LLMs
    """
    def setUp(self):
        super().setUp()
        cache.clear()

    def test_sufficient_context_avoids_unnecessary_web_search(self):
        """When PDF context is evaluated as sufficient, answer is generated from docs and web search is NOT called."""
        from agent_api.rag_service import query_rag
        from langchain_core.documents import Document
        from unittest.mock import MagicMock

        mock_docs = [
            Document(
                page_content="The enterprise annual SLA uptime is guaranteed at 99.95%.",
                metadata={"source": "data/raw_docs/service_sla.pdf", "display_name": "service_sla.pdf", "page": 2}
            )
        ]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = mock_docs

        with patch('agent_api.rag_service.retriever', mock_retriever), \
             patch('agent_api.rag_service.evaluate_context_sufficiency', return_value=True), \
             patch('agent_api.rag_service.generate_document_answer', return_value="The SLA uptime guarantee is 99.95%."), \
             patch('agent_api.rag_service.perform_web_search') as mock_web_search:

            res = query_rag("What is the SLA uptime?")

            # Web search must NEVER be called when PDF context is sufficient
            mock_web_search.assert_not_called()

            # Assert structured result
            self.assertEqual(res["answer"], "The SLA uptime guarantee is 99.95%.")
            self.assertEqual(res["evidence_source"], "document")
            self.assertEqual(len(res["citations"]["documents"]), 1)
            self.assertEqual(res["citations"]["documents"][0]["filename"], "service_sla.pdf")
            self.assertEqual(res["citations"]["documents"][0]["page"], 2)
            self.assertEqual(len(res["citations"]["web"]), 0)
            self.assertEqual(len(res["sources"]), 1)

    def test_insufficient_context_triggers_web_search(self):
        """When PDF context is insufficient, web search is triggered and answer cites web sources."""
        from agent_api.rag_service import query_rag
        from langchain_core.documents import Document
        from unittest.mock import MagicMock

        mock_docs = [
            Document(page_content="Unrelated internal notes.", metadata={"source": "data/raw_docs/notes.pdf", "page": 1})
        ]
        mock_web_results = [
            {
                "title": "Alpha Corp Acquires Beta Inc for $1.2B",
                "url": "https://news.example.com/alpha-beta-deal",
                "snippet": "Alpha Corp finalized its acquisition of Beta Inc for $1.2 billion in cash on Monday."
            }
        ]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = mock_docs

        with patch('agent_api.rag_service.retriever', mock_retriever), \
             patch('agent_api.rag_service.evaluate_context_sufficiency', return_value=False), \
             patch('agent_api.rag_service.perform_web_search', return_value={"success": True, "results": mock_web_results}) as mock_web_search, \
             patch('agent_api.rag_service.evaluate_web_sufficiency', return_value=True), \
             patch('agent_api.rag_service.generate_web_answer', return_value="Alpha Corp acquired Beta Inc for $1.2B."):

            res = query_rag("Did Alpha Corp acquire Beta Inc?")

            # Web search must be triggered
            mock_web_search.assert_called_once()

            # Assert structured web citations
            self.assertEqual(res["answer"], "Alpha Corp acquired Beta Inc for $1.2B.")
            self.assertEqual(res["evidence_source"], "web")
            self.assertEqual(len(res["citations"]["web"]), 1)
            self.assertEqual(res["citations"]["web"][0]["title"], "Alpha Corp Acquires Beta Inc for $1.2B")
            self.assertEqual(res["citations"]["web"][0]["url"], "https://news.example.com/alpha-beta-deal")
            self.assertEqual(len(res["citations"]["documents"]), 0)
            self.assertEqual(len(res["sources"]), 1)
            self.assertEqual(res["sources"][0]["metadata"]["source_type"], "web")

    def test_irrelevant_web_results_does_not_invent_answer(self):
        """When web search returns results but they are irrelevant/insufficient, return unverified response without hallucination."""
        from agent_api.rag_service import query_rag
        from unittest.mock import MagicMock

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        irrelevant_web_results = [
            {
                "title": "Random Recipe for Chocolate Cake",
                "url": "https://recipes.example.com/cake",
                "snippet": "Here is how to bake a delicious chocolate cake with cocoa powder."
            }
        ]

        with patch('agent_api.rag_service.retriever', mock_retriever), \
             patch('agent_api.rag_service.evaluate_context_sufficiency', return_value=False), \
             patch('agent_api.rag_service.perform_web_search', return_value={"success": True, "results": irrelevant_web_results}), \
             patch('agent_api.rag_service.evaluate_web_sufficiency', return_value=False), \
             patch('agent_api.rag_service.generate_web_answer') as mock_web_gen:

            res = query_rag("What is the Q3 corporate compliance revenue?")

            # Must NEVER generate web answer or invent answers when results are irrelevant
            mock_web_gen.assert_not_called()

            # Must return unverified response
            self.assertEqual(res["evidence_source"], "none")
            self.assertIn("cannot be verified", res["answer"].lower())
            self.assertEqual(res["citations"]["web"], [])
            self.assertEqual(res["citations"]["documents"], [])
            self.assertEqual(res["sources"], [])

    def test_evaluate_web_sufficiency_evaluator_logic(self):
        """Verify evaluate_web_sufficiency returns True on affirmation and False on rejection/empty."""
        from agent_api.rag_service import evaluate_web_sufficiency
        from unittest.mock import MagicMock

        # Empty results return False
        self.assertFalse(evaluate_web_sufficiency("query", []))

        mock_llm_yes = MagicMock()
        mock_llm_yes.invoke.return_value = MagicMock(content="YES, direct evidence is present.")
        with patch('agent_api.rag_service.get_llm', return_value=mock_llm_yes):
            self.assertTrue(evaluate_web_sufficiency("query", [{"title": "Title", "snippet": "Snippet"}]))

        mock_llm_no = MagicMock()
        mock_llm_no.invoke.return_value = MagicMock(content="NO, irrelevant to question.")
        with patch('agent_api.rag_service.get_llm', return_value=mock_llm_no):
            self.assertFalse(evaluate_web_sufficiency("query", [{"title": "Title", "snippet": "Snippet"}]))

    def test_local_ollama_failure_during_web_answer_generation(self):
        """Verify that an Ollama outage occurring during web answer generation raises typed error."""
        from agent_api.rag_service import query_rag, OllamaServiceUnavailableError
        from unittest.mock import MagicMock

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        web_results = [{"title": "News", "url": "https://example.com", "snippet": "Relevant news."}]

        with patch('agent_api.rag_service.retriever', mock_retriever), \
             patch('agent_api.rag_service.evaluate_context_sufficiency', return_value=False), \
             patch('agent_api.rag_service.perform_web_search', return_value={"success": True, "results": web_results}), \
             patch('agent_api.rag_service.evaluate_web_sufficiency', return_value=True), \
             patch('agent_api.rag_service.generate_web_answer', side_effect=OllamaServiceUnavailableError("Ollama connection failed")):

            with self.assertRaises(OllamaServiceUnavailableError):
                query_rag("Some live query")

    def test_insufficient_pdf_and_inadequate_web_states_cannot_verify(self):
        """When PDF context is insufficient and web search returns no evidence, state cannot be verified."""
        from agent_api.rag_service import query_rag
        from unittest.mock import MagicMock

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []

        with patch('agent_api.rag_service.retriever', mock_retriever), \
             patch('agent_api.rag_service.evaluate_context_sufficiency', return_value=False), \
             patch('agent_api.rag_service.perform_web_search', return_value={"success": True, "results": []}):

            res = query_rag("What is the top secret formula XYZ-99?")

            self.assertEqual(res["evidence_source"], "none")
            self.assertIn("cannot be verified", res["answer"].lower())
            self.assertEqual(res["citations"]["documents"], [])
            self.assertEqual(res["citations"]["web"], [])
            self.assertEqual(res["sources"], [])

    def test_structured_citation_correctness(self):
        """Verify structured citations strictly conform to required fields and types."""
        from agent_api.rag_service import query_rag
        from langchain_core.documents import Document
        from unittest.mock import MagicMock

        mock_docs = [
            Document(page_content="Policy doc content.", metadata={"display_name": "policy.pdf", "page": 7})
        ]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = mock_docs

        with patch('agent_api.rag_service.retriever', mock_retriever), \
             patch('agent_api.rag_service.evaluate_context_sufficiency', return_value=True), \
             patch('agent_api.rag_service.generate_document_answer', return_value="Document answer."):

            res = query_rag("What is the policy?")
            self.assertIn("citations", res)
            self.assertIn("documents", res["citations"])
            self.assertIn("web", res["citations"])
            doc_cit = res["citations"]["documents"][0]
            self.assertIn("filename", doc_cit)
            self.assertIn("page", doc_cit)
            self.assertEqual(doc_cit["filename"], "policy.pdf")
            self.assertEqual(doc_cit["page"], 7)

    def test_failed_ingestion_batch_preserves_existing_chunks(self):
        """Ensure obsolete chunks are NEVER pruned after a failed batch in bulk ingestion."""
        from django.core.management import call_command
        from django.core.management.base import CommandError
        import tempfile
        from pypdf import PdfWriter
        from unittest.mock import MagicMock

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pdf_file = temp_path / "batch_fail_test.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=72, height=72)
            with open(pdf_file, 'wb') as f:
                writer.write(f)

            mock_vector_db = MagicMock()
            # Simulate batch failure on add_documents
            mock_vector_db.add_documents.side_effect = RuntimeError("Simulated embedding failure")
            mock_vector_db.get.return_value = {"ids": ["chunk_existing_1", "chunk_existing_2"]}

            with patch('agent_api.management.commands.ingest_docs.get_embeddings'), \
                 patch('agent_api.management.commands.ingest_docs.Chroma', return_value=mock_vector_db), \
                 patch('agent_api.management.commands.ingest_docs.PyPDFLoader.load', return_value=[
                     MagicMock(page_content="Some test content", metadata={"page": 0})
                 ]), \
                 override_settings(DOCS_STORAGE_DIR=temp_path):

                with self.assertRaises(CommandError):
                    call_command('ingest_docs', batch_size=1)

                # Obsolete chunk deletion must NEVER be called after a failed batch
                mock_vector_db.delete.assert_not_called()

    def test_successful_ingestion_verifies_indexing_before_pruning(self):
        """Verify successful indexing is checked before deleting old chunks."""
        from django.core.management import call_command
        import tempfile
        from pypdf import PdfWriter
        from unittest.mock import MagicMock

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pdf_file = temp_path / "verify_prune_test.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=72, height=72)
            with open(pdf_file, 'wb') as f:
                writer.write(f)

            mock_vector_db = MagicMock()
            mock_vector_db.add_documents.return_value = None

            from agent_api.rag_service import compute_doc_id
            target_doc_id = compute_doc_id("verify_prune_test.pdf")
            new_chunk_id = f"{target_doc_id}_chunk_0"

            def mock_get(*args, **kwargs):
                if 'ids' in kwargs:
                    return {"ids": [new_chunk_id]}
                if 'where' in kwargs:
                    return {"ids": [new_chunk_id, "obsolete_chunk_old"]}
                return {"ids": []}

            mock_vector_db.get.side_effect = mock_get

            with patch('agent_api.management.commands.ingest_docs.get_embeddings'), \
                 patch('agent_api.management.commands.ingest_docs.Chroma', return_value=mock_vector_db), \
                 patch('agent_api.management.commands.ingest_docs.PyPDFLoader.load', return_value=[
                     MagicMock(page_content="New chunk content", metadata={"page": 0})
                 ]), \
                 override_settings(DOCS_STORAGE_DIR=temp_path):

                call_command('ingest_docs', batch_size=1)

                # Delete should be called for obsolete_chunk_old because verification passed
                mock_vector_db.delete.assert_called_with(ids=["obsolete_chunk_old"])

    def test_unavailable_services_ollama_outage_handling(self):
        """Verify Ollama service outage is handled gracefully with 503 and no cloud fallback."""
        from agent_api.rag_service import OllamaServiceUnavailableError

        with patch('agent_api.rag_service.evaluate_context_sufficiency', side_effect=OllamaServiceUnavailableError("Ollama is down")):
            response = self.client.post(
                reverse('query-rag'),
                data=json.dumps({"query": "What is the policy?"}),
                content_type='application/json'
            )
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["status"], "unavailable")

    def test_unavailable_services_missing_ollama_model(self):
        """Verify missing Ollama model is handled gracefully with 503 and no cloud fallback."""
        from agent_api.rag_service import OllamaModelNotFoundError

        with patch('agent_api.rag_service.evaluate_context_sufficiency', side_effect=OllamaModelNotFoundError("Model not found")):
            response = self.client.post(
                reverse('query-rag'),
                data=json.dumps({"query": "What is the policy?"}),
                content_type='application/json'
            )
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["status"], "model_not_found")

    def test_unavailable_services_missing_serper_credentials(self):
        """Verify missing SERPAPI_API_KEY does not crash and marks evidence as unverified."""
        from agent_api.rag_service import perform_web_search, query_rag
        from unittest.mock import MagicMock

        with patch.dict(os.environ, {}, clear=True):
            res = perform_web_search("latest market updates")
            self.assertFalse(res["success"])
            self.assertIn("missing", res["error"].lower())

            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = []

            with patch('agent_api.rag_service.retriever', mock_retriever), \
                 patch('agent_api.rag_service.evaluate_context_sufficiency', return_value=False):
                rag_res = query_rag("latest market updates")
                self.assertEqual(rag_res["evidence_source"], "none")
                self.assertIn("cannot be verified", rag_res["answer"].lower())

    def test_unavailable_services_web_search_network_error(self):
        """Verify web search HTTP or network error is handled gracefully without crashing."""
        from agent_api.rag_service import perform_web_search

        with patch.dict(os.environ, {"SERPAPI_API_KEY": "dummy_key"}):
            with patch('http.client.HTTPSConnection.request', side_effect=ConnectionResetError("Connection reset by peer")):
                res = perform_web_search("some query")
                self.assertFalse(res["success"])
                self.assertIn("failed", res["error"].lower())
