import hmac
import logging
import os
from pathlib import Path
import re
import uuid
import pypdf
from django.conf import settings
from django.utils.text import get_valid_filename
from django.views.generic import TemplateView
from rest_framework import status
from rest_framework.exceptions import ValidationError
from rest_framework.parsers import JSONParser, MultiPartParser
from rest_framework.permissions import BasePermission
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView
from .rag_service import compute_doc_id, ingest_single_pdf, query_rag, remove_pdf

logger = logging.getLogger('agent_api')


def resolve_safe_doc_path(raw_filename: str) -> tuple[Path, str]:
    """
    Safely validates and resolves a document filename within DOCS_STORAGE_DIR.
    Rejects traversal attempts, path separators, null bytes, and invalid extensions.
    Returns a tuple of (resolved_path, safe_filename).
    """
    if not raw_filename or not isinstance(raw_filename, str):
        raise ValidationError("A valid filename must be provided.")

    if '\x00' in raw_filename:
        raise ValidationError("Invalid filename: null bytes are not permitted.")

    # Explicitly reject directory traversal indicators and path separators
    if '..' in raw_filename or '/' in raw_filename or '\\' in raw_filename:
        raise ValidationError("Path traversal sequence detected in filename.")

    # Reject dangerous characters such as HTML/script injection or illegal filesystem characters
    if re.search(r'[<>:"/\\|?*\'"\x00-\x1f]', raw_filename):
        raise ValidationError("Invalid characters detected in filename.")

    # Sanitize filename
    base_name = os.path.basename(raw_filename).strip()
    safe_name = get_valid_filename(base_name)

    if not safe_name or not safe_name.lower().endswith('.pdf') or safe_name == '.pdf':
        raise ValidationError("Invalid filename. Only non-empty .pdf files are permitted.")

    # Canonical directory check
    storage_dir = Path(settings.DOCS_STORAGE_DIR).resolve()
    target_path = (storage_dir / safe_name).resolve()

    # Strict path containment verification
    try:
        target_path.relative_to(storage_dir)
    except ValueError:
        raise ValidationError("Path traversal attempt detected.")

    if target_path.parent != storage_dir:
        raise ValidationError("Target path does not resolve to the raw documents directory.")

    return target_path, safe_name


def validate_uploaded_pdf(file_obj) -> None:
    """
    Validates uploaded file size, extension, MIME/magic header bytes, and PDF structure.
    """
    if not file_obj:
        raise ValidationError("No file provided.")

    # Extension validation
    if not file_obj.name or not file_obj.name.lower().endswith('.pdf'):
        raise ValidationError("Only PDF files are supported.")

    # Size validation
    if file_obj.size == 0:
        raise ValidationError("The uploaded file is empty.")

    max_size = getattr(settings, 'MAX_UPLOAD_SIZE_BYTES', 50 * 1024 * 1024)
    if file_obj.size > max_size:
        max_mb = max_size // (1024 * 1024)
        raise ValidationError(f"File size exceeds the maximum allowed limit of {max_mb} MB.")

    # MIME / Content-Type check (if provided)
    content_type = getattr(file_obj, 'content_type', '')
    if content_type and content_type not in ('application/pdf', 'application/x-pdf'):
        raise ValidationError("Invalid content type. Only PDF files are supported.")

    # Magic bytes inspection: must start with %PDF-
    try:
        initial_pos = file_obj.tell()
    except (AttributeError, OSError):
        initial_pos = 0

    header = file_obj.read(1024)
    try:
        file_obj.seek(initial_pos)
    except (AttributeError, OSError):
        pass

    if not header.startswith(b'%PDF-'):
        raise ValidationError("Invalid PDF content. File header does not match PDF format.")

    # PDF structural validation via pypdf
    try:
        reader = pypdf.PdfReader(file_obj)
        if len(reader.pages) == 0:
            raise ValidationError("PDF file contains no readable pages.")
    except ValidationError:
        raise
    except Exception as e:
        logger.warning("PDF structure validation failed: %s", type(e).__name__)
        raise ValidationError("Invalid or corrupted PDF file.")
    finally:
        try:
            file_obj.seek(initial_pos)
        except (AttributeError, OSError):
            pass


class ConfigurableAPIPermission(BasePermission):
    """
    Configurable permission class:
    - If settings.API_REQUIRE_AUTH is False: Allows public access (protected by rate limiting).
    - If settings.API_REQUIRE_AUTH is True: Requires an authenticated user session
      or a valid X-API-KEY / Bearer token matching settings.API_SECRET_KEY.
    """
    def has_permission(self, request, view):
        if not getattr(settings, 'API_REQUIRE_AUTH', False):
            return True

        # Session / Token authenticated user
        if request.user and request.user.is_authenticated:
            return True

        # Shared API Secret Key check
        api_secret = getattr(settings, 'API_SECRET_KEY', None)
        if api_secret:
            provided_key = (
                request.headers.get('X-API-KEY')
                or request.headers.get('X-Api-Key')
            )
            if not provided_key:
                auth_header = request.headers.get('Authorization', '')
                if auth_header.startswith('Bearer '):
                    provided_key = auth_header.split(' ', 1)[1].strip()
            if provided_key and hmac.compare_digest(provided_key, api_secret):
                return True

        return False


class QueryRAGView(APIView):
    permission_classes = [ConfigurableAPIPermission]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = 'query'

    def post(self, request, *args, **kwargs):
        query = request.data.get('query')
        if not query or not isinstance(query, str) or not query.strip():
            return Response(
                {"error": "Please provide a 'query' in the request body."},
                status=status.HTTP_400_BAD_REQUEST
            )

        if len(query) > 4000:
            return Response(
                {"error": "Query length exceeds maximum limit of 4000 characters."},
                status=status.HTTP_400_BAD_REQUEST
            )

        history = request.data.get('history', [])
        if not isinstance(history, list):
            return Response(
                {"error": "Chat history must be a list of messages."},
                status=status.HTTP_400_BAD_REQUEST
            )

        sanitized_history = []
        for msg in history:
            if isinstance(msg, dict):
                role = 'user' if msg.get('role') == 'user' else 'assistant'
                content = str(msg.get('content', ''))
                sanitized_history.append({'role': role, 'content': content})

        try:
            result = query_rag(query, sanitized_history)
            return Response(result, status=status.HTTP_200_OK)
        except Exception:
            logger.exception("Unexpected error in QueryRAGView:")
            return Response(
                {"error": "An internal error occurred while processing your query. Please try again later."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ChatInterfaceView(TemplateView):
    template_name = 'agent_api/chat.html'


class UploadPDFView(APIView):
    parser_classes = [MultiPartParser, JSONParser]
    permission_classes = [ConfigurableAPIPermission]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = 'upload'

    def get(self, request, *args, **kwargs):
        storage_dir = Path(settings.DOCS_STORAGE_DIR).resolve()
        if not storage_dir.exists():
            return Response({"files": []}, status=status.HTTP_200_OK)

        files = sorted([f.name for f in storage_dir.iterdir() if f.is_file() and f.name.lower().endswith('.pdf')])
        return Response({"files": files}, status=status.HTTP_200_OK)

    def delete(self, request, *args, **kwargs):
        filename = None
        if isinstance(request.data, dict):
            filename = request.data.get('filename')
        if not filename:
            filename = request.query_params.get('filename')
        if not filename:
            return Response({"error": "Please provide a 'filename'."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            target_path, safe_name = resolve_safe_doc_path(filename)
        except ValidationError as ve:
            err_msg = str(ve.detail[0] if isinstance(ve.detail, list) else ve.detail)
            return Response({"error": err_msg}, status=status.HTTP_400_BAD_REQUEST)

        if not target_path.exists():
            return Response({"error": f"Document '{safe_name}' not found."}, status=status.HTTP_404_NOT_FOUND)

        try:
            doc_id = compute_doc_id(safe_name)
            removed_chunks = remove_pdf(file_path=str(target_path), doc_id=doc_id, filename=safe_name)
            if target_path.exists():
                target_path.unlink()
            return Response(
                {"message": f"Successfully deleted {safe_name} ({removed_chunks} chunks removed)."},
                status=status.HTTP_200_OK
            )
        except Exception:
            logger.exception("Unexpected error deleting document %s:", safe_name)
            return Response(
                {"error": "An internal error occurred while deleting the document. Please try again later."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def post(self, request, *args, **kwargs):
        file_obj = request.FILES.get('file')
        if not file_obj:
            return Response({"error": "Please provide a 'file' parameter."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            target_path, safe_name = resolve_safe_doc_path(file_obj.name)
        except ValidationError as ve:
            err_msg = str(ve.detail[0] if isinstance(ve.detail, list) else ve.detail)
            return Response({"error": err_msg}, status=status.HTTP_400_BAD_REQUEST)

        try:
            validate_uploaded_pdf(file_obj)
        except ValidationError as ve:
            err_msg = str(ve.detail[0] if isinstance(ve.detail, list) else ve.detail)
            return Response({"error": err_msg}, status=status.HTTP_400_BAD_REQUEST)

        replace_requested = (
            request.data.get('replace') in (True, 'true', 'True', '1', 1) or
            request.query_params.get('replace') in (True, 'true', 'True', '1', 1)
        )

        storage_dir = Path(settings.DOCS_STORAGE_DIR).resolve()
        storage_dir.mkdir(parents=True, exist_ok=True)

        if target_path.exists() and not replace_requested:
            return Response(
                {"error": f"A document named '{safe_name}' already exists. Please delete it or rename your file before uploading."},
                status=status.HTTP_409_CONFLICT
            )

        if target_path.exists() and replace_requested:
            # Safe replacement workflow: stage new upload to temp file
            temp_target_path = storage_dir / f".tmp_{uuid.uuid4().hex}_{safe_name}"
            try:
                with open(temp_target_path, 'wb') as destination:
                    for chunk in file_obj.chunks():
                        destination.write(chunk)
            except Exception:
                logger.exception("Failed to write temporary upload file to disk: %s", safe_name)
                return Response(
                    {"error": "An error occurred while saving the uploaded file."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            try:
                res = ingest_single_pdf(str(temp_target_path), display_name=safe_name, storage_name=safe_name, replace=True)
                if isinstance(res, tuple):
                    chunks_processed, doc_id = res
                elif hasattr(res, 'doc_id'):
                    chunks_processed, doc_id = res.chunks_count, res.doc_id
                else:
                    chunks_processed = int(res)
                    doc_id = compute_doc_id(safe_name)

                # Atomically replace old file on disk with the newly verified file
                os.replace(temp_target_path, target_path)
                return Response(
                    {
                        "message": f"Successfully replaced {safe_name} ({chunks_processed} chunks added).",
                        "doc_id": doc_id
                    },
                    status=status.HTTP_200_OK
                )
            except ValueError as ve:
                logger.warning("PDF replacement rejected with ValueError for %s: %s", safe_name, ve)
                if temp_target_path.exists():
                    try:
                        temp_target_path.unlink()
                    except OSError:
                        pass
                return Response({"error": str(ve)}, status=status.HTTP_400_BAD_REQUEST)
            except Exception:
                logger.exception("Unexpected error replacing PDF %s:", safe_name)
                if temp_target_path.exists():
                    try:
                        temp_target_path.unlink()
                    except OSError:
                        pass
                return Response(
                    {"error": "An error occurred while ingesting and processing the PDF."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

        # Standard new file upload workflow
        try:
            with open(target_path, 'wb') as destination:
                for chunk in file_obj.chunks():
                    destination.write(chunk)
        except Exception:
            logger.exception("Failed to write uploaded file to disk: %s", safe_name)
            return Response(
                {"error": "An error occurred while saving the uploaded file."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        try:
            res = ingest_single_pdf(str(target_path), display_name=safe_name, storage_name=safe_name, replace=False)
            if isinstance(res, tuple):
                chunks_processed, doc_id = res
            elif hasattr(res, 'doc_id'):
                chunks_processed, doc_id = res.chunks_count, res.doc_id
            else:
                chunks_processed = int(res)
                doc_id = compute_doc_id(safe_name)

            return Response(
                {
                    "message": f"Successfully ingested {safe_name} ({chunks_processed} chunks added).",
                    "doc_id": doc_id
                },
                status=status.HTTP_200_OK
            )
        except ValueError as ve:
            logger.warning("PDF ingestion rejected with ValueError for %s: %s", safe_name, ve)
            if target_path.exists():
                try:
                    target_path.unlink()
                except OSError:
                    pass
            return Response({"error": str(ve)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception:
            logger.exception("Unexpected error ingesting PDF %s:", safe_name)
            if target_path.exists():
                try:
                    target_path.unlink()
                except OSError:
                    pass
            return Response(
                {"error": "An error occurred while ingesting and processing the PDF."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
