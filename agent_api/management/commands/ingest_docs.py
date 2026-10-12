import logging
import os
import time
from pathlib import Path
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from agent_api.rag_service import DocumentIdentity, IngestionState, get_embeddings

logger = logging.getLogger('agent_api')


class Command(BaseCommand):
    help = 'Processes raw documents (PDFs) and loads them into a local Chroma vector database using local BGE embeddings idempotently.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--batch-size',
            type=int,
            default=100,
            help='Batch size for vector ingestion (default: 100)'
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting document ingestion with local embeddings...'))

        raw_docs_dir = Path(getattr(settings, 'DOCS_STORAGE_DIR', os.path.join(settings.BASE_DIR, 'data', 'raw_docs'))).resolve()
        chroma_db_dir = str(getattr(settings, 'CHROMA_PATH', os.path.join(settings.BASE_DIR, 'data', 'chroma_db')))
        collection_name = getattr(settings, 'CHROMA_COLLECTION_NAME', 'pdf_documents_bge_small_v1')
        chunk_size = int(getattr(settings, 'RAG_CHUNK_SIZE', 1000))
        chunk_overlap = int(getattr(settings, 'RAG_CHUNK_OVERLAP', 150))

        if not raw_docs_dir.exists():
            raw_docs_dir.mkdir(parents=True, exist_ok=True)
            self.stdout.write(self.style.WARNING(f"Created directory {raw_docs_dir}. Please add PDF files there and run again."))
            return

        pdf_files = sorted([f for f in raw_docs_dir.iterdir() if f.is_file() and f.name.lower().endswith('.pdf')])
        if not pdf_files:
            self.stdout.write(self.style.WARNING("No PDF documents found in the data/raw_docs/ directory."))
            return

        self.stdout.write(f"Found {len(pdf_files)} PDF document(s) in {raw_docs_dir}...")

        all_chunks = []
        all_chunk_ids = []
        doc_chunks_map = {}
        processed_docs_count = 0
        skipped_docs_count = 0

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        for pdf_path in pdf_files:
            doc_identity = DocumentIdentity.create(pdf_path.name)
            doc_id = doc_identity.doc_id

            logger.info(
                "Command ingestion state: %s for %s [doc_id=%s]",
                IngestionState.PROCESSING, doc_identity.display_name, doc_id,
                extra={"doc_id": doc_id, "state": IngestionState.PROCESSING, "doc_name": doc_identity.display_name}
            )

            try:
                loader = PyPDFLoader(str(pdf_path))
                documents = loader.load()
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error loading {pdf_path.name}: {e}"))
                logger.error("Failed to load PDF in ingest_docs: %s (%s)", pdf_path.name, e)
                skipped_docs_count += 1
                continue

            if not documents:
                self.stdout.write(self.style.WARNING(f"Document {pdf_path.name} contains no pages. Skipping."))
                skipped_docs_count += 1
                continue

            doc_chunks = text_splitter.split_documents(documents)
            if not doc_chunks:
                self.stdout.write(self.style.WARNING(f"Document {pdf_path.name} contains no extractable text. Skipping."))
                skipped_docs_count += 1
                continue

            doc_ids_for_file = []
            for idx, chunk in enumerate(doc_chunks):
                chunk_id = f"{doc_id}_chunk_{idx}"
                page = chunk.metadata.get("page", 0)
                chunk.metadata = {
                    "doc_id": doc_id,
                    "display_name": doc_identity.display_name,
                    "storage_name": doc_identity.storage_name,
                    "source": doc_identity.source,
                    "chunk_index": idx,
                    "total_chunks": len(doc_chunks),
                    "page": page,
                }
                doc_ids_for_file.append(chunk_id)
                all_chunks.append(chunk)
                all_chunk_ids.append(chunk_id)

            doc_chunks_map[doc_id] = doc_ids_for_file
            processed_docs_count += 1

        if not all_chunks:
            self.stdout.write(self.style.WARNING("No extractable chunks found in any PDF files."))
            return

        self.stdout.write(f"Successfully prepared {len(all_chunks)} chunks across {len(doc_chunks_map)} document(s).")

        # Initialize Embeddings and Chroma
        try:
            self.stdout.write(f"Initializing local BGE embeddings ({getattr(settings, 'LOCAL_EMBEDDING_MODEL_NAME', 'BAAI/bge-small-en-v1.5')})...")
            embeddings = get_embeddings()
            vector_db = Chroma(
                collection_name=collection_name,
                persist_directory=chroma_db_dir,
                embedding_function=embeddings
            )
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error initializing vector database: {e}"))
            raise CommandError(f"Database initialization failed: {e}")

        # Batch ingestion with retry
        batch_size = options.get('batch_size') or 100
        total_batches = (len(all_chunks) + batch_size - 1) // batch_size
        failed_batches = []
        total_embedded = 0

        self.stdout.write(f"Saving chunked documents to Chroma collection '{collection_name}' at {chroma_db_dir}...")

        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i:i + batch_size]
            batch_ids = all_chunk_ids[i:i + batch_size]
            batch_num = i // batch_size + 1
            success = False

            max_retries = 3
            base_delay = 1
            retries = 0
            last_exception = None

            while not success and retries < max_retries:
                try:
                    vector_db.add_documents(batch_chunks, ids=batch_ids)
                    success = True
                    total_embedded += len(batch_chunks)
                    self.stdout.write(f"Embedded batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)")
                except Exception as e:
                    last_exception = e
                    delay = base_delay * (2 ** retries)
                    self.stdout.write(self.style.WARNING(f"Error embedding batch {batch_num} ({e}). Retrying in {delay}s..."))
                    time.sleep(delay)
                    retries += 1

            if not success:
                failed_batches.append({
                    "batch_num": batch_num,
                    "chunk_count": len(batch_chunks),
                    "error": str(last_exception)
                })
                self.stdout.write(self.style.ERROR(f"Failed to embed batch {batch_num}/{total_batches} after {retries} attempts."))

        # If any batch failed, NEVER prune obsolete chunks - preserve existing database state
        if failed_batches:
            self.stdout.write(self.style.ERROR(
                f"Document ingestion completed with failures: {len(failed_batches)} batch(es) failed out of {total_batches}. "
                f"Preserving existing chunks in vector database without pruning."
            ))
            logger.error("Bulk ingestion failed for %d batch(es); aborting obsolete chunk pruning.", len(failed_batches))
            raise CommandError(f"Document ingestion failed: {len(failed_batches)} batch(es) could not be embedded.")

        # Prune obsolete chunk IDs ONLY after all batches succeeded and indexing is verified
        for doc_id, valid_chunk_ids in doc_chunks_map.items():
            try:
                # Verify that all target chunk IDs were successfully indexed before pruning
                verification = vector_db.get(ids=valid_chunk_ids)
                indexed_ids = set(verification.get('ids', []))
                if set(valid_chunk_ids).issubset(indexed_ids):
                    existing = vector_db.get(where={"doc_id": doc_id})
                    existing_ids = set(existing.get('ids', []))
                    obsolete_ids = list(existing_ids - set(valid_chunk_ids))
                    if obsolete_ids:
                        vector_db.delete(ids=obsolete_ids)
                        logger.info("Pruned %d obsolete chunks for doc_id=%s", len(obsolete_ids), doc_id)
                else:
                    logger.warning("Verification failed for doc_id=%s; new chunks missing from vector DB. Skipping pruning.", doc_id)
            except Exception as prune_err:
                logger.warning("Error pruning obsolete chunk IDs for %s: %s", doc_id, prune_err)

        self.stdout.write(self.style.SUCCESS(
            f"Success! Embedded and saved {total_embedded} chunks across {processed_docs_count} document(s) "
            f"into collection '{collection_name}'."
        ))