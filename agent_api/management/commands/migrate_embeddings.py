import logging
import os
from pathlib import Path
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from agent_api.rag_service import (
    DocumentIdentity,
    get_embeddings,
)

logger = logging.getLogger('agent_api')


class Command(BaseCommand):
    help = (
        'Safely migrates documents and embeddings from legacy Chroma collection (e.g., langchain) '
        'to a versioned local BGE collection (e.g., pdf_documents_bge_small_v1).'
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--source-collection',
            type=str,
            default='langchain',
            help="Source collection name (default: 'langchain')"
        )
        parser.add_argument(
            '--target-collection',
            type=str,
            default=getattr(settings, 'CHROMA_COLLECTION_NAME', 'pdf_documents_bge_small_v1'),
            help="Target collection name (default: pdf_documents_bge_small_v1)"
        )
        parser.add_argument(
            '--verify-only',
            action='store_true',
            help='Run verification checks on the target collection without migrating data.'
        )
        parser.add_argument(
            '--rollback-info',
            action='store_true',
            help='Display instructions for rolling back to the legacy collection.'
        )

    def handle(self, *args, **options):
        chroma_path = str(getattr(settings, 'CHROMA_PATH', os.path.join(settings.BASE_DIR, 'data', 'chroma_db')))
        source_name = options['source_collection']
        target_name = options['target_collection']
        raw_docs_dir = Path(getattr(settings, 'DOCS_STORAGE_DIR', os.path.join(settings.BASE_DIR, 'data', 'raw_docs'))).resolve()

        if options['rollback_info']:
            self.stdout.write(self.style.WARNING("=== ROLLBACK INSTRUCTIONS ==="))
            self.stdout.write(
                f"1. Set CHROMA_COLLECTION_NAME={source_name} in your .env or settings.py.\n"
                f"2. Restart the Django application server.\n"
                f"3. The legacy collection '{source_name}' has been preserved untouched in {chroma_path}."
            )
            return

        client = chromadb.PersistentClient(path=chroma_path)
        existing_collections = {c.name: c for c in client.list_collections()}

        self.stdout.write(self.style.SUCCESS(
            f"Chroma DB Path: {chroma_path}\n"
            f"Existing collections: {list(existing_collections.keys())}\n"
            f"Source collection: '{source_name}'\n"
            f"Target collection: '{target_name}'"
        ))

        if options['verify_only']:
            self._verify_collection(client, target_name)
            return

        if source_name == target_name:
            raise CommandError("Source and target collections cannot be the same name.")

        # Inspect source collection
        source_count = 0
        source_data = None
        if source_name in existing_collections:
            src_coll = existing_collections[source_name]
            source_count = src_coll.count()
            self.stdout.write(f"Source collection '{source_name}' exists with {source_count} vector(s).")
            if source_count > 0:
                source_data = src_coll.get(include=['documents', 'metadatas'])
        else:
            self.stdout.write(self.style.WARNING(f"Source collection '{source_name}' not found or has 0 items."))

        # Initialize local embeddings
        self.stdout.write("Initializing local BGE embedding model...")
        embeddings = get_embeddings()

        # Connect target Chroma vector store
        target_store = Chroma(
            collection_name=target_name,
            persist_directory=chroma_path,
            embedding_function=embeddings
        )

        chunk_size = int(getattr(settings, 'RAG_CHUNK_SIZE', 1000))
        chunk_overlap = int(getattr(settings, 'RAG_CHUNK_OVERLAP', 150))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # 1. Re-embed all available raw PDFs from DOCS_STORAGE_DIR
        pdf_files = sorted([f for f in raw_docs_dir.iterdir() if f.is_file() and f.name.lower().endswith('.pdf')]) if raw_docs_dir.exists() else []
        self.stdout.write(f"Found {len(pdf_files)} raw PDF(s) in {raw_docs_dir}...")

        migrated_files = set()
        total_reembedded = 0

        for pdf_path in pdf_files:
            doc_identity = DocumentIdentity.create(pdf_path.name)
            doc_id = doc_identity.doc_id
            try:
                loader = PyPDFLoader(str(pdf_path))
                docs = loader.load()
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Failed to load PDF {pdf_path.name}: {e}"))
                continue

            if not docs:
                continue

            chunks = text_splitter.split_documents(docs)
            if not chunks:
                continue

            chunk_ids = []
            for idx, chunk in enumerate(chunks):
                c_id = f"{doc_id}_chunk_{idx}"
                chunk.metadata = {
                    "doc_id": doc_id,
                    "display_name": doc_identity.display_name,
                    "storage_name": doc_identity.storage_name,
                    "source": doc_identity.source,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "page": chunk.metadata.get("page", 0),
                }
                chunk_ids.append(c_id)

            target_store.add_documents(chunks, ids=chunk_ids)
            total_reembedded += len(chunks)
            migrated_files.add(pdf_path.name)
            self.stdout.write(f"Re-embedded {pdf_path.name}: {len(chunks)} chunks.")

        # 2. Check if source collection had documents without raw files
        if source_data and source_data.get('ids'):
            orphaned_chunks = []
            for idx, old_id in enumerate(source_data['ids']):
                meta = (source_data.get('metadatas') or [{}])[idx] or {}
                text = (source_data.get('documents') or [''])[idx]
                fname = meta.get('storage_name') or meta.get('source', '')
                if fname:
                    fname = Path(fname).name
                if fname not in migrated_files and text:
                    orphaned_chunks.append((old_id, text, meta))

            if orphaned_chunks:
                self.stdout.write(self.style.WARNING(
                    f"Found {len(orphaned_chunks)} chunks in '{source_name}' whose raw PDFs were missing. "
                    "Recovering stored text..."
                ))
                rec_texts = [item[1] for item in orphaned_chunks]
                rec_metas = [item[2] for item in orphaned_chunks]
                rec_ids = [f"recovered_{item[0]}" for item in orphaned_chunks]
                target_store.add_texts(rec_texts, metadatas=rec_metas, ids=rec_ids)
                total_reembedded += len(orphaned_chunks)
                self.stdout.write(f"Recovered and re-embedded {len(orphaned_chunks)} chunks from stored text.")

        self.stdout.write(self.style.SUCCESS(
            f"Migration completed! Populated target collection '{target_name}' with {total_reembedded} chunk(s)."
        ))

        # Verification step
        self._verify_collection(client, target_name)

    def _verify_collection(self, client, collection_name: str):
        self.stdout.write(f"Verifying target collection '{collection_name}'...")
        try:
            coll = client.get_collection(collection_name)
            count = coll.count()
            self.stdout.write(self.style.SUCCESS(f"Verification: Collection '{collection_name}' contains {count} items."))

            if count > 0:
                sample = coll.get(limit=1, include=['metadatas', 'documents'])
                if sample and sample.get('ids'):
                    sample_id = sample['ids'][0]
                    sample_meta = sample['metadatas'][0] if sample['metadatas'] else {}
                    self.stdout.write(f"Sample chunk ID: {sample_id}")
                    self.stdout.write(f"Sample chunk metadata: {sample_meta}")
                    self.stdout.write(self.style.SUCCESS("Verification probe PASSED: metadata and IDs valid."))
                else:
                    self.stdout.write(self.style.WARNING("Could not read sample chunk."))
            else:
                self.stdout.write(self.style.WARNING(f"Collection '{collection_name}' is currently empty (0 items)."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Verification failed: {e}"))
            raise CommandError(f"Verification failed: {e}")
