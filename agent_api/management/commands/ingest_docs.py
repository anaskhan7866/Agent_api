import os
import time
import random
from django.core.management.base import BaseCommand
from django.conf import settings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

class Command(BaseCommand):
    help = 'Processes raw documents (PDFs) and loads them into a local Chroma vector database using LangChain.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting document ingestion...'))

        # Define directories relative to Django BASE_DIR
        base_dir = settings.BASE_DIR
        raw_docs_dir = os.path.join(base_dir, 'data', 'raw_docs')
        chroma_db_dir = os.path.join(base_dir, 'data', 'chroma_db')

        # Ensure raw_docs_dir exists
        if not os.path.exists(raw_docs_dir):
            os.makedirs(raw_docs_dir)
            self.stdout.write(self.style.WARNING(f"Created directory {raw_docs_dir}. Please add PDF files there and run again."))
            return

        # 1. Load PDF documents
        self.stdout.write(f"Loading PDFs from {raw_docs_dir}...")
        try:
            loader = PyPDFDirectoryLoader(raw_docs_dir)
            documents = loader.load()
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error loading PDFs: {e}"))
            return

        if not documents:
            self.stdout.write(self.style.WARNING("No PDF documents found in the data/raw_docs/ directory."))
            return
            
        self.stdout.write(f"Successfully loaded {len(documents)} document pages.")

        # 2. Split text into chunks
        self.stdout.write("Splitting documents into chunks...")
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            )
            chunks = text_splitter.split_documents(documents)
            self.stdout.write(f"Successfully created {len(chunks)} chunks.")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error splitting documents: {e}"))
            # Ensure you return or exit here in your actual script
            return

        # 3. Initialize Embeddings and Save to Chroma
        try:
            self.stdout.write("Initializing Google Generative AI embeddings...")
            # Reverting back to user's original valid model
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")

            self.stdout.write(f"Saving chunked documents to Chroma DB at {chroma_db_dir}...")
            
            vector_db = Chroma(persist_directory=chroma_db_dir, embedding_function=embeddings)
            
            # UPGRADE 1: Maximize batch size for Gemini API (Max 100)
            batch_size = 100 
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                success = False
                
                # UPGRADE 2: Setup exponential backoff variables
                max_retries = 6
                base_delay = 2 # Start with a 2-second wait
                retries = 0
                
                while not success and retries < max_retries:
                    try:
                        vector_db.add_documents(batch)
                        success = True
                        self.stdout.write(f"Embedded batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                    
                    except Exception as e:
                        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                            # Calculate delay: 2s, 4s, 8s, 16s, 32s... plus slight random jitter
                            delay = base_delay * (2 ** retries) + random.uniform(0, 1)
                            self.stdout.write(self.style.WARNING(f"Rate limit hit! Sleeping {delay:.2f} seconds before retrying..."))
                            time.sleep(delay)
                            retries += 1
                        else:
                            # If it's a different error (like network failure or bad data), crash immediately
                            raise e
                            
                if not success:
                    self.stdout.write(self.style.ERROR(f"Failed to embed batch {i//batch_size + 1} after {max_retries} retries. Skipping."))

            self.stdout.write(self.style.SUCCESS(f"Success! Embedded and saved {len(chunks)} chunks to the vector database."))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error during embedding or database insertion: {e}"))