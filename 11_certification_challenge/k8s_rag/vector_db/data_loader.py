"""
Data loader for Kubernetes documentation from markdown files.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .vector_store import K8sDocVectorStore


class K8sDocumentationLoader:
    """Loader for Kubernetes documentation from markdown files."""
    
    def __init__(self, data_dir: Path, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the documentation loader.
        
        Args:
            data_dir: Path to the directory containing Kubernetes documentation
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
        """
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
    
    def load_markdown_files(self) -> List[Document]:
        """Load all markdown files from the data directory."""
        documents = []
        
        # Find all markdown files
        md_files = list(self.data_dir.rglob("*.md"))
        
        print(f"📚 Found {len(md_files)} markdown files")
        
        for md_file in md_files:
            try:
                # Read file content
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse frontmatter and content
                frontmatter, body = self._parse_frontmatter(content)
                
                if not body.strip():
                    continue  # Skip empty files
                
                # Create document metadata
                metadata = {
                    "source": str(md_file.relative_to(self.data_dir)),
                    "file_path": str(md_file),
                    "doc_type": self._determine_doc_type(md_file),
                    "title": frontmatter.get("title", md_file.stem),
                    "description": frontmatter.get("description", ""),
                    **frontmatter  # Include all frontmatter
                }
                
                # Create document
                doc = Document(page_content=body, metadata=metadata)
                documents.append(doc)
                
            except Exception as e:
                print(f"❌ Error loading {md_file}: {e}")
                continue
        
        print(f"✅ Successfully loaded {len(documents)} documents")
        
        # Split documents into chunks
        chunked_docs = []
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            chunked_docs.extend(chunks)
        
        print(f"📄 Split into {len(chunked_docs)} chunks")
        
        return chunked_docs
    
    def _parse_frontmatter(self, content: str) -> tuple[Dict[str, Any], str]:
        """Parse YAML frontmatter from markdown content."""
        if not content.startswith('---'):
            return {}, content
        
        try:
            # Find the end of frontmatter
            end_idx = content.find('\n---\n', 4)
            if end_idx == -1:
                return {}, content
            
            # Extract frontmatter and body
            frontmatter_str = content[4:end_idx]
            body = content[end_idx + 5:]  # Skip the closing ---
            
            # Parse YAML
            frontmatter = yaml.safe_load(frontmatter_str) or {}
            
            return frontmatter, body
            
        except Exception as e:
            print(f"⚠️  Error parsing frontmatter: {e}")
            return {}, content
    
    def _determine_doc_type(self, file_path: Path) -> str:
        """Determine document type based on file path."""
        path_parts = file_path.relative_to(self.data_dir).parts
        
        if not path_parts:
            return "unknown"
        
        # Map directory structure to document types
        if "concepts" in path_parts:
            return "concept"
        elif "tasks" in path_parts:
            return "task"
        elif "tutorials" in path_parts:
            return "tutorial"
        elif "reference" in path_parts:
            return "reference"
        elif "setup" in path_parts:
            return "setup"
        elif "contribute" in path_parts:
            return "contribute"
        else:
            return "general"
    
    def load_all_data(self, vector_store: K8sDocVectorStore) -> None:
        """Load all documentation data into the vector store."""
        print("🔄 Loading Kubernetes documentation...")
        
        # Load markdown documents
        documents = self.load_markdown_files()
        
        if documents:
            vector_store.add_documents(documents)
        
        # Show loading summary
        stats = vector_store.get_stats()
        print(f"\n📊 Loading Summary:")
        print(f"   Total documents: {stats['total_documents']}")
        print(f"   Document types:")
        for doc_type, count in stats['document_types'].items():
            print(f"     - {doc_type.replace('_', ' ').title()}: {count}")
        
        print("✅ Data loading complete!")
