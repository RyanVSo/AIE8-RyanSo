import os
from typing import List, Dict, Any
try:
    import PyPDF2
    import fitz  # PyMuPDF - better for text extraction
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("PDF support not available. Install PyPDF2 and PyMuPDF: pip install PyPDF2 PyMuPDF")


class DocumentLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.metadata = []  # Store metadata for each document
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path):
            if self.path.endswith(".txt"):
                self.load_text_file()
            elif self.path.endswith(".pdf") and PDF_AVAILABLE:
                self.load_pdf_file()
            else:
                raise ValueError(
                    f"Unsupported file type: {self.path}. Supported types: .txt, .pdf"
                )
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a supported file."
            )

    def load_text_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            content = f.read()
            self.documents.append(content)
            self.metadata.append({
                "source": self.path,
                "type": "text",
                "page": 1,
                "total_pages": 1,
                "char_count": len(content)
            })

    def load_pdf_file(self):
        if not PDF_AVAILABLE:
            raise ValueError("PDF support not available. Please install PyPDF2 and PyMuPDF")
        
        try:
            # Use PyMuPDF for better text extraction
            doc = fitz.open(self.path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if text.strip():  # Only add non-empty pages
                    self.documents.append(text)
                    self.metadata.append({
                        "source": self.path,
                        "type": "pdf",
                        "page": page_num + 1,
                        "total_pages": len(doc),
                        "char_count": len(text)
                    })
            doc.close()
        except Exception as e:
            print(f"Error reading PDF with PyMuPDF: {e}")
            # Fallback to PyPDF2
            self._load_pdf_with_pypdf2()

    def _load_pdf_with_pypdf2(self):
        with open(self.path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    self.documents.append(text)
                    self.metadata.append({
                        "source": self.path,
                        "type": "pdf",
                        "page": page_num + 1,
                        "total_pages": total_pages,
                        "char_count": len(text)
                    })

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".txt"):
                    self._load_single_text_file(file_path)
                elif file.endswith(".pdf") and PDF_AVAILABLE:
                    self._load_single_pdf_file(file_path)

    def _load_single_text_file(self, file_path):
        with open(file_path, "r", encoding=self.encoding) as f:
            content = f.read()
            self.documents.append(content)
            self.metadata.append({
                "source": file_path,
                "type": "text",
                "page": 1,
                "total_pages": 1,
                "char_count": len(content)
            })

    def _load_single_pdf_file(self, file_path):
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if text.strip():
                    self.documents.append(text)
                    self.metadata.append({
                        "source": file_path,
                        "type": "pdf",
                        "page": page_num + 1,
                        "total_pages": len(doc),
                        "char_count": len(text)
                    })
            doc.close()
        except Exception:
            # Fallback to PyPDF2 for single files too
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        self.documents.append(text)
                        self.metadata.append({
                            "source": file_path,
                            "type": "pdf",
                            "page": page_num + 1,
                            "total_pages": total_pages,
                            "char_count": len(text)
                        })

    def load_documents(self):
        self.load()
        return self.documents

    def load_documents_with_metadata(self):
        self.load()
        return list(zip(self.documents, self.metadata))


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks

    def split_documents_with_metadata(self, documents_with_metadata: List[tuple]) -> List[tuple]:
        """Split documents while preserving and updating metadata for each chunk."""
        chunks_with_metadata = []
        
        for text, metadata in documents_with_metadata:
            text_chunks = self.split(text)
            
            for chunk_idx, chunk in enumerate(text_chunks):
                # Create new metadata for each chunk
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": chunk_idx,
                    "total_chunks": len(text_chunks),
                    "chunk_char_count": len(chunk),
                    "original_char_count": metadata.get("char_count", len(text))
                })
                
                chunks_with_metadata.append((chunk, chunk_metadata))
        
        return chunks_with_metadata


# Backward compatibility
class TextFileLoader(DocumentLoader):
    """Backward compatible TextFileLoader - now inherits from DocumentLoader"""
    def __init__(self, path: str, encoding: str = "utf-8"):
        super().__init__(path, encoding)
        
    def load_file(self):
        # For backward compatibility, redirect to the parent's load_text_file method
        self.load_text_file()


if __name__ == "__main__":
    loader = TextFileLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])
