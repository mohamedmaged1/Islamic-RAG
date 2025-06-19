import os
import pdfplumber
import pytesseract
import pickle
from langchain_core.documents import Document
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set tesseract path manually (Windows only)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\\tesseract.exe"

# --- Step 1: Define your PDF file path ---
pdf_path = r".\Data\almoslim_alsawy.pdf"

# --- Step 2: OCR settings ---
language = 'ara' 
ocr_config = f'--psm 6 -l {language}'

langchain_docs = []

with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        # Convert page to image (300 DPI for better OCR)
        pil_image = page.to_image(resolution=600).original

        # Apply Tesseract OCR
        text = pytesseract.image_to_string(pil_image, config=ocr_config)

        # Skip empty pages
        if not text.strip():
            continue

        # Create LangChain Document
        doc = Document(
            page_content=text,
            metadata={"source": pdf_path, "page": i + 1}
        )
        langchain_docs.append(doc)



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(langchain_docs)


# Save the split documents to a file
with open(r".\Data\all_splits.pkl", "wb") as f:
    pickle.dump(all_splits, f)

