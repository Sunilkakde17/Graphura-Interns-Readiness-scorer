# utils/resume_parser.py
"""
Resume parsing utilities for extracting text from various file formats.
"""

import re
import pdfplumber
from PyPDF2 import PdfReader
import os


def extract_resume_text_from_file(filepath):
    """
    Extract text from a resume file (PDF, TXT, or DOCX).
    
    Args:
        filepath (str): Path to the resume file
        
    Returns:
        str: Extracted text from the resume
    """
    file_extension = os.path.splitext(filepath)[1].lower()
    
    if file_extension == ".pdf":
        return extract_text_from_pdf(filepath)
    elif file_extension == ".txt":
        return extract_text_from_txt(filepath)
    elif file_extension == ".docx":
        return extract_text_from_docx(filepath)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


def extract_text_from_pdf(filepath):
    """
    Extract text from a PDF file using pdfplumber.
    
    Args:
        filepath (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except Exception as e:
        # Fallback to PyPDF2 if pdfplumber fails
        try:
            reader = PdfReader(filepath)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
        except Exception as e2:
            raise Exception(f"Failed to extract text from PDF: {str(e2)}")
    
    return text.lower() if text else ""


def extract_text_from_txt(filepath):
    """
    Extract text from a TXT file.
    
    Args:
        filepath (str): Path to the TXT file
        
    Returns:
        str: Extracted text from the TXT file
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().lower()


def extract_text_from_docx(filepath):
    """
    Extract text from a DOCX file.
    
    Args:
        filepath (str): Path to the DOCX file
        
    Returns:
        str: Extracted text from the DOCX file
    """
    try:
        import docx
        doc = docx.Document(filepath)
        text = " ".join([paragraph.text for paragraph in doc.paragraphs])
        return text.lower()
    except ImportError:
        raise ImportError("python-docx is required for DOCX files. Install with: pip install python-docx")
    except Exception as e:
        raise Exception(f"Failed to extract text from DOCX: {str(e)}")


def extract_resume_text(text):
    """
    Extract and clean resume text from raw input.
    
    Args:
        text (str): Raw text input
        
    Returns:
        str: Cleaned resume text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep important ones
    text = re.sub(r'[^\w\s\.\,\-\#\+]', ' ', text)
    
    return text.strip()