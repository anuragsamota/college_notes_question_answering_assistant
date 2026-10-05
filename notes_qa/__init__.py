"""College Notes Question-Answering Assistant.

A retrieval-augmented generation (RAG) pipeline that answers natural-language
questions using only the lecture notes and course material you upload.
"""

from notes_qa.config import Settings
from notes_qa.pipeline import Assistant

__all__ = ["Assistant", "Settings"]
__version__ = "0.1.0"
