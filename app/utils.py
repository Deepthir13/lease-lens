"""
utils.py — Shared utility functions for Lease Lens.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def get_ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "mistral")


def get_state() -> str:
    return os.getenv("STATE", "CA")
