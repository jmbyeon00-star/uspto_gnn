"""utils/loader.py — Load all results JSON files."""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
import streamlit as st

RESULTS_DIR = Path(__file__).parent.parent / "results"


def _load(name: str) -> dict:
    path = RESULTS_DIR / name
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_graph()          -> dict: return _load("graph.json")
@st.cache_data
def load_prediction()     -> dict: return _load("prediction.json")
@st.cache_data
def load_attention()      -> dict: return _load("attention.json")
@st.cache_data
def load_explanation()    -> dict: return _load("explanation.json")
@st.cache_data
def load_counterfactual() -> dict: return _load("counterfactual.json")
@st.cache_data
def load_evidence_path()  -> dict: return _load("evidence_path.json")
@st.cache_data
def load_storyline()      -> dict: return _load("storyline.json")
@st.cache_data
def load_metadata()       -> dict: return _load("metadata.json")
@st.cache_data
def load_full_graph()     -> dict: return _load("full_graph.json")

@st.cache_data
def load_embeddings() -> np.ndarray | None:
    path = RESULTS_DIR / "embeddings.npy"
    if not path.exists():
        return None
    return np.load(str(path))


def check_results() -> list[str]:
    """Return list of missing result files."""
    files = ["graph.json", "prediction.json", "attention.json",
             "explanation.json", "counterfactual.json",
             "evidence_path.json", "storyline.json", "full_graph.json"]
    return [f for f in files if not (RESULTS_DIR / f).exists()]
