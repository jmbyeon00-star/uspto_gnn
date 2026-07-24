"""config.py — Color palette, theme, and default settings."""
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# ── Dark theme ────────────────────────────────────────────────────────────────
BG_COLOR     = "#0d1117"
BG_SECONDARY = "#161b22"
TEXT_COLOR   = "#e6edf3"
BORDER_COLOR = "#30363d"

# ── Node colors (by type) ─────────────────────────────────────────────────────
NODE_COLORS = {
    "patent_src":          "#ff4b4b",   # Plaintiff (red star)
    "patent_dst":          "#1e90ff",   # Defendant (blue diamond, XAI mode)
    "patent_predicted_gt": "#f1c40f",   # Ground truth predicted (gold diamond)
    "patent_predicted":    "#5d6b98",   # Other predicted defendants (muted slate)
    "patent":              "#9b59b6",   # Normal patent (purple)
    "company":             "#2ecc71",   # Company (green)
    "ipc":                 "#f39c12",   # IPC (orange/yellow)
    "path":                "#f1c40f",   # Evidence path highlight
}

# ── Edge colors (by relation) ─────────────────────────────────────────────────
EDGE_COLORS = {
    "citation":     "#f39c12",  # Orange
    "litigation":   "#e74c3c",  # Red
    "owner":        "#2ecc71",  # Green
    "ipc":          "#3498db",  # Blue
    "predicted":    "#c678dd",  # Purple — predicted lawsuit edges
    "predicted_gt": "#f1c40f",  # Gold — ground truth predicted edge
}

# ── Plotly chart theme ────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    plot_bgcolor  = BG_COLOR,
    paper_bgcolor = BG_COLOR,
    font_color    = TEXT_COLOR,
    margin        = dict(t=40, b=30, l=30, r=20),
)

# ── PyVis defaults ────────────────────────────────────────────────────────────
PYVIS_OPTIONS = """{
  "physics": {"enabled": false},
  "edges": {
    "smooth": {"type": "curvedCW", "roundness": 0.1},
    "arrows": {"to": {"enabled": true, "scaleFactor": 0.45}},
    "scaling": {"min": 1, "max": 16}
  },
  "nodes": {
    "borderWidth": 2,
    "shadow": {"enabled": true}
  },
  "interaction": {
    "hover": true,
    "navigationButtons": true,
    "zoomView": true
  }
}"""

# ── Default controls ──────────────────────────────────────────────────────────
DEFAULT_THRESHOLD = 0.05
DEFAULT_TOPK      = 10
DEFAULT_NODE_SCALE = 28
DEFAULT_KHOP      = 2
