# Patent Litigation XAI Dashboard - Software Design Document (SDD)

# 1. Vision

목표는 **새 특허가 기존 Heterogeneous Graph에 삽입된 후 GNN이 소송
가능성을 예측하고, 그 근거를 단계적으로 설명하는 인터랙티브 XAI
Dashboard**를 구축하는 것이다.

## Architecture

``` text
Patent Features
        │
        ▼
Hetero Graph Builder
        │
        ▼
Hetero GNN
        │
        ▼
XAI Compute
(GNNExplainer / PGExplainer / Attention / SubgraphX / Counterfactual)
        │
        ▼
JSON Results
        │
        ▼
Streamlit Viewer
```

# 2. Project Structure

``` text
patent-litigation-xai/
├── app.py
├── config.py
├── requirements.txt
├── results/
├── pages/
├── components/
├── utils/
└── assets/
```

## app.py

-   Sidebar
-   JSON Load
-   Navigation
-   Session State

## config.py

-   Color palette
-   Node size
-   Fonts
-   Theme
-   Default hop/layer

## results/

graph.json prediction.json attention.json explanation.json
counterfactual.json evidence_path.json storyline.json metadata.json

# 3. Page Design

## Overview

Question: - What graph was trained?

Visuals: - Full heterograph - Legend - Query patent highlight - Graph
statistics

## Integration

Question: - How is a new patent inserted?

Animation: New Patent → Owner → IPC → Citation → Integrated Graph

## Message Passing

Question: - How far does information propagate?

Controls: - Hop Slider - Layer Selector - Play Animation

Visual: - Receptive field - Activated nodes

## Prediction

Question: - Which company is predicted?

Visual: - Gauge - Top-K ranking - Confidence - Evidence summary

## Explanation

Tabs: 1. GNNExplainer 2. Attention 3. PGExplainer 4. SubgraphX

Each tab answers a different explanation question.

## Counterfactual

Question: - What changes the prediction?

Visual: - Before graph - After graph - ΔProbability - Removed node/edge

## Conclusion

Auto-generated summary: - Prediction - Main evidence - Counterfactual -
Final explanation

# 4. Components

graph.py - draw_graph() - draw_subgraph()

animation.py - integration_animation() - attention_animation()

gauge.py - probability gauge

timeline.py - presentation progress

tooltip.py - patent/company/ipc hover

report.py - PDF/PNG/HTML export

# 5. Utilities

loader.py - load_graph() - load_prediction() - load_attention() -
load_explanation()

graph_utils.py - filter_hop() - neighborhood() - subgraph()

plot_utils.py - node traces - edge traces - legends - heatmaps

# 6. JSON Schema

graph.json

metadata - node_count - edge_count - query_patent - target_company

nodes - id - type - label - short_label - x - y - importance -
hop_from_src - hop_from_dst - min_hop - degree - community - owner -
ipc - citation_degree - embedding_similarity - is_src - is_dst

edges - source - target - relation - importance - attention - is_path -
is_counterfactual

prediction.json - probability - top_k - confidence - main_evidence

attention.json - layer - edge_id - attention - normalized_attention

explanation.json - node_importance - edge_importance - important_nodes -
important_edges - subgraph

counterfactual.json - removed_object - before - after - delta

evidence_path.json - path - score - hop

storyline.json - summary - main_reason - secondary_reason -
final_conclusion

metadata.json - model - epoch - heads - hidden_dim - prediction_time

# 7. Presentation Story

1.  Show full graph.
2.  Insert new patent.
3.  Show message passing.
4.  Predict litigation.
5.  Explain with four XAI methods.
6.  Compare counterfactual.
7.  Finish with conclusion.

# 8. Design Principles

-   Compute and Viewer completely separated.
-   Viewer never runs inference.
-   All visualizations read JSON only.
-   Same JSON can generate dashboard, paper figures and reports.
-   Future GNN models can be swapped without changing Viewer.

# 9. Future Extensions

-   Temporal graph playback
-   Multiple patent comparison
-   Multi-query mode
-   Graph Transformer support
-   SHAP comparison
-   PDF report generation
