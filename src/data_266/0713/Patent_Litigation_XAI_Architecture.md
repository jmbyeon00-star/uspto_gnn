# Patent Litigation XAI Dashboard Architecture

## Goal

Separate **Compute** and **Viewer**.

``` text
HeteroGNN
    ↓
XAI Compute
    ↓
JSON Results
    ↓
Streamlit Viewer
```

## Project Structure

``` text
patent-litigation-xai/
├── app.py
├── config.py
├── requirements.txt
├── results/
│   ├── graph.json
│   ├── prediction.json
│   ├── explanation.json
│   ├── attention.json
│   ├── counterfactual.json
│   ├── evidence_path.json
│   ├── storyline.json
│   └── metadata.json
├── pages/
│   ├── overview.py
│   ├── integration.py
│   ├── message_passing.py
│   ├── prediction.py
│   ├── explanation.py
│   ├── counterfactual.py
│   └── conclusion.py
├── components/
│   ├── graph.py
│   ├── legend.py
│   ├── cards.py
│   ├── gauge.py
│   ├── animation.py
│   ├── timeline.py
│   ├── tooltip.py
│   └── report.py
├── utils/
│   ├── loader.py
│   ├── graph_utils.py
│   ├── plot_utils.py
│   ├── colors.py
│   └── session.py
└── assets/
```

## Page Responsibilities

### Overview

-   Input: graph.json
-   Show full heterograph
-   Highlight query patent

### Integration

-   Animate New Patent → Owner → IPC → Citation

### Message Passing

-   Inputs: graph.json, attention.json
-   Hop slider
-   Layer selector

### Prediction

-   Input: prediction.json
-   Gauge
-   Top-K
-   Summary

### Explanation

Tabs: - GNNExplainer - Attention - PGExplainer - SubgraphX

### Counterfactual

-   Before / After
-   ΔP

### Conclusion

-   Storyline
-   Final explanation

## JSON Design

### graph.json

metadata - node_count - edge_count - query_patent - target_company

nodes - id - type - label - short_label - x - y - importance -
hop_from_src - hop_from_dst - min_hop - degree - community - is_src -
is_dst - owner - ipc - citation_degree - embedding_similarity

edges - source - target - relation - importance - attention - is_path -
is_counterfactual

### prediction.json

-   query_patent
-   target_company
-   probability
-   top_k
-   confidence
-   main_evidence

### explanation.json

-   important_nodes
-   important_edges
-   node_importance
-   edge_importance
-   subgraph

### attention.json

-   edge_id
-   attention
-   normalized_attention

### counterfactual.json

-   removed_object
-   before
-   after
-   delta

### evidence_path.json

-   path
-   score
-   hop
-   importance

### storyline.json

-   summary
-   main_reason
-   secondary_reason
-   counterfactual_result
-   final_conclusion

### metadata.json

-   model
-   dataset
-   epoch
-   hidden_dim
-   heads
-   prediction_time
-   created_at

## Principle

Compute: - GNN inference - XAI - JSON generation

Viewer: - JSON loading - Visualization only
