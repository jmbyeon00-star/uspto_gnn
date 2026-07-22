"""
특허 소송 GNN 시각화 서버.

사용법:
    python app.py           (기본: port 5000)
    python app.py --port 5001
"""
import json, os, argparse
from pathlib import Path
from flask import Flask, request, jsonify, render_template

APP_ROOT = Path(__file__).parent
VIZ_JSON = APP_ROOT / 'data' / 'viz_data.json'

app = Flask(__name__, static_folder='static', template_folder='templates')

# ── 캐시 로드 ─────────────────────────────────────────────────────────────────
print(f'Loading {VIZ_JSON} ...')
if VIZ_JSON.exists():
    with open(VIZ_JSON) as f:
        VIZ = json.load(f)
    nodes_by_id = {n['id']: n for n in VIZ['nodes']}
    edges_by_id = {e['edge_id']: e for e in VIZ['edges']}
    print(f"Loaded {len(VIZ['nodes']):,} nodes, {len(VIZ['edges']):,} edges.")
else:
    VIZ = {'nodes': [], 'edges': [], 'metrics': {}}
    nodes_by_id = {}
    edges_by_id = {}
    print('WARNING: viz_data.json not found. Run generate_viz.py first.')


# ── 유틸 ──────────────────────────────────────────────────────────────────────
def _filter_graph(min_score=0, edge_types=None, n_nodes=300, seed=42):
    import random
    random.seed(seed)

    if edge_types is None:
        edge_types = {'sued', 'cites', 'owns'}

    # 점수 기준으로 노드 필터
    patent_nodes = [
        n for n in VIZ['nodes']
        if n['type'] == 'patent' and n.get('risk_score', 0) >= min_score
    ]
    company_nodes = [n for n in VIZ['nodes'] if n['type'] == 'company']

    # 상위 n_nodes 노드 선택 (patent 우선)
    patent_nodes.sort(key=lambda n: n.get('risk_score', 0), reverse=True)
    selected_patents   = patent_nodes[:n_nodes]
    selected_patent_ids = {n['id'] for n in selected_patents}

    # 연결된 company만
    edges_filtered = [
        e for e in VIZ['edges']
        if e['type'] in edge_types
           and (e['source'] in selected_patent_ids or e['target'] in selected_patent_ids)
    ]
    connected_company_ids = set()
    for e in edges_filtered:
        if e['source'].startswith('c_'): connected_company_ids.add(e['source'])
        if e['target'].startswith('c_'): connected_company_ids.add(e['target'])

    selected_companies = [n for n in company_nodes if n['id'] in connected_company_ids]
    nodes = selected_patents + selected_companies

    return nodes, edges_filtered


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/graph')
def graph_api():
    min_score  = float(request.args.get('min_score', 0))
    edge_types = set(request.args.get('edges', 'sued,cites,owns').split(','))
    n_nodes    = int(request.args.get('nodes', 300))
    seed       = int(request.args.get('seed', 42))

    nodes, edges = _filter_graph(min_score, edge_types, n_nodes, seed)
    return jsonify({
        'nodes'  : nodes,
        'edges'  : edges,
        'metrics': VIZ.get('metrics', {}),
    })


@app.route('/api/node/<node_id>')
def node_api(node_id):
    node = nodes_by_id.get(node_id)
    if not node:
        return jsonify({'error': 'not found'}), 404

    connected_edges = [
        e for e in VIZ['edges']
        if e['source'] == node_id or e['target'] == node_id
    ]

    if node['type'] == 'patent':
        neighbors = {
            e['target'] if e['source'] == node_id else e['source']: e['type']
            for e in connected_edges
        }
        sued_edges   = [e for e in connected_edges if e['type'] == 'sued']
        cite_edges   = [e for e in connected_edges if e['type'] == 'cites']
        own_edges    = [e for e in connected_edges if e['type'] == 'owns']
        top_partners = sorted(sued_edges, key=lambda e: e.get('score') or 0, reverse=True)[:5]
        detail = {
            'id'         : node_id,
            'type'       : 'patent',
            'patent_id'  : node['label'],
            'risk_score' : node.get('risk_score', 0),
            'cite_out'   : node.get('cite_out', 0),
            'cite_in'    : node.get('cite_in',  0),
            'n_sued_connections': len(sued_edges),
            'n_cite_connections': len(cite_edges),
            'n_own_connections' : len(own_edges),
            'top_sued_pairs': [
                {
                    'partner'  : e['target'] if e['source'] == node_id else e['source'],
                    'score'    : e.get('score'),
                    'is_positive': e.get('is_positive', False),
                }
                for e in top_partners
            ],
        }
    else:
        sued_out   = [e for e in connected_edges if e['type'] == 'sues' and e['source'] == node_id]
        owned_pats = [e for e in connected_edges if e['type'] == 'owns']
        detail = {
            'id'        : node_id,
            'type'      : 'company',
            'name'      : node['label'],
            'risk_score': node.get('risk_score', 0),
            'att_ratio' : node.get('att_ratio', 0),
            'n_owned_patents': len(owned_pats),
            'n_suits_filed'  : len(sued_out),
        }

    return jsonify(detail)


@app.route('/api/metrics')
def metrics_api():
    m = VIZ.get('metrics', {})
    sued_edges  = [e for e in VIZ['edges'] if e['type'] == 'sued']
    pos_count   = sum(1 for e in sued_edges if e.get('is_positive'))
    total_count = len(sued_edges)
    lawsuit_rate = pos_count / total_count * 100 if total_count else 0

    return jsonify({
        'metrics': {
            **m,
            'lawsuit_rate'  : round(lawsuit_rate, 2),
            'viz_pos_pairs' : pos_count,
            'viz_pred_pairs': total_count,
        },
        'model_info': {
            'architecture'  : 'Two-stream HeteroSAGE (3-layer)',
            'patent_feat'   : 'FAISS + 18 extra (IPC, cite, sue, SK, IE, ...)',
            'company_feat'  : 'FAISS avg + 6 extra (att_ratio, TCP, LI1/3/5/7)',
            'loss'          : 'Focal Loss (α=0.25, γ=2.0)',
            'negatives'     : 'Hard Negative (IPC-based)',
            'decoder'       : 'Dot product + L2-normalize',
            'test_ap'       : m.get('test_ap'),
            'test_auc'      : m.get('test_auc'),
            'best_val_ap'   : m.get('best_val_ap'),
        },
    })


@app.route('/api/search')
def search_api():
    q     = request.args.get('q', '').lower().strip()
    limit = int(request.args.get('limit', 10))
    if not q:
        return jsonify({'results': []})

    results = [
        {
            'id'         : n['id'],
            'label'      : n['label'],
            'type'       : n['type'],
            'risk_score' : n.get('risk_score', 0),
        }
        for n in VIZ['nodes']
        if q in n['label'].lower() or q in n['id'].lower()
    ][:limit]

    return jsonify({'results': results})


@app.route('/api/top_pairs')
def top_pairs_api():
    limit = int(request.args.get('limit', 20))
    sued  = [e for e in VIZ['edges'] if e['type'] == 'sued' and e.get('score') is not None]
    sued  = sorted(sued, key=lambda e: e['score'], reverse=True)[:limit]
    pairs = [
        {
            'attacker'   : e['source'],
            'defender'   : e['target'],
            'score'      : e['score'],
            'is_positive': e.get('is_positive', False),
        }
        for e in sued
    ]
    return jsonify({'pairs': pairs})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    m = VIZ.get('metrics', {})
    print(f"\n{'='*60}")
    print(f"  Patent Litigation GNN Visualization")
    print(f"{'='*60}")
    print(f"  Nodes      : {len(VIZ['nodes']):,}  (patents + companies)")
    print(f"  Edges      : {len(VIZ['edges']):,}")
    print(f"  Test AP    : {m.get('test_ap', 'N/A')}")
    print(f"  Test AUC   : {m.get('test_auc', 'N/A')}")
    print(f"  Running at : http://0.0.0.0:{args.port}")
    print(f"{'='*60}\n")

    app.run(host='0.0.0.0', port=args.port, debug=True)
