import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon
from sklearn.manifold import TSNE


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_assignment_sankey(
    conf_counts: np.ndarray,
    categories: Sequence[str],
    out_path: str,
    *,
    min_flow_frac: float = 0.01,
) -> None:
    """
    Render a static alluvial-style Sankey from true (left) -> pred (right).

    conf_counts: confusion matrix with raw counts, shape [K, K], rows=true, cols=pred.
    categories: labels for both axes (assumed same order).
    min_flow_frac: drop tiny ribbons below this fraction of total count.
    """
    K = conf_counts.shape[0]
    total = max(1.0, float(conf_counts.sum()))
    flows = conf_counts.astype(float) / total

    left_heights = flows.sum(axis=1)
    right_heights = flows.sum(axis=0)

    # Vertical positions of stacked bars
    left_pos = np.cumsum(np.r_[0.0, left_heights[:-1]])
    right_pos = np.cumsum(np.r_[0.0, right_heights[:-1]])

    # Figure and bars
    fig, ax = plt.subplots(figsize=(max(8, K * 1.2), 6))
    ax.axis('off')
    color_map = plt.cm.Set3(np.linspace(0, 1, K))

    # Draw left and right stacked bars
    for i in range(K):
        ax.add_patch(Polygon([[0, left_pos[i]], [0, left_pos[i] + left_heights[i]], [0.05, left_pos[i] + left_heights[i]], [0.05, left_pos[i]]],
                             closed=True, color=color_map[i], alpha=0.9, ec='none'))
        ax.text(-0.02, left_pos[i] + left_heights[i] / 2.0, categories[i], va='center', ha='right', fontsize=9)
        ax.add_patch(Polygon([[0.95, right_pos[i]], [0.95, right_pos[i] + right_heights[i]], [1.0, right_pos[i] + right_heights[i]], [1.0, right_pos[i]]],
                             closed=True, color=color_map[i], alpha=0.9, ec='none'))
        ax.text(1.02, right_pos[i] + right_heights[i] / 2.0, categories[i], va='center', ha='left', fontsize=9)

    # Draw ribbons (parallelogram ribbons with gentle curvature via mid points)
    left_offsets = left_pos.copy()
    right_offsets = right_pos.copy()
    for i in range(K):
        for j in range(K):
            f = flows[i, j]
            if f <= 0 or f < min_flow_frac:
                continue
            y0 = left_offsets[i]
            y1 = y0 + f
            y2 = right_offsets[j]
            y3 = y2 + f
            left_offsets[i] = y1
            right_offsets[j] = y3

            # Ribbon polygon with slight bulge in middle
            xL0, xL1 = 0.05, 0.45
            xR0, xR1 = 0.55, 0.95
            pts = [
                (xL0, y0),
                (xL1, y0 + (y1 - y0) * 0.5),
                (xR0, y2 + (y3 - y2) * 0.5),
                (xR1, y2),
                (xR1, y3),
                (xR0, y2 + (y3 - y2) * 0.5),
                (xL1, y0 + (y1 - y0) * 0.5),
                (xL0, y1),
            ]
            ax.add_patch(Polygon(pts, closed=True, color=color_map[i], alpha=0.35, ec='none'))

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(0, 1)
    ax.set_title('Validation Assignment Sankey (True → Pred)')
    _ensure_dir(out_path)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (v or '') for k, v in r.items()})
    return rows


def _write_html(path: str, html: str) -> None:
    _ensure_dir(path)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)


def write_assignment_gallery_html_train_val(csv_path: str, out_html: str) -> None:
    rows = _read_csv_rows(csv_path)
    # Minimal sortable/filterable table via vanilla JS
    html = f"""
<!DOCTYPE html>
<html lang=\"en\">\n<meta charset=\"utf-8\">\n<title>Per-sample Assignments (Train/Val)</title>\n<style>
body {{ font-family: Arial, sans-serif; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 6px; font-size: 12px; }}
th {{ cursor: pointer; background: #f2f2f2; position: sticky; top: 0; }}
.controls {{ margin: 8px 0; }}
.badge {{ padding: 2px 6px; border-radius: 4px; background: #eee; margin-right: 6px; }}
</style>
<div class=\"controls\">
  <span class=\"badge\" id=\"count\"></span>
  <button onclick=\"setFilter('all')\">All</button>
  <button onclick=\"setFilter('most_wrong')\">Most confident wrong</button>
  <button onclick=\"setFilter('least_right')\">Least confident right</button>
  <button onclick=\"setFilter('ambig')\">Ambiguous (small margin)</button>
  <label>Search: <input id=\"search\" oninput=\"applyFilter()\"></label>
</div>
<table id=\"tbl\">\n<thead><tr>
  <th onclick=\"sortTable(0)\">split</th>
  <th onclick=\"sortTable(1)\">index</th>
  <th onclick=\"sortTable(2)\">true_class</th>
  <th onclick=\"sortTable(3)\">pred_class</th>
  <th onclick=\"sortTable(4)\">top1_conf</th>
  <th>top3</th>
  <th>text_snippet</th>
</tr></thead>
<tbody>
"""
    for r in rows:
        html += "<tr>" + "".join(
            [
                f"<td>{r.get('split','')}</td>",
                f"<td>{r.get('index','')}</td>",
                f"<td>{r.get('true_class','')}</td>",
                f"<td>{r.get('pred_class','')}</td>",
                f"<td>{r.get('top1_conf','')}</td>",
                f"<td>{r.get('top3','')}</td>",
                f"<td>{r.get('text_snippet','').replace('<','&lt;')}</td>",
            ]
        ) + "</tr>\n"
    html += """
</tbody></table>
<script>
let filterMode = 'all';
function sortTable(n){
  const table=document.getElementById('tbl');
  let rows=Array.from(table.tBodies[0].rows);
  let asc=table.getAttribute('data-sort')!=='col'+n+':asc';
  rows.sort((a,b)=>{
    let x=a.cells[n].innerText, y=b.cells[n].innerText;
    let nx=parseFloat(x), ny=parseFloat(y);
    if(!isNaN(nx) && !isNaN(ny)){ return asc? nx-ny : ny-nx; }
    return asc? x.localeCompare(y) : y.localeCompare(x);
  });
  rows.forEach(r=>table.tBodies[0].appendChild(r));
  table.setAttribute('data-sort','col'+n+(asc?':asc':':desc'));
}
function marginFromTop3(s){
  // expects format: Class:0.9|Class2:0.07|...
  let parts=s.split('|').map(x=>x.split(':')).map(p=>parseFloat(p[1]||'0'));
  parts=parts.filter(x=>!isNaN(x));
  parts.sort((a,b)=>b-a);
  if(parts.length<2) return 1.0;
  return (parts[0]-parts[1]);
}
function applyFilter(){
  const q=(document.getElementById('search').value||'').toLowerCase();
  const rows=Array.from(document.getElementById('tbl').tBodies[0].rows);
  let shown=0;
  rows.forEach(r=>{
    const trueC=r.cells[2].innerText; const predC=r.cells[3].innerText;
    const conf=parseFloat(r.cells[4].innerText||'0');
    const margin=marginFromTop3(r.cells[5].innerText||'');
    const text=r.cells[6].innerText.toLowerCase();
    let ok=true;
    if(filterMode==='most_wrong') ok = (trueC!==predC);
    if(filterMode==='least_right') ok = (trueC===predC);
    if(filterMode==='ambig') ok = (margin<0.1);
    if(ok && q){ ok = (text.indexOf(q)>=0 || trueC.toLowerCase().includes(q) || predC.toLowerCase().includes(q)); }
    r.style.display = ok? '' : 'none';
    if(ok) shown++;
  });
  document.getElementById('count').innerText = shown+' rows';
}
function setFilter(mode){ filterMode=mode; applyFilter(); }
applyFilter();
</script>
</html>
"""
    _write_html(out_html, html)


def write_assignment_gallery_html_generated(csv_path: str, out_html: str) -> None:
    rows = _read_csv_rows(csv_path)
    html = f"""
<!DOCTYPE html>
<html lang=\"en\">\n<meta charset=\"utf-8\">\n<title>Per-sample Assignments (Generations)</title>\n<style>
body {{ font-family: Arial, sans-serif; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 6px; font-size: 12px; }}
th {{ cursor: pointer; background: #f2f2f2; position: sticky; top: 0; }}
.controls {{ margin: 8px 0; }}
</style>
<div class=\"controls\">
  <label>Search: <input id=\"search\" oninput=\"applyFilter()\"></label>
  <button onclick=\"sortTable(0)\">Sort by index</button>
  <button onclick=\"sortTable(2)\">Sort by pred_class</button>
  <button onclick=\"sortTable(3)\">Sort by conf</button>
  <span id=\"count\"></span>
</div>
<table id=\"tbl\">\n<thead><tr>
  <th>gen_index</th>
  <th>prompt_id</th>
  <th>pred_class</th>
  <th>top1_conf</th>
  <th>top3</th>
  <th>prompt</th>
  <th>text_snippet</th>
</tr></thead><tbody>
"""
    for r in rows:
        html += (
            f"<tr><td>{r.get('gen_index','')}</td>"
            f"<td>{r.get('prompt_id','')}</td>"
            f"<td>{r.get('pred_class','')}</td>"
            f"<td>{r.get('top1_conf','')}</td>"
            f"<td>{r.get('top3','')}</td>"
            f"<td>{(r.get('prompt','') or '').replace('<','&lt;')}</td>"
            f"<td>{(r.get('text_snippet','') or '').replace('<','&lt;')}</td></tr>\n"
        )
    html += """
</tbody></table>
<script>
function sortTable(n){
  const table=document.getElementById('tbl');
  let rows=Array.from(table.tBodies[0].rows);
  const asc=table.getAttribute('data-sort')!=='col'+n+':asc';
  rows.sort((a,b)=>{
    let x=a.cells[n].innerText, y=b.cells[n].innerText;
    let nx=parseFloat(x), ny=parseFloat(y);
    if(!isNaN(nx) && !isNaN(ny)){ return asc? nx-ny : ny-nx; }
    return asc? x.localeCompare(y) : y.localeCompare(x);
  });
  rows.forEach(r=>table.tBodies[0].appendChild(r));
  table.setAttribute('data-sort','col'+n+(asc?':asc':':desc'));
  applyFilter();
}
function applyFilter(){
  const q=(document.getElementById('search').value||'').toLowerCase();
  const rows=Array.from(document.getElementById('tbl').tBodies[0].rows);
  let shown=0;
  rows.forEach(r=>{
    const cellTxt = Array.from(r.cells).map(td=>td.innerText.toLowerCase()).join(' ');
    const ok = !q || cellTxt.includes(q);
    r.style.display = ok? '' : 'none'; if(ok) shown++;
  });
  document.getElementById('count').innerText = ' '+shown+' rows';
}
applyFilter();
</script>
</html>
"""
    _write_html(out_html, html)


def nn_composition_and_diagnostics(
    neighbors_jsonl_path: str,
    *,
    out_dir: str,
    categories: Sequence[str],
    pred_classes: Optional[List[str]] = None,
) -> None:
    """
    Read gen_neighbors.jsonl and compute:
      - Per-sample neighbor class histogram CSV
      - Aggregated composition by predicted class (bar plot)
      - Distance diagnostics (in-class vs out-of-class) per predicted class
      - Hubness (how often each train index appears)
    """
    if not os.path.exists(neighbors_jsonl_path):
        return
    _ensure_dir(os.path.join(out_dir, 'dummy'))  # ensure dir exists

    per_sample: List[Dict] = []
    hub_counts: Dict[int, int] = {}
    with open(neighbors_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            neigh = rec.get('neighbors', [])
            counts: Dict[str, int] = {c: 0 for c in categories}
            for nb in neigh:
                cls = nb.get('true_class')
                if isinstance(cls, str) and cls in counts:
                    counts[cls] += 1
                ti = nb.get('train_index')
                if isinstance(ti, int):
                    hub_counts[ti] = hub_counts.get(ti, 0) + 1
            per_sample.append({
                'gen_index': rec.get('gen_index'),
                'pred_class': rec.get('pred_class'),
                'top1_conf': rec.get('top1_conf'),
                **{f'class_{c}': counts[c] for c in categories}
            })

    # Write per-sample composition CSV
    comp_csv = os.path.join(out_dir, 'nn_composition_per_sample.csv')
    with open(comp_csv, 'w', encoding='utf-8', newline='') as fcsv:
        fieldnames = ['gen_index', 'pred_class', 'top1_conf'] + [f'class_{c}' for c in categories]
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_sample:
            writer.writerow(r)

    # Aggregate composition by predicted class
    by_pred: Dict[str, np.ndarray] = {}
    for r in per_sample:
        pc = r.get('pred_class') or 'Unknown'
        vec = np.array([int(r.get(f'class_{c}', 0)) for c in categories], dtype=float)
        by_pred.setdefault(pc, np.zeros_like(vec))
        by_pred[pc] += vec
    # Normalize
    for k in list(by_pred.keys()):
        s = by_pred[k].sum()
        if s > 0:
            by_pred[k] = by_pred[k] / s

    # Bar plot
    if by_pred:
        plt.figure(figsize=(max(10, len(categories)), 5))
        idx = np.arange(len(categories))
        width = max(0.8 / max(1, len(by_pred)), 0.1)
        for i, (pc, vec) in enumerate(sorted(by_pred.items())):
            plt.bar(idx + i * width, vec, width=width, label=pc)
        plt.xticks(idx + width * (len(by_pred) - 1) / 2.0, categories, rotation=45, ha='right')
        plt.ylabel('Neighbor-class share')
        plt.title('NN class composition aggregated by predicted class')
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'nn_composition_by_pred_class.png'), dpi=200)
        plt.close()

    # Hubness histogram
    if hub_counts:
        vals = np.array(list(hub_counts.values()), dtype=int)
        plt.figure(figsize=(7, 4))
        sns.histplot(vals, bins=30)
        plt.xlabel('#times a train sample appears as NN')
        plt.ylabel('Count of train samples')
        plt.title('Hubness distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'hubness_hist.png'), dpi=200)
        plt.close()
        # Top hubs CSV
        top = sorted(hub_counts.items(), key=lambda kv: kv[1], reverse=True)[:200]
        with open(os.path.join(out_dir, 'hubness_top.csv'), 'w', encoding='utf-8') as f:
            f.write('train_index,count\n')
            for i, c in top:
                f.write(f'{i},{c}\n')


def distance_diagnostics(
    gen_pred_classes: List[int],
    neighbors: Tuple[np.ndarray, np.ndarray],
    train_labels: List[int],
    categories: Sequence[str],
    out_path: str,
) -> None:
    """
    For each generated sample, approximate d_in and d_out using the k-NN set:
      d_in: min distance among neighbors whose true class == predicted
      d_out: min distance among neighbors whose true class != predicted
    Then plot distributions of d_in, d_out and ratio d_out/d_in per predicted class.
    neighbors: (dists, nbr_idxs) as returned by NearestNeighbors.kneighbors
    """
    dists, nbrs = neighbors
    K = len(categories)
    per_class = {j: {'din': [], 'dout': [], 'ratio': []} for j in range(K)}
    for i in range(len(gen_pred_classes)):
        pc = int(gen_pred_classes[i])
        dd = dists[i]
        nb = nbrs[i]
        din = float('inf')
        dout = float('inf')
        for d, ti in zip(dd, nb):
            ti = int(ti)
            if int(train_labels[ti]) == pc:
                if d < din:
                    din = d
            else:
                if d < dout:
                    dout = d
        if not math.isfinite(din) or not math.isfinite(dout):
            continue
        per_class[pc]['din'].append(din)
        per_class[pc]['dout'].append(dout)
        if din > 0:
            per_class[pc]['ratio'].append(dout / din)

    # Plot ratio box/violin per class
    labels = []
    ratios = []
    for j in range(K):
        rs = per_class[j]['ratio']
        if rs:
            labels.append(categories[j])
            ratios.append(rs)
    if ratios:
        plt.figure(figsize=(max(8, len(labels)), 4))
        plt.violinplot(ratios, showmedians=True)
        plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=45, ha='right')
        plt.ylabel('d_out / d_in')
        plt.title('Distance diagnostics by predicted class')
        plt.tight_layout()
        _ensure_dir(out_path)
        plt.savefig(out_path, dpi=200)
        plt.close()


def plot_embeddings_map(
    train_emb: np.ndarray,
    train_labels: List[int],
    val_emb: np.ndarray,
    val_labels: List[int],
    gen_emb: Optional[np.ndarray],
    gen_pred: Optional[List[int]],
    categories: Sequence[str],
    out_path: str,
    *,
    max_points_per_split: int = 1000,
) -> None:
    """
    2D map of embeddings using t-SNE. Colors: train/val by true class, generations by predicted class.
    To keep runtime manageable, subsample each split to <= max_points_per_split.
    """
    rng = np.random.default_rng(0)

    def _subsample(X, y, m):
        if X is None or len(X) == 0:
            return X, y
        n = len(X)
        if n <= m:
            return X, y
        idx = rng.choice(n, size=m, replace=False)
        return X[idx], [y[i] for i in idx]

    Xtr, ytr = _subsample(train_emb, train_labels, max_points_per_split)
    Xva, yva = _subsample(val_emb, val_labels, max_points_per_split)
    if gen_emb is not None and gen_pred is not None and len(gen_emb) > 0:
        Xge, yge = _subsample(gen_emb, gen_pred, max_points_per_split)
    else:
        Xge, yge = None, None

    to_fit = [Xtr, Xva] + ([Xge] if Xge is not None else [])
    Xcat = np.vstack([x for x in to_fit if x is not None])
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=0)
    Z = tsne.fit_transform(Xcat)

    # Split back
    ntr = len(Xtr)
    nva = len(Xva)
    nge = len(Xge) if Xge is not None else 0
    Ztr = Z[:ntr]
    Zva = Z[ntr:ntr + nva]
    Zge = Z[ntr + nva:] if nge > 0 else None

    plt.figure(figsize=(8, 6))
    cmap = plt.cm.get_cmap('tab20', len(categories))
    for j, c in enumerate(categories):
        idx_tr = [i for i, y in enumerate(ytr) if y == j]
        if idx_tr:
            plt.scatter(Ztr[idx_tr, 0], Ztr[idx_tr, 1], s=8, alpha=0.5, label=f'Train:{c}', color=cmap(j))
        idx_va = [i for i, y in enumerate(yva) if y == j]
        if idx_va:
            plt.scatter(Zva[idx_va, 0], Zva[idx_va, 1], s=12, alpha=0.8, marker='^', label=f'Val:{c}', color=cmap(j))
    if Zge is not None:
        for j, c in enumerate(categories):
            idx_ge = [i for i, y in enumerate(yge) if y == j]
            if idx_ge:
                plt.scatter(Zge[idx_ge, 0], Zge[idx_ge, 1], s=20, alpha=0.9, marker='x', label=f'Gen:{c}', color=cmap(j))
    plt.legend(ncol=2, fontsize=8)
    plt.title('Embedding map (t-SNE)')
    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path, dpi=200)
    plt.close()


def class_prototypes_and_medoids(
    texts: List[str],
    labels: List[int],
    emb: np.ndarray,
    categories: Sequence[str],
    out_json: str,
    *,
    max_exact_medoids: int = 1000,
) -> None:
    """
    For each class, compute centroid and choose medoid (min avg distance). For large classes, approximate
    by picking the closest to centroid.
    Save a JSON with per-class: centroid_idx (approx), medoid_idx, medoid_text, exemplars (3 diverse).
    """
    out = {}
    for j, c in enumerate(categories):
        idxs = [i for i, y in enumerate(labels) if y == j]
        if not idxs:
            continue
        E = emb[idxs]
        centroid = E.mean(axis=0, keepdims=True)
        # Prototype: nearest to centroid
        d2 = np.sum((E - centroid) ** 2, axis=1)
        proto_local = int(np.argmin(d2))
        proto_idx = idxs[proto_local]

        # Medoid
        if len(idxs) <= max_exact_medoids:
            # Exact average pairwise distance using cosine distance proxy (1 - cosine)
            norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
            U = E / norms
            sim = U @ U.T
            dist = 1.0 - sim
            avg = dist.mean(axis=1)
            med_local = int(np.argmin(avg))
        else:
            med_local = proto_local
        med_idx = idxs[med_local]

        # Diverse exemplars: farthest-first from centroid
        order = list(np.argsort(-np.linalg.norm(E - centroid, axis=1)))
        exemplars = []
        seen = set()
        for k in order:
            if len(exemplars) >= 3:
                break
            if k in seen:
                continue
            exemplars.append(idxs[k])
            seen.add(k)

        out[c] = {
            'prototype_index': int(proto_idx),
            'prototype_snippet': texts[proto_idx][:200],
            'medoid_index': int(med_idx),
            'medoid_snippet': texts[med_idx][:200],
            'exemplar_indices': [int(i) for i in exemplars],
            'exemplar_snippets': [texts[i][:200] for i in exemplars],
        }

    _ensure_dir(out_json)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

