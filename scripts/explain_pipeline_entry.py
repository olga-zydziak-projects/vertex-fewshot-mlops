#!/usr/bin/env python3
"""Explainability entry point - case-based reasoning report for ProtoNet.

Renders an HTML report that explains HOW the model decides, on concrete
examples. ProtoNet's decision IS a distance comparison ("closest prototype
wins"), so the explanation shows the actual mechanism, not a post-hoc proxy:

  - "How to read this" primer (3 sentences on the mechanism)
  - three case studies picked from real test episodes:
      * most CONFIDENT correct decision (largest margin)
      * most BORDERLINE decision (smallest margin)
      * a MISCLASSIFICATION, if one occurs in the sample (most instructive)
    each shown as: query image | distance bars to all prototypes | the support
    images of the chosen class ("it's class A because it's closest to THESE")
  - a 2D PCA map of one episode's embedding space (prototypes + queries)
  - a histogram of decision margins across many episodes (how often the model
    decides confidently vs by a whisker)

Everything is dependency-free beyond what's already in the image: images via
PIL (torchvision dependency), plots as hand-rolled SVG, PCA via torch.svd.

NOTE on episode_breakdown(): fsl's fast_adapt returns only loss/acc, so this
wrapper re-implements the episode layout (label sort + support selection) to
get at raw embeddings/distances/images. It MUST match
fsl.models.protonet.fast_adapt's convention - if that ever changes, change
this too (or better: move episode_breakdown into fsl).
"""
import argparse
import base64
import io
import json
import tempfile
from pathlib import Path


def _write(path: str, value) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(value))


def tensor_to_png_b64(t, scale=3):
    import numpy as np
    from PIL import Image
    arr = (t.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").resize((28 * scale, 28 * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def img_tag(b64, size=84, border=""):
    style = f"width:{size}px;height:{size}px;image-rendering:pixelated;border-radius:6px;{border}"
    return f'<img src="data:image/png;base64,{b64}" style="{style}"/>'


def svg_distance_bars(distances, labels, chosen_idx, true_idx=None, width=420):
    n = len(distances)
    row_h, pad_l, pad_t = 30, 90, 8
    height = pad_t * 2 + n * row_h
    max_d = max(distances) or 1.0
    bar_max_w = width - pad_l - 70
    rows = []
    for i, (d, lab) in enumerate(zip(distances, labels)):
        y = pad_t + i * row_h
        w = (d / max_d) * bar_max_w
        fill = "#2a7d4f" if i == chosen_idx else "#b8c4d8"
        stroke = (' stroke="#e67e22" stroke-width="2.5"'
                  if (true_idx is not None and i == true_idx and i != chosen_idx) else "")
        rows.append(
            f'<text x="{pad_l-8}" y="{y+15}" font-size="12" text-anchor="end" fill="#333">{lab}</text>'
            f'<rect x="{pad_l}" y="{y+3}" width="{w:.1f}" height="16" fill="{fill}"{stroke} rx="3"/>'
            f'<text x="{pad_l+w+6:.1f}" y="{y+15}" font-size="11" fill="#555">{d:.2f}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
            + "".join(rows) + '</svg>')


def svg_pca_scatter(proto_xy, query_xy, query_labels, n_way, width=560, height=420):
    import numpy as np
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
              "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
    allxy = np.vstack([proto_xy, query_xy])
    lo, hi = allxy.min(0) - 0.5, allxy.max(0) + 0.5
    pad = 40

    def px(p):
        x = pad + (p[0] - lo[0]) / (hi[0] - lo[0] or 1) * (width - 2 * pad)
        y = pad + (1 - (p[1] - lo[1]) / (hi[1] - lo[1] or 1)) * (height - 2 * pad)
        return x, y

    parts = []
    for i, q in enumerate(query_xy):
        x, y = px(q)
        c = colors[int(query_labels[i]) % len(colors)]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{c}" opacity="0.55"/>')
    for k in range(n_way):
        x, y = px(proto_xy[k])
        c = colors[k % len(colors)]
        parts.append(f'<rect x="{x-8:.1f}" y="{y-8:.1f}" width="16" height="16" fill="{c}" '
                     f'stroke="#222" stroke-width="1.5" transform="rotate(45 {x:.1f} {y:.1f})"/>')
        parts.append(f'<text x="{x+12:.1f}" y="{y-10:.1f}" font-size="12" font-weight="600" '
                     f'fill="{c}">class {chr(65+k)}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
            f'<rect width="{width}" height="{height}" fill="#fafafa" rx="8"/>'
            + "".join(parts) + '</svg>')


def svg_histogram(values, bins=25, width=640, height=280, color="#4C72B0", title=""):
    import numpy as np
    counts, edges = np.histogram(values, bins=bins)
    max_c = counts.max() if counts.max() > 0 else 1
    pad_l, pad_t = 50, 30
    plot_w, plot_h = width - pad_l - 20, height - 70
    lo, hi = float(edges[0]), float(edges[-1])
    span = (hi - lo) or 1.0
    bars, ticks = [], []
    for i, c in enumerate(counts):
        x0 = pad_l + (edges[i] - lo) / span * plot_w
        x1 = pad_l + (edges[i+1] - lo) / span * plot_w
        y = pad_t + plot_h - (c / max_c) * plot_h
        bars.append(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(x1-x0-1,1):.1f}" '
                    f'height="{pad_t+plot_h-y:.1f}" fill="{color}"/>')
    for v in [lo, (lo + hi) / 2, hi]:
        x = pad_l + (v - lo) / span * plot_w
        ticks.append(f'<text x="{x:.1f}" y="{height-12}" font-size="11" '
                     f'text-anchor="middle" fill="#555">{v:.2f}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
            f'<text x="{width/2}" y="18" font-size="13" text-anchor="middle" fill="#333">{title}</text>'
            f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{width-20}" y2="{pad_t+plot_h}" stroke="#999"/>'
            + "".join(bars) + "".join(ticks) + '</svg>')


def episode_breakdown(model, batch, n_way, k_shot, query, device):
    """Raw view of one episode. MUST match fsl.models.protonet.fast_adapt's
    layout convention (label sort + arange(ways)*(shot+q) support selection)."""
    import numpy as np
    import torch
    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    srt = torch.sort(labels)
    data, labels = data[srt.indices], labels[srt.indices]
    with torch.no_grad():
        emb = model(data)
    sm = np.zeros(data.size(0), dtype=bool)
    sel = np.arange(n_way) * (k_shot + query)
    for off in range(k_shot):
        sm[sel + off] = True
    smt = torch.from_numpy(sm).to(device)
    qmt = torch.from_numpy(~sm).to(device)
    protos = emb[smt].reshape(n_way, k_shot, -1).mean(dim=1)
    support_imgs = data[smt].reshape(n_way, k_shot, *data.shape[1:])
    q_emb, q_imgs = emb[qmt], data[qmt]
    q_labels = labels[qmt].long()
    dists = torch.cdist(q_emb, protos)          # [n_query_total, n_way]
    preds = dists.argmin(dim=1)
    sorted_d, _ = dists.sort(dim=1)
    margins = (sorted_d[:, 1] - sorted_d[:, 0])  # margines pewnosci
    return dict(protos=protos, support_imgs=support_imgs, q_emb=q_emb,
                q_imgs=q_imgs, q_labels=q_labels, dists=dists,
                preds=preds, margins=margins)


def case_study_html(title, subtitle, q_img_b64, dist_svg, support_b64s, chosen, true=None):
    support_html = "".join(img_tag(b, 64) for b in support_b64s)
    verdict = (f'predicted <b>class {chr(65+chosen)}</b>'
               + (f' - <span style="color:#c0392b">true class was {chr(65+true)}</span>'
                  if true is not None and true != chosen else
                  ' <span style="color:#2a7d4f">(correct)</span>'))
    return f"""<div style="border:1px solid #e0e0e0;border-radius:10px;padding:18px;margin:18px 0">
<h3 style="margin:0 0 4px 0">{title}</h3>
<p style="margin:0 0 12px 0;font-size:13px;color:#666">{subtitle}</p>
<table style="border:none"><tr>
<td style="border:none;vertical-align:top;padding-right:22px">
  <div style="font-size:12px;color:#666;margin-bottom:6px">query</div>{q_img_b64}
  <div style="font-size:13px;margin-top:8px">{verdict}</div></td>
<td style="border:none;vertical-align:top;padding-right:22px">
  <div style="font-size:12px;color:#666;margin-bottom:6px">distance to each prototype (shorter = closer = wins)</div>
  {dist_svg}</td>
<td style="border:none;vertical-align:top">
  <div style="font-size:12px;color:#666;margin-bottom:6px">the support examples it matched (class {chr(65+chosen)}'s prototype = their average)</div>
  {support_html}</td>
</tr></table></div>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--dataset", default="omniglot")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-version", default="?")
    ap.add_argument("--n-episodes", type=int, default=60)
    ap.add_argument("--report-html-path", required=True)
    ap.add_argument("--summary-output-path", required=True)
    args = ap.parse_args()

    import numpy as np
    import torch
    from google.cloud import storage

    from fsl.config import TrainConfig
    from fsl.data.omniglot import build_task_samplers, load_frozen_omniglot
    from fsl.models.protonet import Conv4

    # load the saved artifact
    bucket_name, _, prefix = args.model_dir[len("gs://"):].partition("/")
    client = storage.Client(project=args.project)
    gcs_bucket = client.bucket(bucket_name)
    local = Path(tempfile.mkdtemp())
    for name in ("model.pt", "architecture.json"):
        gcs_bucket.blob(f"{prefix}/{name}").download_to_filename(str(local / name))
    arch = json.loads((local / "architecture.json").read_text())
    model = Conv4(in_c=arch["in_channels"], hid=arch["embedding_hid"])
    model.load_state_dict(torch.load(local / "model.pt", map_location="cpu"))
    model.eval()
    device = torch.device("cpu")

    n_way, k_shot, q = arch["n_way"], arch["k_shot"], 15
    cfg = TrainConfig(project=args.project, region=args.region, bucket=args.bucket,
                      dataset=args.dataset, seed=arch["seed"], n_way=n_way,
                      k_shot=k_shot, embedding_hid=arch["embedding_hid"],
                      log_to_vertex=False)
    dataset, _ = load_frozen_omniglot(cfg)
    _, _, test_tasks = build_task_samplers(dataset, cfg)

    # scan episodes; collect margins + candidates for the case studies
    all_margins = []
    best_conf = None   # (margin, ep, qi) - largest margin, correct
    worst_marg = None  # smallest margin
    misclass = None    # first misclassification found
    kept_ep = None
    for e in range(args.n_episodes):
        ep = episode_breakdown(model, test_tasks.sample(), n_way, k_shot, q, device)
        all_margins.extend(ep["margins"].tolist())
        correct = ep["preds"] == ep["q_labels"]
        for qi in range(len(ep["preds"])):
            m = float(ep["margins"][qi])
            if correct[qi] and (best_conf is None or m > best_conf[0]):
                best_conf = (m, ep, qi)
            if worst_marg is None or m < worst_marg[0]:
                worst_marg = (m, ep, qi)
            if not correct[qi] and misclass is None:
                misclass = (m, ep, qi)
        if kept_ep is None:
            kept_ep = ep

    def render_case(title, sub, tup):
        m, ep, qi = tup
        chosen = int(ep["preds"][qi])
        true = int(ep["q_labels"][qi])
        dist_svg = svg_distance_bars(
            [float(d) for d in ep["dists"][qi]],
            [f"class {chr(65+i)}" for i in range(n_way)],
            chosen_idx=chosen, true_idx=(true if true != chosen else None))
        q_b64 = img_tag(tensor_to_png_b64(ep["q_imgs"][qi]))
        sup = [tensor_to_png_b64(ep["support_imgs"][chosen][s]) for s in range(k_shot)]
        return case_study_html(title, sub + f" (margin {m:.2f})", q_b64, dist_svg,
                               sup, chosen, true=(true if true != chosen else None))

    cases = render_case("Most confident decision",
                        "largest gap between best and second-best prototype", best_conf)
    cases += render_case("Most borderline decision",
                         "smallest gap - the model almost chose differently", worst_marg)
    if misclass is not None:
        cases += render_case("A misclassification",
                             "orange outline = the TRUE class's bar; the model picked a closer prototype",
                             misclass)
    else:
        cases += ('<div style="border:1px dashed #bbb;border-radius:10px;padding:14px;margin:18px 0;'
                  'font-size:13px;color:#666">No misclassification occurred in the sampled '
                  f'{args.n_episodes} episodes - nothing to show here (a good sign).</div>')

    # PCA map of one episode
    ep = kept_ep
    both = torch.cat([ep["protos"], ep["q_emb"]])
    X = both - both.mean(dim=0, keepdim=True)
    _, _, V = torch.svd(X)
    xy = (X @ V[:, :2]).numpy()
    pca_svg = svg_pca_scatter(xy[:n_way], xy[n_way:], ep["q_labels"].numpy(), n_way)

    margins = np.array(all_margins)
    hist = svg_histogram(margins, title=f"Decision margins across {len(margins)} query decisions "
                                        f"({args.n_episodes} episodes)")
    pct_tight = float((margins < 0.5).mean()) * 100

    html = f"""<!DOCTYPE html><html><head><style>
body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:24px;color:#1a1a1a;max-width:980px}}
h2{{margin-bottom:2px}} .meta{{font-size:12px;color:#888;margin-bottom:16px}}
.primer{{background:#f2f6fb;border-radius:10px;padding:14px 18px;font-size:14px;line-height:1.5}}
td,th{{font-size:13px}}</style></head><body>
<h2>Explainability Report - model version {args.model_version}</h2>
<p class="meta">artifact: {args.model_dir} | seed {arch["seed"]} | {n_way}-way {k_shot}-shot |
{args.n_episodes} test episodes sampled</p>
<div class="primer"><b>How to read this.</b> This model classifies by <b>distance</b>: each class's
few support examples are averaged into a <i>prototype</i>, and a query is assigned to the class whose
prototype is nearest in embedding space. So every decision has a built-in explanation: <i>"it's
class A because it is closest to these specific examples"</i>. The bars below are those distances -
the shortest bar wins. The <b>margin</b> (gap between the best and second-best) is how confidently
the decision was made.</div>
{cases}
<h3>The embedding space of one episode</h3>
<p style="font-size:13px;color:#666">Diamonds = class prototypes, dots = queries (colored by true
class). Well-separated clusters = easy episode; overlapping colors = where mistakes happen.
2D PCA projection - illustrative, not exact.</p>
{pca_svg}
<h3>How confident are the decisions overall?</h3>
{hist}
<p style="font-size:13px;color:#666">{pct_tight:.1f}% of decisions had a margin below 0.5
(decided "by a whisker"). A left-heavy histogram means many near-ties; right-heavy means
comfortable decisions.</p>
<p class="meta">Honest scope: this explains the <b>decision</b> (distances to prototypes - the
actual mechanism, auditable). Why the encoder considers two images similar remains inside the
network - decision-level explainability, not full representation-level.</p>
</body></html>"""
    _write(args.report_html_path, html)
    summary = (f"cases: confident(m={best_conf[0]:.2f}), borderline(m={worst_marg[0]:.2f}), "
               f"misclass={'yes' if misclass else 'no'}; tight-margin={pct_tight:.1f}%")
    _write(args.summary_output_path, summary)
    print("Explainability report written.", summary)


if __name__ == "__main__":
    main()
