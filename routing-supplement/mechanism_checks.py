"""CPU checks of the crossed-grid and original boundary studies."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
EVIDENCE=ROOT/"evidence"
sys.path.insert(0,str(ROOT/"source/stage_ab"))
sys.path.insert(0,str(ROOT/"source/stage_ag"))
from statistics_ab import decomposition, paired_document_mean
from analyze_ag import contrast, pairing

def crossed_grids():
    legacy=json.loads((EVIDENCE/"earlier/ab-summary.json").read_text())
    values=[];parts=[];deltas=[];deltas_ok=[]
    for label in ("l-zero-success","m-zero-s57"):
        cache={}
        def get(kind,m,r):
            f=EVIDENCE/"crossed-grids"/f"ab-grid-{label}-{kind}-m{m}-r{r}.npz"
            with np.load(f,allow_pickle=False) as data: a=dict(data)
            np.testing.assert_array_equal(a["correct"],(a["predicted_token_ids"]==a["gold_token_ids"]).all(-1))
            if cache:
                pairing(next(iter(cache.values())),a)
            cache[kind,m,r]=a
            return a
        raw=[[get("refine16",m,r) for r in range(8)] for m in range(8)]
        def stack(key):return np.stack([np.stack([cell[key] for cell in row],1) for row in raw],1)
        pred,loss,ok=(stack(k) for k in ("predicted_token_ids","nll","correct"))
        frac=(~(pred==pred[:,:,:1]).all((2,3))).mean(1)
        values.append(frac);parts.append(decomposition(loss))
        d=np.arange(8)
        for kind in ("axis16_c32_k16","fullscan16"):
            panel=[get(kind,m,"none") for m in range(8)]
            if kind=="axis16_c32_k16":
                a=np.stack([p["nll"] for p in panel],1);correct=np.stack([p["correct"] for p in panel],1)
                deltas.append(a.var(1)-loss[:,d,d].var(1))
                deltas_ok.append(correct.all(1).astype(float)-ok[:,d,d].all(1))
    def match(obs,expected):
        for key in ("estimate","interval"):np.testing.assert_allclose(obs[key],expected[key],atol=1e-10)
    change=paired_document_mean(values)
    match(change,legacy["aggregate_factorial"]["conditional_route_prediction_change"])
    for key in ("memory_main_variance","route_main_variance","interaction_variance","total_variance"):
        match(paired_document_mean([p[key] for p in parts]),legacy["aggregate_factorial"][key])
    targets=legacy["aggregate_contrasts"]["axis16_c32_k16"]
    match(paired_document_mean(deltas,level=.975),targets["nll_phase_variance_delta"])
    match(paired_document_mean(deltas_ok,level=.975),targets["all_grid_correct_delta"])
    print(f"Crossed grids: changing the routing origin changes predictions on {100*change['estimate']:.2f}% of documents.")

def original_boundary():
    rows=json.loads((EVIDENCE/"earlier/ad-summary.json").read_text())["aggregate_primary"]
    for row in rows:
        panel=row["panel"]; primary=row["primary_endpoint"]
        by_kind={}
        for k in (row["condition"],row["reference"]):
            arrays=[]
            for label in ("l-zero-success","m-zero-s57"):
                with np.load(EVIDENCE/"original-boundary"/f"ad-{label}-{k}-{panel}.npz",allow_pickle=False) as data:arrays.append(dict(data))
            by_kind[k]=arrays
        for a,b in zip(*by_kind.values()):pairing(a,b)
        for arrays in by_kind.values():pairing(*arrays)
        a,b=(np.stack([x["correct"] for x in arr]) for arr in by_kind.values())
        endpoint="worst_accuracy" if panel=="source" else "all_variants_correct"
        got=contrast(a,b,endpoint,.975,seed=649199,reps=10000)
        np.testing.assert_allclose([got["delta"],got["lower"],got["upper"]],
                                   [row[primary],*row["intervals"][primary]],atol=1e-10)
    print("Reproduced the original two-checkpoint boundary study's paired 97.5% intervals on retrieval and natural text.")

