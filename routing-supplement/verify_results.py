"""Reproduce the camera-ready tables from the paired predictions, on CPU."""
from pathlib import Path
from statistics import median
import csv
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
SOURCE = ROOT / "source"
sys.path.insert(0, str(SOURCE / "stage_ag"))
from analyze_ag import contrast, pairing

LABELS = ("e-neutral", "l-zero-success", "m-zero-s57")
METHODS = ("refine16", "boundary_refill16", "axis16_c32_k16", "fullscan16")

def read_json(path):
    return json.loads(path.read_text())

def read_csv(name):
    with (EVIDENCE / name).open(newline="") as stream:
        return list(csv.DictReader(stream))

def check_digests():
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        with (ROOT / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("Checksum mismatch for " + name)

def quality():
    arrays = {}
    rows = read_csv("quality-table.csv")
    for row in rows:
        label, panel, kind = row["checkpoint"], row["panel"], row["router"]
        with np.load(EVIDENCE / "predictions" / label / (panel + "-" + kind + ".npz"), allow_pickle=False) as data:
            a = dict(data)
        arr = a["correct"]
        assert arr.shape == (int(row["logical_examples"]), 8)
        if panel == "natural-grid":
            expected = (a["predicted_token_ids"] == a["gold_token_ids"][:, None]).all(-1)
            ids = a["row_ids"]
            if kind == "dense":
                assert (arr == arr[:, :1]).all()
        else:
            expected = a["predicted_token_ids"] == a["gold_token_ids"]
            ids = a["logical_ids"]
        assert np.unique(ids).size == len(ids)
        np.testing.assert_array_equal(arr, expected)
        actual = (arr.mean(0).min(), arr.mean(), arr.all(1).mean(), a["nll"].mean())
        for x, key in zip(actual, ("worst_accuracy", "mean_accuracy", "all_phases_correct", "mean_nll")):
            np.testing.assert_allclose(x, float(row[key]), rtol=1e-6, atol=1e-7)
        arrays[label, panel, kind] = a
    assert len(arrays) == 42
    for panel in ("retrieval-8192", "retrieval-2048", "natural-grid"):
        anchor = arrays[LABELS[0],panel,METHODS[0]]
        for (label, name, kind), data in arrays.items():
            if name == panel:
                pairing(anchor, data)
    print("8K worst-source-phase accuracy (%)")
    print("Checkpoint  Refine  Boundary  Axis16  Fullscan")
    result = []
    for label, name in zip(LABELS, ("E","L","S57")):
        result.append([100*arrays[label,"retrieval-8192",k]["correct"].mean(0).min() for k in METHODS])
        print(name, *(f"{value:.2f}" for value in result[-1]))
    print("Mean", *(f"{value:.2f}" for value in np.mean(result,axis=0)))
    def stack(panel, kind, key):
        return np.stack([arrays[label,panel,kind][key] for label in LABELS])
    spec = read_json(EVIDENCE / "primary-intervals.json")
    for row in spec:
        ref = row["reference"]
        computed = contrast(stack("retrieval-8192",METHODS[2],"correct"),stack("retrieval-8192",ref,"correct"),
                            "worst_accuracy",.975,seed=694717,reps=10000)
        for key in ("delta","lower","upper"):
            np.testing.assert_allclose(computed[key],row[key],rtol=0,atol=1e-12)
        print(f"Axis16 minus {ref}, pp: {100*computed['delta']:.2f} [{100*computed['lower']:.2f}, {100*computed['upper']:.2f}]")
    print("512 reused documents: mean accuracy (%), all-grid correctness (%), word NLL")
    for k in METHODS:
        c, loss = stack("natural-grid",k,"correct"),stack("natural-grid",k,"nll")
        print(k, f"{100*c.mean():.2f} {100*c.all(-1).mean():.2f} {loss.mean():.4f}")
    row = arrays["dense-e4800","natural-grid","dense"]
    print("Dense at 4,800 updates:", f"{100*row['correct'][:,0].mean():.2f} (one origin-free score), {row['nll'][:,0].mean():.4f}")
    # All six signed natural-text contrasts use the same documents and bootstrap seed.
    all_rows = read_json(EVIDENCE / "ag-summary.json")["contrasts"]
    for ref in METHODS[:2]:
        for endpoint in ("mean_accuracy","all_variants_correct","mean_nll"):
            key = "nll" if endpoint=="mean_nll" else "correct"
            expected = next(r for r in all_rows if r["panel"]=="natural-grid" and r["method"]==METHODS[2]
                            and r["reference"]==ref and r["checkpoints"]==list(LABELS) and r["endpoint"]==endpoint)
            c = contrast(stack("natural-grid",METHODS[2],key),stack("natural-grid",ref,key),endpoint,.95)
            np.testing.assert_allclose([c[k] for k in ("delta","lower","upper")],
                                       [expected[k] for k in ("delta","lower","upper")],atol=1e-9)
    print("All 42 quality cells, both primary intervals and six descriptive natural contrasts agree.")

def cost_and_distance():
    trials = read_csv("decode-trials.csv")
    assert len(trials)==140
    for row in read_csv("decode-table.csv"):
        chosen=[r for r in trials if all(r[k]==row[k] for k in ("context","batch","kind"))]
        assert len(chosen)==5 and len({r["repetition"] for r in chosen})==5
        for key,out in (("wall_ms_per_step","median_wall_ms_per_step"),("tokens_per_second","median_tokens_per_second")):
            np.testing.assert_allclose(median(float(r[key]) for r in chosen),float(row[out]),atol=1e-10)
    diagnoses=read_csv("diagnostic-table.csv")
    assert len(diagnoses)==66
    for row in diagnoses:
        code="-".join(row[k] for k in ("model","case","condition"))
        with np.load(EVIDENCE / "distance"/(code+".npz"), allow_pickle=False) as a:
            correct=a["predicted_token_ids"]==a["gold_token_ids"]
            np.testing.assert_array_equal(correct,a["correct"])
            np.testing.assert_allclose(correct.mean(),float(row["mean_accuracy"]))
            np.testing.assert_allclose(correct.mean(0).min(),float(row["worst_accuracy"]))
    print("Seven-router medians from 140 trials and 66 distance-diagnostic cells agree.")

if __name__=="__main__":
    check_digests()
    quality()
    cost_and_distance()
    from mechanism_checks import crossed_grids, original_boundary
    crossed_grids()
    original_boundary()
    from selector_checks import check_window, check_group_scores
    check_window(EVIDENCE)
    check_group_scores()

