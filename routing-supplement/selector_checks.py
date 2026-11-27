"""Check the independently timed raw-window sweep and grouped-query score identity."""
from collections import defaultdict
from pathlib import Path
from statistics import median
import csv
import json

import numpy as np


def check_window(evidence):
    trials = json.loads((evidence / "selector-window-trials.json").read_text())
    with (evidence / "selector-window-medians.csv").open(newline="") as stream:
        medians = list(csv.DictReader(stream))
    by_method = defaultdict(list)
    for trial in trials:
        key = trial["batch"], trial["candidates"], trial["mode"], trial["name"]
        by_method[key].append(trial)
    counts = (48, 96, 192, 384, 768, 1536, 3072)
    assert len(trials) == 280 and len(medians) == len(by_method) == 56
    event_times = {}
    for summary in medians:
        key = int(summary["batch"]), int(summary["candidates"]), summary["mode"], summary["name"]
        sample = by_method[key]
        assert sorted(t["repetition"] for t in sample) == list(range(5))
        for measure in ("wall", "event"):
            value = median(t[measure + "_ms"] for t in sample)
            expected = float(summary["median_" + measure + "_ms"])
            np.testing.assert_allclose(value, expected, atol=1e-10, rtol=0)
        event_times[key] = median(t["event_ms"] for t in sample)
    for n in counts:
        for batch in (1, 8):
            for mode in ("eager", "graph"):
                a = by_method[batch, n, mode, "axis_replayable"]
                f = by_method[batch, n, mode, "fullscan_replayable"]
                assert {(t["repetition"], t["order"]) for t in a+f} == {
                    (rep, position) for rep in range(5) for position in (0, 1)}
    for batch, mode, expected in [(1, "eager", []), (1, "graph", []),
                                  (8, "eager", [3072]), (8, "graph", [1536, 3072])]:
        faster = [n for n in counts if event_times[batch,n,mode,"axis_replayable"]
                  < event_times[batch,n,mode,"fullscan_replayable"]]
        assert faster == expected
    for count, mode, expected in [(1536, "graph", [.178, .249]), (3072, "eager", [.394, .688])]:
        values = [round(event_times[8,count,mode,method], 3)
                  for method in ("axis_replayable", "fullscan_replayable")]
        assert values == expected
    print("All 280 raw-window timing blocks, 56 medians and observed changes in ordering agree.")


def check_group_scores():
    rng = np.random.default_rng(5417)
    q = rng.normal(size=(2, 128))
    k = rng.normal(size=(48, 128))
    per_head = q @ k.T
    group_query = q.mean(axis=0)
    np.testing.assert_allclose(per_head.mean(axis=0), group_query @ k.T, atol=1e-12, rtol=0)
    coords = np.argsort(-np.abs(group_query), kind="stable")[:16]
    projected = k[:, coords] @ group_query[coords]
    shortlist = np.argsort(-projected, kind="stable")[:32]
    grouped = shortlist[np.argsort(-per_head[:,shortlist].mean(axis=0), kind="stable")[:16]]
    same_q = shortlist[np.argsort(-(k[shortlist] @ group_query), kind="stable")[:16]]
    np.testing.assert_array_equal(grouped, same_q)
    assert any(set(grouped) != set(shortlist[np.argsort(-head[shortlist])[:16]]) for head in per_head)
    print("The group-mean query reproduces the mean-head score and its shared top-16 reranking.")

