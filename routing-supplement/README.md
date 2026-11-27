# Routing by Block Summaries Can Miss High-Scoring Keys in Compressed Sparse Attention

Supplementary artifact accompanying the camera-ready manuscript by [Jin Kang](https://orcid.org/0009-0003-5901-4546) and [Lin Tian](https://orcid.org/0009-0002-2067-3600).

## Reproduce the reported numbers on a CPU

Unzip the archive, open the `routing-supplement` directory and run the following with Python 3.11 or newer and NumPy installed.

```sh
python verify_results.py
```

The script verifies file checksums, reconstructs correctness from predicted and gold token IDs, and reproduces the three-checkpoint 8K table and its two paired intervals. Expected differences in percentage points are +3.06 [2.08, 4.10] against Refine and +4.95 [3.84, 6.32] against Boundary. It also checks the absolute results for 42 quality conditions, six descriptive natural-text contrasts, 140 timing trials and 66 distance-diagnostic conditions. It reconstructs the crossed-grid loss and prediction surfaces, the two original boundary-study endpoints and the selector timing for seven raw-window sizes. It checks the average-query identity in Section IV against per-head scores on shared keys. The CPU procedure needs the files in this archive and NumPy. Model weights and a network connection are unnecessary for this calculation.

Each bootstrap draw samples 512 logical IDs with replacement, shares the draw across all routers and checkpoints, and carries all eight variants of each ID together. Within each checkpoint the calculation takes the minimum over source-phase accuracies before averaging the three checkpoints. The two primary 8K intervals each use 97.5%, 10,000 draws and seed 694717. The natural reevaluation reuses the boundary study's original 512 documents. Its 95% intervals are descriptive. For the original L/S57 primary results, see `evidence/earlier/ad-summary.json` and the 16 files in `evidence/original-boundary`.

| Manuscript evidence | Location in this archive |
|---|---|
| Expanded 8K retrieval and dense comparison, Table II and Section VI-C | `evidence/quality-table.csv`, `evidence/predictions/`, `evidence/diagnostic-table.csv`, `evidence/distance/` |
| Two earlier retrieval studies, Table III and Fig. 3 | `evidence/earlier/x-summary.json` and `aa-summary.json` in the same directory |
| Independent natural transfer, crossed grids and candidate controls, Tables IV–V and Figs. 4–5 | `evidence/earlier/z-summary.json`, `ab-summary.json` and `ac-summary.json` |
| Boundary control, Table VI and the native dense natural reference | `evidence/earlier/ad-summary.json`, `evidence/natural-reporting-audit.json`, `evidence/quality-table.csv` |
| L40S latency and throughput, Table VII | `evidence/decode-trials.csv`, `evidence/decode-table.csv`, `evidence/selector-timing.json`, `evidence/refill-profiling.json` and `evidence/incremental-fidelity.json` |
| Independent RTX PRO 6000 breakdown for Axis16 and Fullscan | `evidence/axis-profiling.csv` |
| Raw-window sweep in Section VI-H | `evidence/selector-window-trials.json`, `evidence/selector-window-medians.csv` and the table below |

For the natural rows, mean accuracy averages over all grid-document pairs and all-grid correctness requires exact prediction on all eight grids for a document. Word NLL sums the negative log probabilities of the target subtokens with teacher forcing. Dense attention evaluates each document once. Identical dense grid columns in the stored arrays accommodate the shared analysis code. The archive identifies the document cohorts by row ID in `evidence/natural-cohorts.json`. Original source text and pretrained or adapted model weights are obtained or generated separately.

## Recreate the adapted checkpoints

The three branches start from Qwen/Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`. The training targets are the generator's gold answers and the next tokens of natural text. The auxiliary stream uses the sample-10BT configuration of HuggingFaceFW/fineweb-edu at revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. The LAMBADA OpenAI source revision is `900124bf3b8235c6daf21033af9948b3f07346c4`. Full revisions, source and token hashes, branch parents, input hashes, final weight-file and model-tensor hashes are in `checkpoint-lineage.json`. The archived selection and scoring code is under `source/`.

| Updates | E | L | S57 | Schedule and attention |
|---|---|---|---|---|
| 1–1,200 | seed 54, learns gate offsets | seed 55, zero offsets | seed 57, zero offsets | Width/stride 8, local 16, layer origin = layer index modulo 8. Learning-rate base 10⁻⁴. |
| 1,201–2,400 (p) | 71 | 72 | 73 | Compressed summaries and local 64. Rate 2 × 10⁻⁵. |
| 2,401–3,600 (q) | 81 | 82 | 83 | Refine, local 16, raw horizon 64. Rate 5 × 10⁻⁶. |
| 3,601–4,800 (r) | 91 | 92 | 93 | Refine at the same budget and rate. Final checkpoint used in the paper. |

The affine gate multiplies the shared content score by one plus a centered offset scale correction and adds a centered offset bias. E's two offset corrections are cleared at the start of p. All three branches freeze them at zero in p, q and r. Remaining parameters, including backbone projections, receive gradients. Stage one uses a 100-step linear warmup multiplier times a cosine that falls from 1 to 0.1 over 1,200 steps. Each continuation uses 100 steps of linear warmup, then a constant rate. AdamW uses betas (0.9, 0.95), decay 0.1 and global gradient clipping at 1. Each of the four stages starts with a new optimizer. Each complete four-stage trajectory processes 20,889,600 forward tokens in the training objectives (5,222,400 per stage).

Each update uses four length-1,024 sequences with 16 key/value pairs and full-vocabulary final-token cross-entropy. Every fourth update adds a next-token objective on one further length-1,024 FineWeb-Edu sample, weighted by one. The initial distance mixture is {1, 3, 7, 15}. Later stages use {1, 3, 7, 15, 23, 27}. The generators and sampling streams are in the archived source. Training never uses D63 or an 8K sequence. The dense reference starts natively dense at seed 54 and uses the same four-stage budget as E. Its three continuation streams match E at each of the 3,600 updates. `source/stage_ag/train_dense_ag2.py` records the reference implementation and its input checks.

The following commands use a Linux workstation with a CUDA-enabled PyTorch environment. The recorded environment uses Python 3.12, PyTorch 2.9.1+cu128, Transformers 4.57.1, Datasets 3.6.0 and the package versions in `environment.txt`. The first command downloads the pinned base model and prepares the token streams in a new work directory. Prepare that directory once for all three branches. Each stage writes a final checkpoint. The work directory must have space for these files and the base model cache.

```sh
python train_branch.py plan --branch E
python train_branch.py prepare --work ./replay
python train_branch.py check-inputs --work ./replay --branch E --stage p
for branch in E L S57; do
  for stage in initial p q r; do
    python train_branch.py train --work ./replay --branch "$branch" --stage "$stage" || exit 1
  done
done
```

The wrapper calls the archived training functions for the twelve stages used in the manuscript. Replayed parents are checked against their own file receipts. Original file hashes identify the recorded weights. Device-dependent arithmetic and metadata can change file hashes across installations. All nine continuation runs check their full 1,200-step input-stream digest against the historical run. The raw weight files are approximately 3.0 GB each and are excluded from this submission supplement.

To regenerate a sample of the 8K prompts with the model software installed, run the input check below on a CPU. To score the complete 512-example, eight-phase panel on a GPU after reconstructing a checkpoint, run the second command. Use a fresh evaluation directory for each branch.

```sh
python evaluate_8k.py --inputs-only
python evaluate_8k.py --work ./replay --branch E
```

## Code and interpretation

The source files were taken from the frozen experiment snapshot. The original functions for Refine and Refill are in `source/stage_t/routing.py`, Boundary in `source/stage_ad/routing_ad.py`, and Axis16 in `source/stage_u/sketch.py` with its parent selector. The incremental caches and seven-router timing harness are in `source/stage_ae/` and `source/stage_ah/`. The standalone entry points above resolve paths relative to the unpacked supplement. The historical scripts preserve original experiment manifests and cluster-specific execution checks for inspection.

The expanded retrieval cohort uses generation seed 674717. The separate distance diagnosis uses seed 724717 and 32 fresh logical examples per condition. The matched dense run provides a full-history quality reference, with its distance profile shown in Section VI-C. Table VII reports synchronized eager decoding cost with prefill excluded. The paired BF16 fidelity checks accompany the timing data. Figures and tables identify which predictions came from full-sequence evaluation.



## Raw-window timing and group-shared routing

The window sweep uses one NVIDIA RTX PRO 6000 Blackwell Server Edition. Its inputs are seeded synthetic BF16 queries and keys, reused across repetitions. The key count excludes the 16 local positions. Both selectors use eight KV groups, two query heads per group, a head dimension of 128 and 16 final links. Axis16 uses 16 coordinates and a 32-key shortlist. The table gives the ratio of median CUDA-event times per standalone selector call, Axis16 divided by Fullscan. Values below one favor Axis16. The benchmark randomizes method order within five paired timing blocks per condition. Eager blocks average 50 calls. CUDA Graph blocks average 100 effective calls. The trial file also includes per-call wall times. Model-level decoding in Table VII uses the trained 64-position horizon with 48 extra keys.

| Nonlocal keys | Batch 1, eager | Batch 1, graph | Batch 8, eager | Batch 8, graph |
|---:|---:|---:|---:|---:|
| 48 | 2.776 | 2.539 | 2.772 | 2.477 |
| 96 | 2.733 | 2.534 | 2.759 | 2.281 |
| 192 | 2.775 | 2.615 | 2.772 | 2.130 |
| 384 | 2.734 | 2.476 | 2.783 | 1.643 |
| 768 | 2.753 | 2.229 | 2.763 | 1.221 |
| 1536 | 2.754 | 1.808 | 1.543 | 0.713 |
| 3072 | 2.774 | 1.443 | 0.572 | 0.421 |

The first sampled candidate counts favoring Axis16 at batch eight are 3,072 in eager mode and 1,536 in graph replay. At batch one, Fullscan is faster throughout the sweep. The timings characterize standalone selection on warm synthetic inputs. The quality benchmarks retain the 64-position raw horizon.

For Eqs. (2) and (3), both heads in a KV group read the same key for each token. Averaging their unnormalized dot products gives the dot product with the group-mean query. Selection uses one list of 32 and one final set of 16 for the group. Attention then computes the two heads' weights separately. The frozen source implements the projection in source/stage_u/sketch.py (feature_scores), reranking in source/stage_t/routing.py (select), and score sharing in source/stage_r/sparse_r.py (shared_scores).
