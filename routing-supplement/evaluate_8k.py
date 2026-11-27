"""Regenerate the D63 evaluation and score a reconstructed sparse checkpoint."""
from pathlib import Path
import argparse
import json
import os
import sys
from train_branch import HERE, SOURCE, digest, load_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work",type=Path,default=HERE/"work")
    parser.add_argument("--branch",choices=("E","L","S57"),default="E")
    parser.add_argument("--inputs-only",action="store_true")
    args=parser.parse_args(); work=args.work.resolve()
    os.environ.update(PHASE_PROJECT_ROOT=str(work),PHASE_RUNTIME=str(work),CUBLAS_WORKSPACE_CONFIG=":4096:8")
    sys.path.insert(0,str(SOURCE/"stage_ag"))
    import numpy as np
    import torch
    from evaluate_ag import base,configure,setup,WideSlotRecall,evaluate_recall
    info=load_json(HERE/"checkpoint-lineage.json")["branches"][args.branch]
    task=WideSlotRecall(load_json(HERE/"token-bank.json"),8,2)
    if args.inputs_only:
        # Check the key identities, answers and positions on three spaced IDs at all eight origins.
        original=HERE/"evidence"/"predictions"/info["label"]/"retrieval-8192-refine16.npz"
        ids=np.repeat([0,255,511],8); phases=np.tile(np.arange(8),3)
        x,y,pos=task.batch(ids,phases,length=8192,pairs=16,distance=63,split_seed=674717)
        with np.load(original,allow_pickle=False) as a:
            np.testing.assert_array_equal(pos,a["source_positions"][ids,phases])
            np.testing.assert_array_equal(y,a["gold_token_ids"][ids,phases])
            np.testing.assert_array_equal(x[torch.arange(len(pos)),pos],a["source_key_ids"][ids,phases])
            np.testing.assert_array_equal(x[torch.arange(len(pos)),pos+63],y)
        print("24 D63 prompts regenerated from the archived token bank and matched to Table II.")
        return
    setup()
    folder=work/"checkpoints"/info["r"]["output"]
    receipt=load_json(folder/"replay-receipt.json")
    if receipt["branch"]!=args.branch or receipt["stage"]!="r" or digest(folder/"last.pt")!=receipt["file_sha256"]:
        raise ValueError("The local final checkpoint disagrees with its training receipt")
    model=base.loading.load_model(dict(label=info["label"],checkpoint=str(folder/"last.pt"),
                                       sha256=receipt["file_sha256"]),load_json(work/"provenance.json"))
    out=work/"evaluation-8k"/args.branch;out.mkdir(exist_ok=False,parents=True)
    for kind in ("refine16","boundary_refill16","axis16_c32_k16","fullscan16"):
        with configure(model,kind):
            metrics,arrays=evaluate_recall(model,task,examples=512,length=8192,pairs=16,
                                           distance=63,split_seed=674717,batch_size=info["evaluation_batch_8k"])
        np.savez_compressed(out/(kind+".npz"),**arrays)
        (out/(kind+".json")).write_text(json.dumps(metrics,indent=2))
        print(kind,metrics["worst_accuracy"],flush=True)

if __name__=="__main__": main()

