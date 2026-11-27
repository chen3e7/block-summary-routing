"""Replay a published branch using the archived training functions and local paths."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source"
def load_json(p): return json.loads(p.read_text())
def digest(path):
    with path.open("rb") as stream: return hashlib.file_digest(stream,"sha256").hexdigest()
def source_module(stage, name):
    sys.path.insert(0,str(SOURCE / ("stage_"+stage)))
    return importlib.import_module(name)

def prepare(work, lineage):
    pins = lineage["pins"]
    metadata = work / "provenance.json"
    fields = {k:pins[k] for k in ("model_id","model_revision","dataset_id","dataset_revision","dataset_config")}
    if metadata.exists():
        old = load_json(metadata)
        if any(old.get(k) != v for k,v in fields.items()):
            raise ValueError("The work directory contains a different dataset or model revision.")
    else:
        metadata.write_text(json.dumps(fields,indent=2))
    subprocess.run([sys.executable,str(SOURCE/"prepare.py"),"--tokens","20000000"],check=True)
    verify_inputs(work, lineage)

def verify_inputs(work, lineage):
    if digest(work/"tokens.json") != lineage["token_bank_sha256"]:
        raise ValueError("Synthetic token bank mismatch")
    for split, expected in lineage["natural_stream_sha256"].items():
        if digest(work/"data"/(split+".bin")) != expected: raise ValueError(split+" token stream mismatch")

def settings(strict, tf32):
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS",4)))
    torch.use_deterministic_algorithms(strict)
    torch.backends.cudnn.deterministic=strict
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=tf32

def initial(work, branch, desc):
    label=desc["output"]
    if branch=="E":
        args=desc["command"].copy()
        # The original scheduler guard can stop an otherwise identical run on a slower GPU.
        args[args.index("--max-seconds")+1]="864000"
        subprocess.run([sys.executable,str(SOURCE/"stage_e"/args[0]),*args[1:]],check=True)
        if load_json(work/"results"/label/"summary.json")["completed_steps"] != 1200:
            raise ValueError("Initial E adaptation terminated before update 1200")
    else:
        settings(True,True)
        tr=source_module("l","experiment")
        spec=load_json(SOURCE/"sweeps"/("l-development-scope.json" if branch=="L" else "m-development-scope.json"))
        if branch=="S57": spec=spec["replication_training"]
        model,opt,rng,grid,initial_state=tr.make_initial("zero",desc["seed"],load_json(work/"provenance.json"))
        task=tr.Recall(load_json(work/"tokens.json"),8,2)
        natural=tr.Natural(work/"data/train.bin")
        tag="aux1-cos1200" if branch=="L" else "m-repl-aux1-cos1200"
        summary=tr.run_condition(model,opt,initial_state,task,natural,rng,grid,spec,desc["manifest_sha256"],
                                 desc["seed"],tag,1,1200,1200)
        if summary["full_input_sha256"]!=desc["full_input_sha256"]: raise ValueError("Initial-stage inputs differ")

def continuation(work, stage, desc, branch_info):
    settings(True,False)
    mod=source_module(stage,"experiment_"+stage)
    spec=load_json(SOURCE/"sweeps"/(stage+"-development-scope.json"))
    row=next(r.copy() for r in spec["models"] if r["label"]==branch_info["label"])
    parent=work/row["checkpoint"]
    receipt=load_json(parent.parent/"replay-receipt.json")
    expected_parent=branch_info[{"p":"initial","q":"p","r":"q"}[stage]]["output"]
    if receipt["output"] != expected_parent: raise ValueError("Wrong parent checkpoint")
    row["sha256"]=digest(parent)
    if row["sha256"]!=receipt["file_sha256"]: raise ValueError("Replayed parent changed")
    model=mod.base.loading.load_model(row,load_json(work/"provenance.json"))
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    banks=load_json(work/"tokens.json")
    tasks={"common31":mod.CommonSlotRecall(banks,8,2),"wide63":mod.WideSlotRecall(banks,8,2)}
    if stage=="p":
        recipe=next(r for r in spec["recipes"] if r["label"]=="span_aux1")
        checks=mod.train_run(model,state,row,"local64",recipe,spec,desc["manifest_sha256"],tasks)
    else:
        tasks["long95"]=mod.LongSlotRecall(banks,8,2)
        condition=next(r for r in spec["train_conditions"] if r["label"]=="refine16")
        if stage=="q":
            # The recorded Q saver expects a symbolic link to its output tier.
            target=work/"storage"/desc["output"]; target.mkdir(parents=True,exist_ok=True)
            link=work/"checkpoints"/desc["output"]
            if not link.exists(): link.symlink_to(target,target_is_directory=True)
        checks=mod.train_run(model,state,row,condition,spec,desc["manifest_sha256"],tasks)
    if checks["inputs"]!=desc["full_input_sha256"]: raise ValueError("Continuation inputs differ")

def check_streams(work, stage, desc):
    mod=source_module(stage,"experiment_"+stage)
    banks=load_json(work/"tokens.json"); task=mod.Recall(banks,8,2); natural=mod.Natural(work/"data/train.bin")
    spec=load_json(SOURCE/"sweeps"/(stage+"-development-scope.json"))
    extra=[next(r for r in spec["recipes"] if r["label"]=="span_aux1")] if stage=="p" else []
    for step, expected in desc["input_examples"].items():
        x,y,nb,d=mod.stream(task,natural,desc["seed"],int(step),spec,*extra)
        if mod.base.input_digest(x,y,nb,d)!=expected: raise ValueError("Stream mismatch at "+step)
    print("Matched recorded inputs at updates",", ".join(desc["input_examples"]))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("plan","prepare","train","check-inputs"))
    parser.add_argument("--work",type=Path,default=HERE/"work")
    parser.add_argument("--branch",choices=("E","L","S57"),default="E")
    parser.add_argument("--stage",choices=("initial","p","q","r"),default="initial")
    args=parser.parse_args(); lineage=load_json(HERE/"checkpoint-lineage.json")
    info=lineage["branches"][args.branch]; desc=info[args.stage]
    if args.action=="plan":
        print(json.dumps(info,indent=2)); return
    work=args.work.resolve()
    os.environ["PHASE_PROJECT_ROOT"]=str(work); os.environ["PHASE_RUNTIME"]=str(work)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    if args.action=="check-inputs":
        if args.stage=="initial": parser.error("check-inputs covers the p, q and r continuation streams")
        verify_inputs(work,lineage); check_streams(work,args.stage,desc); return
    work.mkdir(exist_ok=True,parents=True)
    if args.action=="prepare":
        prepare(work,lineage); return
    verify_inputs(work,lineage)
    (work/"results").mkdir(exist_ok=True);(work/"checkpoints").mkdir(exist_ok=True)
    label=desc["output"]; weight_dir=work/"checkpoints"/label
    if (work/"results"/label).exists() or (weight_dir/"last.pt").exists():
        raise FileExistsError("This stage already has results in the specified work directory.")
    if args.stage=="initial": initial(work,args.branch,desc)
    else: continuation(work,args.stage,desc,info)
    record={"output":label,"file_sha256":digest(weight_dir/"last.pt"),"branch":args.branch,"stage":args.stage}
    (weight_dir/"replay-receipt.json").write_text(json.dumps(record,indent=2))
    print(json.dumps(record))

if __name__=="__main__": main()

