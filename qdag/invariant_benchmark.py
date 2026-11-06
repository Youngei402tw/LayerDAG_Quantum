"""Qubit-label-invariant DAG versus canonical-grid transfer benchmark."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .core import Op
from .schedule_benchmark import CircuitExample, auc_score, canonical_order, critical_labels, dependency_edges, examples_hash
from .structured_transfer import generate_structured

GATES = ("h", "ry", "rz", "cx")


def invariant_node_features(example: CircuitExample) -> np.ndarray:
    rows = []
    for op in example.ops:
        theta = op.params[0] if op.params else 0.0
        rows.append([float(op.gate == g) for g in GATES] + [math.sin(theta), math.cos(theta), len(op.wires) / 2])
    return np.asarray(rows, np.float32)


def role_edges(example: CircuitExample) -> tuple[np.ndarray, np.ndarray]:
    edge, wires = dependency_edges(example.n_qubits, example.ops)
    attrs = []
    for (source, target), wire in zip(edge.T, wires):
        roles = []
        for node in (int(source), int(target)):
            op = example.ops[node]
            if op.gate != "cx": role = 0
            elif op.wires[0] == int(wire): role = 1
            else: role = 2
            roles.extend(float(role == k) for k in range(3))
        attrs.append(roles)
    return edge, np.asarray(attrs, np.float32).reshape(-1, 6)


class InvariantDag(nn.Module):
    def __init__(self, width: int = 64, layers: int = 6):
        super().__init__(); self.node_in = nn.Linear(7, width)
        self.fwd = nn.ModuleList([nn.Linear(width + 6, width) for _ in range(layers)])
        self.bwd = nn.ModuleList([nn.Linear(width + 6, width) for _ in range(layers)])
        self.self_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.out = nn.Linear(width, 1)

    def forward_one(self, example: CircuitExample) -> torch.Tensor:
        device = self.node_in.weight.device
        h = F.gelu(self.node_in(torch.tensor(invariant_node_features(example), device=device)))
        edge_np, attr_np = role_edges(example)
        edge = torch.tensor(edge_np, dtype=torch.long, device=device)
        attr = torch.tensor(attr_np, device=device)
        for fwd, bwd, self_layer, norm in zip(self.fwd, self.bwd, self.self_layers, self.norms):
            af = torch.zeros_like(h); ab = torch.zeros_like(h)
            if edge.shape[1]:
                af.index_add_(0, edge[1], fwd(torch.cat([h[edge[0]], attr], 1)))
                ab.index_add_(0, edge[0], bwd(torch.cat([h[edge[1]], attr], 1)))
            h = norm(h + F.gelu(self_layer(h) + af + ab))
        return self.out(h).squeeze(1)

    def forward(self, examples: list[CircuitExample]) -> list[torch.Tensor]:
        return [self.forward_one(x) for x in examples]


class InvariantGrid(nn.Module):
    def __init__(self, width: int = 96):
        super().__init__(); self.node_in = nn.Linear(7, width)
        self.gru = nn.GRU(width, width // 2, num_layers=2, bidirectional=True, batch_first=True)
        self.out = nn.Linear(width, 1)

    def forward(self, examples: list[CircuitExample]) -> list[torch.Tensor]:
        device = self.node_in.weight.device; canonical=[]; orders=[]
        for example in examples:
            order=canonical_order(example);orders.append(order)
            canonical.append(invariant_node_features(example)[order])
        lengths=[len(x) for x in canonical]; padded=torch.zeros(len(examples),max(lengths),7,device=device)
        for i,x in enumerate(canonical): padded[i,:len(x)]=torch.tensor(x,device=device)
        packed=nn.utils.rnn.pack_padded_sequence(F.gelu(self.node_in(padded)),lengths,batch_first=True,enforce_sorted=False)
        encoded,_=self.gru(packed); unpacked,_=nn.utils.rnn.pad_packed_sequence(encoded,batch_first=True)
        output=[]
        for i,(length,order) in enumerate(zip(lengths,orders)):
            logits=self.out(unpacked[i,:length]).squeeze(1); restored=torch.empty_like(logits)
            restored[torch.tensor(order,device=device)]=logits;output.append(restored)
        return output


class InvariantTransformer(nn.Module):
    def __init__(self, width: int = 64, max_length: int = 192):
        super().__init__(); self.node_in=nn.Linear(7,width);self.position=nn.Embedding(max_length,width)
        block=nn.TransformerEncoderLayer(width,4,dim_feedforward=160,dropout=0.0,activation="gelu",batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(block,2,enable_nested_tensor=False);self.out=nn.Linear(width,1)

    def forward(self, examples: list[CircuitExample]) -> list[torch.Tensor]:
        device=self.node_in.weight.device;orders=[];features=[]
        for example in examples:
            order=canonical_order(example);orders.append(order);features.append(invariant_node_features(example)[order])
        lengths=[len(x) for x in features];maximum=max(lengths)
        padded=torch.zeros(len(examples),maximum,7,device=device);mask=torch.ones(len(examples),maximum,dtype=torch.bool,device=device)
        for i,x in enumerate(features):padded[i,:len(x)]=torch.tensor(x,device=device);mask[i,:len(x)]=False
        encoded=self.encoder(F.gelu(self.node_in(padded))+self.position(torch.arange(maximum,device=device))[None],src_key_padding_mask=mask)
        output=[]
        for i,(length,order) in enumerate(zip(lengths,orders)):
            logits=self.out(encoded[i,:length]).squeeze(1);restored=torch.empty_like(logits);restored[torch.tensor(order,device=device)]=logits;output.append(restored)
        return output


@torch.no_grad()
def evaluate(model: nn.Module, examples: list[CircuitExample]) -> dict:
    model.eval();scores=[];labels=[]
    for start in range(0,len(examples),16):
        batch=examples[start:start+16];scores.extend(x.cpu().numpy() for x in model(batch));labels.extend(x.labels for x in batch)
    logits=np.concatenate(scores);truth=np.concatenate(labels);prob=1/(1+np.exp(-logits));pred=prob>=.5
    tp=np.sum(pred&(truth==1));fp=np.sum(pred&(truth==0));fn=np.sum((~pred)&(truth==1))
    return {"nodes":len(truth),"positive_fraction":float(truth.mean()),"auc":auc_score(truth,prob),
            "f1":float(2*tp/max(1,2*tp+fp+fn)),"accuracy":float(np.mean(pred==truth))}


def train(kind: str, training: list[CircuitExample], validation: list[CircuitExample], seed: int, updates: int):
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed);torch.set_num_threads(2)
    if kind=="dag": model=InvariantDag()
    elif kind=="grid": model=InvariantGrid()
    elif kind=="transformer": model=InvariantTransformer()
    else: raise ValueError(kind)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    pos=sum(float(x.labels.sum()) for x in training);neg=sum(len(x.labels)-float(x.labels.sum()) for x in training)
    weight=torch.tensor(neg/pos);rng=np.random.default_rng(seed);best=-1.;state=None;history=[];started=time.monotonic()
    for update in range(1,updates+1):
        batch=[training[int(i)] for i in rng.integers(len(training),size=16)]
        loss=torch.stack([F.binary_cross_entropy_with_logits(y,torch.tensor(x.labels),pos_weight=weight) for y,x in zip(model(batch),batch)]).mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if update%250==0 or update==updates:
            metrics=evaluate(model,validation);history.append({"update":update,"loss":float(loss.detach()),**metrics})
            if metrics["auc"]>best:best=metrics["auc"];state={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state)
    return model,{"kind":kind,"seed":seed,"updates":updates,"parameters":sum(p.numel() for p in model.parameters()),
                  "train_seconds":time.monotonic()-started,"best_validation_auc":best,"history":history}


def relabel(example: CircuitExample, rng: np.random.Generator) -> CircuitExample:
    permutation=rng.permutation(example.n_qubits)
    ops=[Op(op.gate,tuple(int(permutation[q]) for q in op.wires),op.params) for op in example.ops]
    return CircuitExample(example.n_qubits,ops,critical_labels(example.n_qubits,ops))


def run_development(output: Path, updates: int=2000) -> dict:
    output.mkdir(parents=True,exist_ok=True);families=("qaoa","vqe","arithmetic")
    training=generate_structured(13100,600,(4,7),(32,64),families);validation=generate_structured(13101,128,(4,7),(32,64),families)
    same=generate_structured(13102,128,(4,7),(32,64),families);near=generate_structured(13103,128,(8,12),(72,104),families)
    far=generate_structured(13104,128,(13,20),(96,144),families);rng=np.random.default_rng(13105);renamed=[relabel(x,rng) for x in same]
    rows={};models={}
    for kind in ("dag","grid"):
        models[kind],training_result=train(kind,training,validation,101,updates)
        rows[kind]={"training":training_result,"same":evaluate(models[kind],same),"near_width":evaluate(models[kind],near),
                    "far_width":evaluate(models[kind],far),"renamed":evaluate(models[kind],renamed)}
    with torch.no_grad():
        differences=[]
        for original,changed in zip(same[:32],renamed[:32]):
            differences.append(float(torch.max(torch.abs(models["dag"]([original])[0]-models["dag"]([changed])[0]))))
    passed=rows["dag"]["far_width"]["auc"]>rows["grid"]["far_width"]["auc"] and max(differences)<=1e-6
    result={"protocol":"frozen qubit-label-invariant development","updates":updates,"models":rows,
            "dag_max_qubit_relabel_logit_difference":max(differences),"development_gate_passed":passed}
    (output/"development_result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"metrics":{k:{s:v for s,v in row.items() if s!="training"} for k,row in rows.items()},
                      "dag_max_qubit_relabel_logit_difference":max(differences),"development_gate_passed":passed},indent=2))
    return result


def run_final(output: Path, updates: int=4000) -> dict:
    output.mkdir(parents=True,exist_ok=True);families=("qaoa","vqe","arithmetic")
    training=generate_structured(14100,1000,(4,7),(32,64),families);validation=generate_structured(14101,256,(4,7),(32,64),families)
    same=generate_structured(14102,512,(4,7),(32,64),families);near=generate_structured(14103,512,(8,12),(72,104),families)
    far=generate_structured(14104,512,(13,20),(96,144),families);rng=np.random.default_rng(14105);renamed=[relabel(x,rng) for x in same]
    splits={"training":training,"validation":validation,"same":same,"near_width":near,"far_width":far,"renamed":renamed}
    seed_results=[];invariance=[]
    for seed in (111,112,113,114,115):
        row={"seed":seed,"models":{}}
        for kind in ("dag","grid"):
            model,training_result=train(kind,training,validation,seed,updates)
            row["models"][kind]={"training":training_result,"same":evaluate(model,same),"near_width":evaluate(model,near),
                                        "far_width":evaluate(model,far),"renamed":evaluate(model,renamed)}
            if kind=="dag":
                with torch.no_grad():
                    invariance.extend(float(torch.max(torch.abs(model([a])[0]-model([b])[0]))) for a,b in zip(same[:64],renamed[:64]))
        seed_results.append(row)
    aggregate={}
    for kind in ("dag","grid"):
        aggregate[kind]={}
        for split in ("same","near_width","far_width","renamed"):
            aggregate[kind][split]={}
            for metric in ("auc","f1","accuracy"):
                values=np.asarray([r["models"][kind][split][metric] for r in seed_results])
                aggregate[kind][split][metric+"_mean"]=float(values.mean());aggregate[kind][split][metric+"_sample_std"]=float(values.std(ddof=1));aggregate[kind][split][metric+"_values"]=values.tolist()
    auc_diff=np.asarray([r["models"]["dag"]["far_width"]["auc"]-r["models"]["grid"]["far_width"]["auc"] for r in seed_results])
    f1_diff=aggregate["dag"]["far_width"]["f1_mean"]-aggregate["grid"]["far_width"]["f1_mean"]
    same_relabel_diff=max(abs(aggregate["dag"]["same"][m+"_mean"]-aggregate["dag"]["renamed"][m+"_mean"]) for m in ("auc","f1","accuracy"))
    result={"protocol":"frozen fresh five-seed qubit-label-invariant final","updates_per_model":updates,"seeds":[111,112,113,114,115],
            "dataset_hashes":{k:examples_hash(v) for k,v in splits.items()},"seed_results":seed_results,"aggregate":aggregate,
            "far_width_auc_paired_differences":auc_diff.tolist(),"far_width_auc_mean_difference":float(auc_diff.mean()),
            "far_width_f1_mean_difference":float(f1_diff),"dag_max_qubit_relabel_logit_difference":max(invariance),
            "dag_max_same_vs_relabel_metric_difference":float(same_relabel_diff),
            "positive_claim_passed":bool(np.all(auc_diff>0) and auc_diff.mean()>=.03 and f1_diff>0 and max(invariance)<=1e-6 and same_relabel_diff<=1e-12)}
    (output/"final_result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"aggregate":aggregate,"far_width_auc_paired_differences":auc_diff.tolist(),"far_width_auc_mean_difference":float(auc_diff.mean()),
                      "far_width_f1_mean_difference":float(f1_diff),"dag_max_qubit_relabel_logit_difference":max(invariance),"positive_claim_passed":result["positive_claim_passed"]},indent=2))
    return result


def run_transformer_control(output: Path, updates: int=4000) -> dict:
    output.mkdir(parents=True,exist_ok=True);families=("qaoa","vqe","arithmetic")
    training=generate_structured(14100,1000,(4,7),(32,64),families);validation=generate_structured(14101,256,(4,7),(32,64),families)
    same=generate_structured(14102,512,(4,7),(32,64),families);near=generate_structured(14103,512,(8,12),(72,104),families)
    far=generate_structured(14104,512,(13,20),(96,144),families);rng=np.random.default_rng(14105);renamed=[relabel(x,rng) for x in same]
    rows=[]
    for seed in (111,112,113,114,115):
        model,training_result=train("transformer",training,validation,seed,updates)
        rows.append({"seed":seed,"training":training_result,"same":evaluate(model,same),"near_width":evaluate(model,near),"far_width":evaluate(model,far),"renamed":evaluate(model,renamed)})
    aggregate={}
    for split in ("same","near_width","far_width","renamed"):
        aggregate[split]={}
        for metric in ("auc","f1","accuracy"):
            values=np.asarray([r[split][metric] for r in rows]);aggregate[split][metric+"_mean"]=float(values.mean());aggregate[split][metric+"_sample_std"]=float(values.std(ddof=1));aggregate[split][metric+"_values"]=values.tolist()
    result={"protocol":"fixed post-confirmatory invariant-feature canonical Transformer control","updates_per_model":updates,"seeds":[111,112,113,114,115],"seed_results":rows,"aggregate":aggregate,
            "status":"descriptive_control_not_part_of_confirmatory_gate"}
    (output/"control_result.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(aggregate,indent=2));return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=Path("experiments/qai_invariant_development_20261102"));parser.add_argument("--updates",type=int);parser.add_argument("--final",action="store_true");parser.add_argument("--transformer-control",action="store_true")
    args=parser.parse_args()
    if args.transformer_control: run_transformer_control(args.output,args.updates or 4000)
    elif args.final: run_final(args.output,args.updates or 4000)
    else: run_development(args.output,args.updates or 2000)


if __name__=="__main__":main()
