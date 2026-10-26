"""Frozen small CPU baseline protocol. Run: python -m qdag.pilot"""
import json, time, hashlib, platform
from pathlib import Path
from dataclasses import asdict
import numpy as np
import qiskit, scipy
from qiskit import QuantumCircuit, transpile, qpy
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from .core import Op, simulate, resources, evaluate, wire_dag, REGISTRY

OUT = Path('experiments/pilot_20261021')
BASIS = ['rz','sx','x','cx']

def encode(qc):
    return [Op(i.operation.name,tuple(qc.find_bit(q).index for q in i.qubits),tuple(float(p) for p in i.operation.params)) for i in qc.data]

def decode(n,ops):
    qc = QuantumCircuit(n)
    for op in ops: getattr(qc,op.gate)(*op.params,*op.wires)
    return qc

def request(n, target_ref, constraints):
    return {'schema_version':'1','task':{'kind':'pure_state_preparation','data_qubits':n,'input':'zero',
        'target_ref':target_ref,'target_basis_order':'qiskit_little_endian','fidelity_min':0.999,
        'fidelity_checkpoint':'end_of_preparation_before_readout'},
        'hard_constraints':[{'kind':'allowed_gates','values':BASIS,'scope':'logical'}]+constraints}

def refine(n, target, source):
    started = time.monotonic(); calls = 0; proposals = 0; pool = [('seed',source)]
    current = list(source)
    # Deterministic suffix-first, then distributed singleton deletion proposals.
    cuts = [list(range(k,len(source))) for k in range(max(0,len(source)-4),len(source))]
    cuts += [[int(i)] for i in np.linspace(0,max(0,len(source)-1),12)] if source else []
    for cut in cuts[:16]:
        if time.monotonic()-started >= 2 or calls >= 1200: break
        candidate = [op for i,op in enumerate(source) if i not in cut]
        proposals += 1
        indices = [i for i,op in enumerate(candidate) if op.gate=='rz']
        def score(x):
            nonlocal calls
            if time.monotonic()-started >= 2 or calls >= 1200: raise TimeoutError
            patch = [Op(op.gate,op.wires,(float(x[indices.index(i)]),)) if i in indices else op
                     for i,op in enumerate(candidate)]
            calls += 1
            return 1-float(abs(np.vdot(target,simulate(n,patch)))**2)
        try:
            x0 = np.array([candidate[i].params[0] for i in indices])
            if indices:
                opt = minimize(score,x0,method='L-BFGS-B',options={'maxiter':20,'maxls':10})
                candidate = [Op(op.gate,op.wires,(float(opt.x[indices.index(i)]),)) if i in indices else op
                             for i,op in enumerate(candidate)]
            calls += 1
            if abs(np.vdot(target,simulate(n,candidate)))**2 >= .999:
                pool.append(('classical_delete_reopt',candidate))
                if len(candidate)<len(current): current = candidate
        except TimeoutError: break
    # Exact native x replacement, repairs explicit x prohibition without a compression claim.
    if any(op.gate=='x' for op in source):
        repaired = [p for op in source for p in ([Op('sx',op.wires),Op('sx',op.wires)] if op.gate=='x' else [op])]
        pool.append(('classical_x_to_sx_sx',repaired))
    return pool, {'wall_seconds':time.monotonic()-started,'simulator_calls':calls,'proposals':proposals}

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    start = time.monotonic(); rng = np.random.default_rng(20261021)
    targets = []
    for n in (2,3):
        for k in range(12):
            family = ['haar','product','ghz','witness'][k%4]
            if family == 'haar':
                target = rng.normal(size=2**n)+1j*rng.normal(size=2**n); target /= np.linalg.norm(target)
            else:
                qc = QuantumCircuit(n)
                if family == 'product':
                    for q in range(n): qc.ry(float(rng.uniform(-np.pi,np.pi)),q); qc.rz(float(rng.uniform(-np.pi,np.pi)),q)
                elif family == 'ghz':
                    qc.ry(float(rng.uniform(.3,2.8)),0)
                    for q in range(1,n): qc.cx(0,q)
                else:
                    for _ in range(3):
                        for q in range(n): qc.ry(float(rng.uniform(-np.pi,np.pi)),q); qc.rz(float(rng.uniform(-np.pi,np.pi)),q)
                        qc.cx(0,1)
                target = np.asarray(Statevector.from_instruction(qc))
            tid = f'n{n}_{family}_{k:02d}'; np.save(OUT/f'{tid}.npy',target)
            targets.append((tid,n,family,target))
    manifest = {'rng_seed':20261021,'transpiler_seed':17,'threshold':.999,'basis':BASIS,
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'qiskit':qiskit.__version__,'scipy':scipy.__version__},
        'target_sha256':{tid:hashlib.sha256((OUT/f'{tid}.npy').read_bytes()).hexdigest() for tid,_,_,_ in targets},
        'protocol_sha256':hashlib.sha256(Path('PLAN.md').read_bytes()).hexdigest(),
        'registry':REGISTRY,'training':'not run','prior_compute_budget_consumption':'unknown'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[]; seed_records=[]
    for tid,n,family,target in targets:
        if time.monotonic()-start > 225: raise TimeoutError('pilot execution cap; preserve reporting reserve')
        source = QuantumCircuit(n); source.append(StatePreparation(target),range(n))
        compiled={}; timing={}
        for method,level in [('A',0),('B',3)]:
            t=time.monotonic()
            qc=transpile(source,basis_gates=BASIS,optimization_level=level,seed_transpiler=17,approximation_degree=1.0)
            compiled[method]=encode(qc); timing[method]=time.monotonic()-t
            with (OUT/f'{tid}_{method}.qpy').open('wb') as f: qpy.dump(qc,f)
            independent=simulate(n,compiled[method]); reference=np.asarray(Statevector.from_instruction(qc))
            assert abs(np.vdot(independent,reference))**2 > 1-1e-10
            assert abs(np.vdot(target,independent))**2 > 1-1e-10
        pool,edit_cost=refine(n,target,compiled['A'])
        seed_records.append({'target_id':tid,'seed_a': [asdict(x) for x in compiled['A']],
            'seed_b':[asdict(x) for x in compiled['B']], 'seed_dag':wire_dag(n,compiled['A']),
            'editing_cost':edit_cost,'compiler_seconds':timing,
            'candidate_pool':[{'origin':origin,'ops':[asdict(x) for x in ops]} for origin,ops in pool]})
        constraints=[('loose',[]),('tight',[{'kind':'unitary_gate_count','max':max(0,len(compiled['A'])-1),'scope':'logical'}])]
        if any(op.gate=='x' for op in compiled['A']):
            constraints.append(('forbid_x',[{'kind':'forbidden_gates','values':['x'],'scope':'logical'}]))
        for task,cs in constraints:
            req=request(n,f'{tid}.npy',cs)
            for method in ('A','B','C'):
                candidates=pool if method=='C' else [(f'qiskit_level_{0 if method=="A" else 3}',compiled[method])]
                scored=[(evaluate(req,target,ops),origin,ops) for origin,ops in candidates]
                feasible=[x for x in scored if x[0]['status']=='qualified_solution']
                selection=min(feasible,key=lambda x:(len(x[2]),resources(n,x[2])['depth'])) if feasible else scored[0]
                result,origin,ops=selection
                row={'request_id':f'{tid}_{task}','target_id':tid,'family':family,'qubits':n,'task':task,
                    'method':method,'request':req,'result':result,'origin':origin,'unchanged_from_seed':ops==compiled['A'],
                    'ops':[asdict(x) for x in ops], 'candidate_verifications':len(scored),
                    'incremental_edit_seconds':edit_cost['wall_seconds'] if method=='C' else 0,
                    'end_to_end_seconds':timing['A']+edit_cost['wall_seconds'] if method=='C' else timing[method],
                    'candidate_pool_reused_across_requests':True}
                rows.append(row)
                with (OUT/'runs.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps({'target':tid,'seed_gates':len(compiled['A']),'optimized_gates':len(compiled['B']),**edit_cost}),flush=True)
    (OUT/'seeds_and_edits.json').write_text(json.dumps(seed_records,indent=2))
    summary={'wall_seconds':time.monotonic()-start,'targets':len(targets),'rows':len(rows),'groups':{}}
    for task in ('loose','tight','forbid_x'):
        for method in ('A','B','C'):
            group=[r for r in rows if r['task']==task and r['method']==method]
            if not group: continue
            passed=[r for r in group if r['result']['status']=='qualified_solution']
            summary['groups'][f'{task}/{method}']={'qualified':len(passed),'requests':len(group),
                'unchanged':sum(r['unchanged_from_seed'] for r in passed),
                'mean_gate_count_all':float(np.mean([r['result']['resources']['unitary_gate_count'] for r in group])),
                'mean_depth_all':float(np.mean([r['result']['resources']['depth'] for r in group])),
                'mean_fidelity_all':float(np.mean([r['result']['fidelity'] for r in group]))}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
