import json, time, hashlib
from pathlib import Path
import numpy as np
from qiskit.quantum_info import Statevector
from .core import Op,evaluate,simulate,resources
from .pilot import OUT,decode,request

def ops_from(data): return [Op(x['gate'],tuple(x['wires']),tuple(x['params'])) for x in data]

def main():
    start=time.monotonic(); count=0; max_error=0.
    manifest=json.loads((OUT/'manifest.json').read_text())
    for tid,digest in manifest['target_sha256'].items():
        assert hashlib.sha256((OUT/f'{tid}.npy').read_bytes()).hexdigest()==digest
    for line in (OUT/'runs.jsonl').read_text().splitlines():
        row=json.loads(line); ops=ops_from(row['ops']); n=row['qubits']
        target=np.load(OUT/f'{row["target_id"]}.npy')
        result=evaluate(row['request'],target,ops)
        assert result==row['result']
        qstate=np.asarray(Statevector.from_instruction(decode(n,ops)))
        state=simulate(n,ops)
        error=abs(float(abs(np.vdot(target,qstate))**2)-result['fidelity'])
        max_error=max(max_error,error)
        assert abs(np.vdot(state,qstate))**2 > 1-1e-10
        assert error<1e-10
        count+=1
    seeds=json.loads((OUT/'seeds_and_edits.json').read_text()); extra=[]
    for record in seeds:
        tid=record['target_id']; target=np.load(OUT/f'{tid}.npy'); n=int(np.log2(len(target)))
        req=request(n,f'{tid}.npy',[{'kind':'forbidden_gates','values':['sx'],'scope':'logical'}])
        req['hard_constraints'][0]['values']=['ry','rz','x','cx']
        for key in ('seed_a','seed_b'):
            original=ops_from(record[key])
            for repair in (False,True):
                t=time.monotonic()
                ops=[p for op in original for p in (
                    [Op('rz',op.wires,(np.pi/2,)),Op('ry',op.wires,(np.pi/2,)),Op('rz',op.wires,(-np.pi/2,))]
                    if repair and op.gate=='sx' else [op])]
                result=evaluate(req,target,ops)
                ref=np.asarray(Statevector.from_instruction(decode(n,ops)))
                assert abs(float(abs(np.vdot(target,ref))**2)-result['fidelity'])<1e-10
                assert (result['status']=='qualified_solution')==repair
                extra.append({'request_id':tid+'_forbid_sx','target_id':tid,'method':key+('_exact_repair' if repair else ''),
                    'request':req,'result':result,'ops':[{'gate':op.gate,'wires':op.wires,'params':op.params} for op in ops],
                    'wall_seconds':time.monotonic()-t,'origin':'deterministic_gate_decomposition' if repair else 'compiler_seed',
                    'compression_claim':False})
    (OUT/'gate_repair.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in extra))
    result={'replayed_pilot_rows':count,'gate_repair_rows':len(extra),'target_hashes_verified':len(seeds),
        'max_independent_fidelity_error':max_error,'wall_seconds':time.monotonic()-start,
        'gate_repair_summary':{method:{'qualified':sum(r['result']['status']=='qualified_solution' for r in extra if r['method']==method),
            'total':sum(r['method']==method for r in extra)} for method in sorted(set(r['method'] for r in extra))}}
    (OUT/'replay.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
