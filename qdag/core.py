from dataclasses import dataclass
from collections import Counter
import numpy as np

ALIASES = {'cnot': 'cx', 'fredkin': 'cswap'}
ARITY = {'x':1, 'h':1, 'sx':1, 'ry':1, 'rz':1, 'cx':2, 'swap':2, 'cswap':3, 'ccx':3}
REGISTRY = {g: {'arity': a, 'classical_arity': 0,
    'roles': ({'cx':['control','target'], 'cswap':['control','swap_target_1','swap_target_2'],
               'ccx':['control_1','control_2','target']}.get(g, [f'wire_{i}' for i in range(a)])),
    'parameter_period': 4*np.pi if g in ('ry','rz') else None,
    'simulator_support': True, 'checkpoint_generation_support': False, 'hardware_support': False}
    for g,a in ARITY.items()}

def canonical(name):
    return ALIASES.get(name.lower(), name.lower())

@dataclass(frozen=True)
class Op:
    gate: str
    wires: tuple
    params: tuple = ()
    def __post_init__(self):
        object.__setattr__(self, 'gate', canonical(self.gate))
        object.__setattr__(self, 'wires', tuple(self.wires))
        object.__setattr__(self, 'params', tuple(self.params))

def validate_ops(n, ops):
    for op in ops:
        if op.gate not in ARITY:
            raise ValueError('unknown operation')
        if len(op.wires) != ARITY[op.gate] or len(set(op.wires)) != len(op.wires):
            raise ValueError('invalid ordered operands')
        if any(type(q) is not int or not 0 <= q < n for q in op.wires):
            raise ValueError('wire out of range')
        if len(op.params) != (1 if op.gate in ('ry','rz') else 0) or not np.all(np.isfinite(op.params)):
            raise ValueError('invalid parameters')

def simulate(n, ops, initial=None):
    validate_ops(n, ops)
    state = np.zeros(2**n, complex) if initial is None else np.array(initial, complex, copy=True)
    if initial is None: state[0] = 1
    for op in ops:
        g, wires = op.gate, op.wires
        out = np.zeros_like(state)
        if len(wires) == 1:
            if g == 'x': m = np.array([[0,1],[1,0]])
            elif g == 'h': m = np.array([[1,1],[1,-1]]) / np.sqrt(2)
            elif g == 'sx': m = np.array([[1+1j,1-1j],[1-1j,1+1j]]) / 2
            elif g == 'ry':
                c,s = np.cos(op.params[0]/2), np.sin(op.params[0]/2)
                m = np.array([[c,-s],[s,c]])
            else: m = np.diag(np.exp(np.array([-1j,1j])*op.params[0]/2))
            q = wires[0]
            for i,a in enumerate(state):
                b = (i >> q) & 1
                for j in (0,1): out[(i & ~(1<<q)) | (j<<q)] += m[j,b]*a
        else:
            for i,a in enumerate(state):
                j = i
                if g == 'cx' and (i >> wires[0]) & 1: j ^= 1 << wires[1]
                elif g == 'ccx' and all((i >> q)&1 for q in wires[:2]): j ^= 1 << wires[2]
                elif g in ('swap','cswap'):
                    active = g == 'swap' or ((i >> wires[0]) & 1)
                    u,v = wires[-2:]
                    if active and ((i>>u)&1) != ((i>>v)&1): j ^= (1<<u)|(1<<v)
                out[j] += a
        state = out
    return state

def resources(n, ops):
    validate_ops(n, ops)
    frontier, two = [0]*n, [0]*n
    counts = Counter(op.gate for op in ops)
    for op in ops:
        d = 1 + max(frontier[q] for q in op.wires)
        d2 = max(two[q] for q in op.wires) + (len(op.wires)==2)
        for q in op.wires: frontier[q], two[q] = d, d2
    return {'gate_counts':dict(counts), 'unitary_gate_count':len(ops),
            'depth':max(frontier,default=0), 'two_qubit_depth':max(two,default=0)}

def wire_dag(n, ops):
    last, nodes = {}, []
    for i,op in enumerate(ops):
        nodes.append({'id':i, 'gate':op.gate, 'wires':op.wires, 'params':op.params,
                      'incoming':[{'wire':q,'operand_role':REGISTRY[op.gate]['roles'][r], 'source':last.get(q)}
                                  for r,q in enumerate(op.wires)]})
        for q in op.wires: last[q] = i
    return nodes

KINDS = {'allowed_gates','forbidden_gates','required_gates','gate_count',
         'unitary_gate_count','depth','two_qubit_depth','allowed_operands','ancilla_count'}

def normalize(req):
    if set(req)-{'schema_version','task','hard_constraints','readout','preferences'}:
        raise NotImplementedError('request fields (including search orchestration)')
    if req.get('schema_version') != '1': raise NotImplementedError('schema version')
    t = req['task']
    if set(t)-{'kind','data_qubits','input','target_ref','target_basis_order',
               'fidelity_min','fidelity_checkpoint'}:
        raise NotImplementedError('task fields')
    if t.get('kind') != 'pure_state_preparation' or t.get('input') != 'zero': raise NotImplementedError('task/input')
    if t.get('target_basis_order') != 'qiskit_little_endian': raise NotImplementedError('basis order')
    if t.get('fidelity_checkpoint') != 'end_of_preparation_before_readout': raise NotImplementedError('checkpoint')
    n = t['data_qubits']
    if type(n) is not int or n < 1 or not 0 <= t['fidelity_min'] <= 1: raise ValueError('task bounds')
    if req.get('readout') or req.get('preferences'): raise NotImplementedError('readout/preferences')
    normalized = []
    for raw in req.get('hard_constraints',[]):
        c = dict(raw)
        if c.get('kind') not in KINDS or c.get('scope','logical') != 'logical': raise NotImplementedError('constraint/scope')
        kind = c['kind']
        allowed = {'kind','scope','version'}
        if c.get('version','1') != '1': raise NotImplementedError('constraint version')
        if kind in ('allowed_gates','forbidden_gates','required_gates'):
            allowed |= {'values'}
            c['values'] = [canonical(g) for g in c['values']]
            gates = c['values']
        elif kind == 'allowed_operands':
            allowed |= {'gate','values'}
            c['gate'] = canonical(c['gate']); gates = [c['gate']]
            if any(len(v) != ARITY.get(c['gate']) or len(set(v)) != len(v) or
                   any(type(q) is not int or not 0 <= q < n for q in v) for v in c['values']): raise ValueError('operand tuples')
        else:
            allowed |= {'exact','min','max'}
            if kind == 'gate_count':
                allowed.add('gate'); c['gate'] = canonical(c['gate']); gates = [c['gate']]
            else: gates = []
            if kind == 'depth':
                allowed.add('metric')
                if c.get('metric') != 'unit_duration_dependency_depth': raise NotImplementedError('depth metric')
            bounds = [c[k] for k in ('exact','min','max') if k in c]
            if not bounds or any(type(x) is not int or x < 0 for x in bounds): raise ValueError('count bounds')
            if 'exact' in c and ('min' in c or 'max' in c): raise ValueError('mixed bound syntax')
            if kind == 'ancilla_count' and (c.get('exact',0)>0 or c.get('min',0)>0): raise NotImplementedError('ancillas')
        if set(c)-allowed: raise NotImplementedError('unknown constraint fields')
        if any(g not in ARITY for g in gates): raise NotImplementedError('operation')
        normalized.append(c)
    return normalized

def contradictions(n, cs):
    lower,upper = Counter(), {g:float('inf') for g in ARITY}
    total = float('inf')
    for c in cs:
        k = c['kind']
        if k == 'allowed_gates':
            for g in ARITY:
                if g not in c['values']: upper[g] = 0
        elif k == 'forbidden_gates':
            for g in c['values']: upper[g] = 0
        elif k == 'required_gates':
            for g in c['values']: lower[g] = max(lower[g],1)
        elif k == 'gate_count':
            g = c['gate']; lower[g] = max(lower[g], c.get('exact', c.get('min',0)))
            upper[g] = min(upper[g], c.get('exact',c.get('max',float('inf'))))
        elif k == 'unitary_gate_count': total = min(total,c.get('exact',c.get('max',float('inf'))))
        if 'min' in c and 'max' in c and c['min']>c['max']: return 'lower exceeds upper'
    if any(lower[g]>upper[g] for g in ARITY): return 'required and forbidden or conflicting counts'
    if any(lower[g] and ARITY[g]>n for g in ARITY): return 'operation arity exceeds qubits'
    if sum(lower.values())>total: return 'required counts exceed total budget'
    return None

def evaluate(req, target, ops):
    try:
        cs = normalize(req)
    except NotImplementedError as e: return {'status':'unsupported_capability','reason':str(e)}
    except (ValueError,KeyError,TypeError) as e: return {'status':'execution_incomplete','reason':f'invalid request: {e}'}
    n = req['task']['data_qubits']
    why = contradictions(n,cs)
    if why: return {'status':'proven_inconsistent','reason':why}
    target = np.asarray(target,complex)
    if target.shape != (2**n,) or not np.all(np.isfinite(target)) or not np.isclose(np.linalg.norm(target),1,rtol=0,atol=1e-10):
        return {'status':'execution_incomplete','reason':'invalid target'}
    try:
        r = resources(n,ops); state = simulate(n,ops)
    except ValueError as e: return {'status':'execution_incomplete','reason':str(e)}
    fidelity = float(np.clip(abs(np.vdot(target,state))**2,0,1))
    violations = []
    for c in cs:
        k = c['kind']; counts = r['gate_counts']
        if k == 'allowed_gates': ok = set(counts)<=set(c['values'])
        elif k == 'forbidden_gates': ok = not (set(counts)&set(c['values']))
        elif k == 'required_gates': ok = set(c['values'])<=set(counts)
        elif k == 'allowed_operands': ok = all(list(op.wires) in c['values'] for op in ops if op.gate == c['gate'])
        else:
            value = counts.get(c['gate'],0) if k == 'gate_count' else (0 if k == 'ancilla_count' else r[k])
            ok = all(value == v if b == 'exact' else value >= v if b == 'min' else value <= v
                     for b,v in c.items() if b in ('exact','min','max'))
        if not ok: violations.append(c)
    if fidelity < req['task']['fidelity_min']: violations.append({'kind':'fidelity'})
    return {'status':'qualified_solution' if not violations else 'no_qualifying_solution_within_budget',
            'fidelity':fidelity,'resources':r,'violations':violations}
