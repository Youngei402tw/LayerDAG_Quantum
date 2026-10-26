import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from qdag.core import Op, simulate, evaluate, normalize, resources
from qdag.pilot import decode, request

@pytest.mark.parametrize('gate,wires', [('cx',(0,1)),('cx',(1,0)),('swap',(2,0)),('cswap',(0,2,1)),('ccx',(2,0,1))])
def test_all_basis_inputs(gate,wires):
    for i in range(8):
        ops=[Op('x',(q,)) for q in range(3) if (i>>q)&1]+[Op(gate,wires)]
        np.testing.assert_allclose(simulate(3,ops),Statevector.from_instruction(decode(3,ops)),atol=1e-13)

@pytest.mark.parametrize('gate', ['ry','rz','sx','h','x'])
def test_complex_states_angles(gate):
    rng=np.random.default_rng(81)
    for _ in range(8):
        state=rng.normal(size=8)+1j*rng.normal(size=8); state/=np.linalg.norm(state)
        op=Op(gate,(2,),(float(rng.uniform(-12,12)),) if gate in ('ry','rz') else ())
        expected=Statevector(state).evolve(decode(3,[op]))
        np.testing.assert_allclose(simulate(3,[op],state),expected,atol=1e-13)

def test_contract_and_scopes():
    target=simulate(2,[Op('h',(0,)),Op('cx',(0,1))])
    req=request(2,'target.npy',[{'kind':'gate_count','gate':'CNOT','exact':1},
        {'kind':'depth','metric':'unit_duration_dependency_depth','min':2,'max':2}])
    # H must explicitly be in allowed set.
    req['hard_constraints'][0]['values'].append('h')
    assert evaluate(req,target,[Op('h',(0,)),Op('cx',(0,1))])['status']=='qualified_solution'
    req['hard_constraints'].append({'kind':'forbidden_gates','values':['cx']})
    assert evaluate(req,target,[])['status']=='proven_inconsistent'
    req['hard_constraints']=[{'kind':'gate_count','gate':'cswap','min':1}]
    assert evaluate(req,target,[])['status']=='proven_inconsistent'
    req['hard_constraints']=[{'kind':'gate_count','gate':'cx','max':2,'scope':'native'}]
    assert evaluate(req,target,[])['status']=='unsupported_capability'

def test_invalid_and_exhausted():
    req=request(2,'x',[])
    assert evaluate(req,np.array([1,0,0,0]),[Op('cx',(0,0))])['status']=='execution_incomplete'
    assert evaluate(req,np.array([0,1,0,0]),[])['status']=='no_qualifying_solution_within_budget'
    assert evaluate(req,np.array([2,0,0,0]),[])['status']=='execution_incomplete'
    req['hard_constraints'].append({'kind':'unitary_gate_count','max':1,'mystery':True})
    assert evaluate(req,np.array([1,0,0,0]),[])['status']=='unsupported_capability'

def test_full_depth_and_minima():
    assert resources(3,[Op('x',(0,)),Op('x',(1,)),Op('cx',(0,2))])['depth']==2
    req=request(2,'x',[{'kind':'gate_count','gate':'cx','min':1,'max':2},
        {'kind':'allowed_operands','gate':'cx','values':[[1,0]]}])
    assert evaluate(req,np.array([1,0,0,0]),[])['status']=='no_qualifying_solution_within_budget'
    assert evaluate(req,np.array([1,0,0,0]),[Op('cx',(0,1))])['status']=='no_qualifying_solution_within_budget'
    assert evaluate(req,np.array([1,0,0,0]),[Op('cx',(1,0))])['status']=='qualified_solution'
