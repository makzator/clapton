import numpy as np
import stim
from clapton.clifford import ParametrizedCliffordCircuit

### Transformation
def transform_paulis(
        trans_circ: stim.Circuit, 
        paulis: list[str], 
        replace_I: bool = False
    ):
    trans = [stim.PauliString(p).before(trans_circ) for p in paulis]
    signs = [t.sign.real for t in trans]
    if replace_I:
        paulis_trans = [str(t)[1:].replace("_", "I") for t in trans]
    else:
        paulis_trans = [str(t)[1:] for t in trans]
    return paulis_trans, signs



### VQE
def get_expectations_tableau(
        base_pcirc: ParametrizedCliffordCircuit, 
        paulis: list[str]
    ):
    if base_pcirc.circ_snapshot_noiseless is None:
        base_pcirc.snapshot_noiseless()
    if base_pcirc.has_custom_measurement_map():
        meas_map_inv = base_pcirc.get_inverse_measurement_map()
        def map_pauli(p, p_qb):
            try:
                return p[meas_map_inv(p_qb)]
            except:
                return "_"
        paulis = ["".join(
            [map_pauli(p, p_qb) for p_qb in range(base_pcirc.num_physical_qubits)]
            ) for p in paulis]
    sim = stim.TableauSimulator()
    sim.do_circuit(base_pcirc.circ_snapshot_noiseless)
    expectations = [sim.peek_observable_expectation(stim.PauliString(p)) for p in paulis]
    return expectations


def get_expectations(
        base_pcirc: ParametrizedCliffordCircuit, 
        paulis: list[str], 
        get_noiseless: bool = False, 
        shots: int = int(1e4)
    ):
    if get_noiseless:
        return get_expectations_tableau(base_pcirc, paulis)
    if base_pcirc.has_errors():
        # use hidden routines for speedup (build main part of stim circ only once)
        if base_pcirc.circ_snapshot is None:
            base_pcirc.snapshot()
        base_circ = base_pcirc.circ_snapshot
        def _num_true(pauli):
            circ = base_circ.copy()
            base_pcirc._add_measurements(circ, pauli)
            sampler = circ.compile_sampler()
            results = sampler.sample(shots)
            return np.sum(results)
        num_trues = np.fromiter((_num_true(pauli) for pauli in paulis), float, len(paulis))
        expectations = 1 - num_trues/shots * 2
    else:
        expectations = get_expectations_tableau(base_pcirc, paulis)
    return expectations


def get_energy(
        pcirc: ParametrizedCliffordCircuit, 
        paulis: list[str], 
        coeffs: list[float], 
        get_noiseless: bool = False, 
        **expectations_kwargs
    ):
    expectations = get_expectations(pcirc, paulis, get_noiseless, **expectations_kwargs)
    energy = np.inner(expectations, coeffs)
    return energy


def weighted_relative_pauli_weight(
        paulis: list[str], 
        coeffs: list[str]
    ):
    N = len(paulis[0])
    relative_pauli_weights = [1 - p.replace("I", "_").count("_")/N for p in paulis]
    return np.inner(relative_pauli_weights, coeffs)