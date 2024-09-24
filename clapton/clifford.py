import stim
from copy import deepcopy
from clapton.gate_ids import C1ids, RXids, RYids, RZids, Q2ids
from clapton.depolarization import DepolarizationModel


class ParametrizedClifford:
    def __init__(
            self, 
            label: str, 
            qbs: tuple[int] | tuple[int,int], 
            param_dim: int
        ):
        self.label = label
        self.qbs = qbs
        self.param_dim = param_dim
        self.k = 0 # current param, initialized to 0 (often this means no gate / identity)
        self.fixed = False # start out as variable
    def assign(self, k: int):
        assert not self.is_fixed(), "Gate is fixed, not allowed to assign parameter."
        self.k = k
        return self
    def fix(self, k: int):
        self.fixed = False
        self.assign(k)
        self.fixed = True
        return self
    def unfix(self):
        self.fixed = False
        return self
    def is_fixed(self):
        return self.fixed
    def get_stim_id(self):
        pass


class Parametrized1QClifford(ParametrizedClifford):
    def __init__(
            self, 
            label: str, 
            qb: int, 
            param_dim: int
        ):
        super().__init__(label, (qb,), param_dim)


class Parametrized2QClifford(ParametrizedClifford):
    # currently only CX and SWAP
    def __init__(self, qb1: int, qb2: int):
        super().__init__("2Q", (qb1, qb2), 4)
        self.ct_flipped = False
    def assign(self, k: int):
        assert not self.is_fixed(), "Gate is fixed, not allowed to assign parameter."
        self.k = k
        if k == 1 and self.ct_flipped:
            self.qbs = self.qbs[::-1]
            self.ct_flipped = False 
        elif k == 2 and not self.ct_flipped:
            self.qbs = self.qbs[::-1]
            self.ct_flipped = True
        return self
    def get_stim_id(self):
        return Q2ids[self.k]


class ParametrizedRXClifford(Parametrized1QClifford):
    def __init__(self, qb: int):
        super().__init__("RX", qb, 4)
    def get_stim_id(self):
        return RXids[self.k]
    

class ParametrizedRYClifford(Parametrized1QClifford):
    def __init__(self, qb: int):
        super().__init__("RY", qb, 4)
    def get_stim_id(self):
        return RYids[self.k]


class ParametrizedRZClifford(Parametrized1QClifford):
    def __init__(self, qb: int):
        super().__init__("RZ", qb, 4)
    def get_stim_id(self):
        return RZids[self.k]


class ParametrizedAny1QClifford(Parametrized1QClifford):
    def __init__(self, qb: int):
        super().__init__("C1", qb, 24)
    def get_stim_id(self):
        return C1ids[self.k]
    

class ParametrizedCliffordCircuit:
    def __init__(self):
        self.gates: list[ParametrizedClifford] = []
        self.num_physical_qubits: int = 0
        self.parameter_map = None
        self.measurement_map = None
        self.inverse_measurement_map = None
        self.depolarization_model = None
        self.readout_errors = None
        self.circ_snapshot = None
        self.circ_snapshot_noiseless = None
    def _append_gate(self, GateType, *qbs):
        gate = GateType(*qbs)
        self.gates.append(gate)
        for qb in qbs:
            if qb > self.num_physical_qubits - 1:
                self.num_physical_qubits = qb + 1
        return gate
    def RX(self, qb: int):
        return self._append_gate(ParametrizedRXClifford, qb)
    def RY(self, qb: int):
        return self._append_gate(ParametrizedRYClifford, qb)
    def RZ(self, qb: int):
        return self._append_gate(ParametrizedRZClifford, qb)
    def C1(self, qb: int):
        return self._append_gate(ParametrizedAny1QClifford, qb)
    def Q2(self, qb1: int, qb2: int):
        return self._append_gate(Parametrized2QClifford, qb1, qb2)
    def read(self):
        return [gate.k for gate in self.gates if not gate.is_fixed()]
    def assign(self, params: list[int]):
        num_param_gates = self.number_parametrized_gates()
        assert num_param_gates == len(params), f"Number of parameters given ({len(params)}) not consistent with number of parametrized gates ({num_param_gates})."
        param_map = self.get_parameter_map()
        param_idx = 0
        for gate in self.gates:
            if not gate.is_fixed():
                # since there is a non-fixed gate, param_map should not be None and contain this index
                input_param_idx = param_map[param_idx]
                k = params[input_param_idx]
                gate.assign(k)
                param_idx += 1
        return self
    def fix(self, params: list[int], return_copy=False):
        if return_copy:
            pcirc = deepcopy(self)
        else:
            pcirc = self
        num_param_gates = pcirc.number_parametrized_gates()
        assert num_param_gates == len(params), f"Number of parameters given ({len(params)}) not consistent with number of parametrized gates ({num_param_gates})."
        param_map = pcirc.get_parameter_map()
        param_idx = 0
        for gate in pcirc.gates:
            if not gate.is_fixed():
                # since there is a non-fixed gate, param_map should not be None and contain this index
                input_param_idx = param_map[param_idx]
                k = params[input_param_idx]
                gate.fix(k)
                param_idx += 1
        return pcirc
    def _add_measurements(self, circ: stim.Circuit, pauli: str):
        meas_map = self.get_measurement_map()
        targets = []
        if self.readout_errors is None: 
            for v_qb, p in enumerate(pauli):
                p_qb = meas_map(v_qb)
                if p == "X":
                    targets.append(stim.target_x(p_qb))
                    targets.append(stim.target_combiner())
                elif p == "Y":
                    targets.append(stim.target_y(p_qb))
                    targets.append(stim.target_combiner())
                elif p == "Z":
                    targets.append(stim.target_z(p_qb))
                    targets.append(stim.target_combiner())
        else:
            # meas basis is already prepared
            for v_qb, p in enumerate(pauli):
                p_qb = meas_map(v_qb)
                if p == "X":
                    circ.append("S", p_qb)
                    circ.append("SQRT_X", p_qb)
                    if self.depolarization_model is not None:
                        gate = ParametrizedRXClifford(p_qb).assign(1)
                        p = self.depolarization_model.get_gate_depolarization(gate)
                        circ.append(f"DEPOLARIZE{len(gate.qbs)}", gate.qbs, p)
                elif p == "Y":
                    circ.append("SQRT_X", p_qb)
                    if self.depolarization_model is not None:
                        gate = ParametrizedRXClifford(p_qb).assign(1)
                        p = self.depolarization_model.get_gate_depolarization(gate)
                        circ.append(f"DEPOLARIZE{len(gate.qbs)}", gate.qbs, p)
                elif p == "Z":
                    pass
                else:
                    # no measurement for other characters
                    continue
                if p_qb in self.readout_errors:
                    circ.append("X_ERROR", p_qb, self.readout_errors[p_qb])
                targets.append(stim.target_z(p_qb))
                targets.append(stim.target_combiner())
        circ.append("MPP", targets[:-1])
    def stim_circuit(self, pauli: str = None):
        """pauli: Pauli to measure"""
        circ = stim.Circuit()
        for gate in self.gates:
            gate_id = gate.get_stim_id()
            if self.depolarization_model is not None:
                p = self.depolarization_model.get_gate_depolarization(gate)
            else:
                p = None
            if gate_id is not None:
                if not gate is ParametrizedAny1QClifford:
                    gate_id = [gate_id]
                for _gate_id in gate_id:
                    circ.append(_gate_id, gate.qbs)
                if p is not None:
                    circ.append(f"DEPOLARIZE{len(gate.qbs)}", gate.qbs, p)
        if pauli is not None:
            self._add_measurements(circ, pauli)
        return circ
    def snapshot(self, pauli: str = None):
        self.circ_snapshot = self.stim_circuit(pauli)
        return self
    def snapshot_noiseless(self, pauli: str = None):
        depol_model = self.depolarization_model
        self.remove_depolarization()
        self.circ_snapshot_noiseless = self.stim_circuit(pauli)
        self.add_depolarization_model(depol_model)
        return self
    def define_parameter_map(self, param_map: dict[int, int]):
        """{circuit_param_idx: input_param_idx}"""
        self.parameter_map = param_map
        return self
    def get_parameter_map(self):
        if self.parameter_map is None:
            return {i: i for i in range(self.number_parametrized_gates())}
        else:
            return self.parameter_map
    def define_measurement_map(self, meas_map_dict):
        """function: physical_qb = meas_map(virtual_qb)"""
        if meas_map_dict is None:
            self.measurement_map = None
            self.inverse_measurement_map = None
        else:
            def meas_map(i):
                if i in meas_map_dict:
                    return meas_map_dict[i]
                else:
                    raise Exception(f"measurement input {i} was not defined in custom map")
            inv_meas_map_dict = {v: k for (k, v) in meas_map_dict.items()}
            def inv_meas_map(i):
                if i in inv_meas_map_dict:
                    return inv_meas_map_dict[i]
                else:
                    raise Exception(f"measurement output {i} was not defined in custom map")
            self.measurement_map = meas_map
            self.inverse_measurement_map = inv_meas_map
        return self
    def get_measurement_map(self):
        if self.measurement_map is None:
            return lambda i: i
        else:
            return self.measurement_map
    def get_inverse_measurement_map(self):
        if self.inverse_measurement_map is None:
            return lambda i: i
        else:
            return self.inverse_measurement_map
    def has_custom_measurement_map(self):
        return self.measurement_map is not None
    def add_depolarization_model(self, depol_model):
        assert isinstance(depol_model, DepolarizationModel) or depol_model is None
        self.depolarization_model = depol_model
        return self
    def has_depolarization(self):
        return self.depolarization_model is not None
    def remove_depolarization(self):
        self.depolarization_model = None
        return self
    def add_readout_errors(self, r_dict):
        """{qb: r_error}"""
        self.readout_errors = r_dict
        return self
    def has_readout_errors(self):
        return self.readout_errors is not None
    def remove_readout_errors(self):
        self.readout_errors = None
        return self
    def has_errors(self):
        return self.has_depolarization() or self.has_readout_errors()
    def number_parametrized_gates(self):
        return sum([1 for gate in self.gates if not gate.is_fixed()])
    def parameter_dimensions(self):
        return [gate.param_dim for gate in self.gates if not gate.is_fixed()]
    def parameter_space(self):
        return [list(range(d)) for d in self.parameter_dimensions()]
    def is_fixed(self):
        return self.number_parametrized_gates() == 0
    def idc_param_2qb(self):
        return [i for i, gate in enumerate(self.gates) if isinstance(gate, Parametrized2QClifford) and not gate.is_fixed()]