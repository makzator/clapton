import stim
from copy import deepcopy
from clapton.gate_ids import C1ids, RXids, RYids, RZids, Q2ids
from clapton.depolarization import DepolarizationModel


class ParametrizedClifford:
    """
    Superclass.
    A discretely parametrized Clifford operation with parameter k.
    A gate can be "fixed", meaning that the value of k cannot be changed by the optimizer and
    is thus not an optimization variable.

    Args:
        label: Label of the operation, e.g. "RX".
        qbs: Qubit indices the operation acts on, e.g. (2,) for 1Q gate or (2,3) for 2Q gate.
        param_dim: Dimension of parameter space, i.e. how many values k can assume.
    """
    def __init__(
            self, 
            label: str, 
            qbs: tuple[int] | tuple[int,int], 
            param_dim: int
        ):
        self.label = label
        self.qbs = qbs
        self.param_dim = param_dim
        # initialize k to 0, often this means no gate / identity
        self.k = 0
        # start out as variable
        self.fixed = False 
    def assign(self, k: int):
        """Assign a new value to k"""
        assert not self.is_fixed(), "Gate is fixed, not allowed to assign parameter."
        self.k = k
        return self
    def fix(self, k: int):
        """Fix the gate to a value of k. Not optimizable."""
        self.fixed = False
        self.assign(k)
        self.fixed = True
        return self
    def unfix(self):
        """Release fixed state. Remains at k."""
        self.fixed = False
        return self
    def is_fixed(self):
        return self.fixed
    def get_stim_id(self):
        """Return the stim gate identifier."""
        pass


class Parametrized1QClifford(ParametrizedClifford):
    """
    1Q Clifford operation.
    """
    def __init__(
            self, 
            label: str, 
            qb: int, 
            param_dim: int
        ):
        super().__init__(label, (qb,), param_dim)


class Parametrized2QClifford(ParametrizedClifford):
    """
    2Q Clifford operation. dim = 4.
    [None (no gate), CX, CX (control-target swapped), SWAP]
    """
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
    """
    1Q Clifford RX gate. dim = 4.
    [I, SQRT_X, X, SQRT_X_DAG]
    """
    def __init__(self, qb: int):
        super().__init__("RX", qb, 4)
    def get_stim_id(self):
        return RXids[self.k]
    

class ParametrizedRYClifford(Parametrized1QClifford):
    """
    1Q Clifford RY gate. dim = 4.
    [I, SQRT_Y, Y, SQRT_Y_DAG]
    """
    def __init__(self, qb: int):
        super().__init__("RY", qb, 4)
    def get_stim_id(self):
        return RYids[self.k]


class ParametrizedRZClifford(Parametrized1QClifford):
    """
    1Q Clifford RZ gate. dim = 4.
    [I, S, Z, S_DAG]
    """
    def __init__(self, qb: int):
        super().__init__("RZ", qb, 4)
    def get_stim_id(self):
        return RZids[self.k]


class ParametrizedAny1QClifford(Parametrized1QClifford):
    """
    Arbitrary 1Q Clifford gate. dim = 24.
    Every possible 1Q Clifford.
    """
    def __init__(self, qb: int):
        super().__init__("C1", qb, 24)
    def get_stim_id(self):
        return C1ids[self.k]
    

class ParametrizedCliffordCircuit:
    """
    Circuit of ParametrizedClifford objects. Can be viewed as an object holding
    all the gate parameters and circuit info.

    No initializaion, instead add instructions similar to a qiskit or stim
    circuit.
    """
    def __init__(self):
        self.gates: list[ParametrizedClifford] = []
        self.num_physical_qubits: int = 0
        self.parameter_map = None
        self.inverse_parameter_map = None
        self.measurement_map = None
        self.inverse_measurement_map = None
        self.depolarization_model = None
        self.readout_errors = None
        self.circ_snapshot = None
        self.circ_snapshot_noiseless = None
    def _append_gate(self, GateType, *qbs):
        """Hidden function that appends gate to list and updates #qubits."""
        gate = GateType(*qbs)
        self.gates.append(gate)
        for qb in qbs:
            if qb > self.num_physical_qubits - 1:
                self.num_physical_qubits = qb + 1
        return gate
    def RX(self, qb: int):
        """Add RX gate."""
        return self._append_gate(ParametrizedRXClifford, qb)
    def RY(self, qb: int):
        """Add RY gate."""
        return self._append_gate(ParametrizedRYClifford, qb)
    def RZ(self, qb: int):
        """Add RZ gate."""
        return self._append_gate(ParametrizedRZClifford, qb)
    def C1(self, qb: int):
        """Add abitrary 1Q Clifford gate."""
        return self._append_gate(ParametrizedAny1QClifford, qb)
    def Q2(self, qb1: int, qb2: int):
        """Add 2Q Clifford gate."""
        return self._append_gate(Parametrized2QClifford, qb1, qb2)
    def read(self):
        """Return parameter vector of gates that are not fixed, respecting parameter map."""
        param_map = self.get_parameter_map()
        pgates_total_idc = [i for (i, gate) in enumerate(self.gates) if not gate.is_fixed()]
        return [self.gates[pgates_total_idc[param_map(i)]].k for i in range(self.number_parametrized_gates())]
    def internal_read(self):
        """Return parameter vector of gates that are not fixed, as ordered in pcirc."""
        return [gate.k for gate in self.gates if not gate.is_fixed()]
    def assign(self, params: list[int]):
        """
        Assign parameter vector to gates that are not fixed, 
        respecting the parameter mapping.
        """
        num_param_gates = self.number_parametrized_gates()
        assert num_param_gates == len(params), f"Number of parameters given ({len(params)}) not consistent with number of parametrized gates ({num_param_gates})."
        inv_param_map = self.get_inverse_parameter_map()
        param_idx = 0
        for gate in self.gates:
            # only check variable gates
            if not gate.is_fixed():
                # since there is a non-fixed gate, param_map should not be None and contain this index
                circ_param_idx = inv_param_map(param_idx)
                k = params[circ_param_idx]
                gate.assign(k)
                param_idx += 1
        return self
    def fix(self, params: list[int], return_copy=False):
        """
        Fix all parametrized gates to a value determined by given param vector,
        respecting the parameter mapping.
        Can return copy of the circuit if desired as this would permute the
        original circuit and knowledge about which gates were variable would be 
        lost.
        """
        if return_copy:
            pcirc = deepcopy(self)
        else:
            pcirc = self
        num_param_gates = pcirc.number_parametrized_gates()
        assert num_param_gates == len(params), f"Number of parameters given ({len(params)}) not consistent with number of parametrized gates ({num_param_gates})."
        inv_param_map = pcirc.get_inverse_parameter_map()
        param_idx = 0
        for gate in pcirc.gates:
            # only check variable gates
            if not gate.is_fixed():
                # since there is a non-fixed gate, inv_param_map should not be None and contain this index
                circ_param_idx = inv_param_map(param_idx)
                k = params[circ_param_idx]
                gate.fix(k)
                param_idx += 1
        return pcirc
    def _add_measurements(self, circ: stim.Circuit, pauli: str):
        """
        Hidden function that adds Pauli measurement to stim circuit
        generated from self, including noise model and respecting measurement
        map.
        """
        meas_map = self.get_measurement_map()
        targets = []
        if self.depolarization_model is None and self.readout_errors is None: 
            # if no noise, can directly evaluate measurement
            for v_qb, p in enumerate(pauli):
                # convert virtual qb (index of p in Pauli) to physical qb in circuit
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
            # if noise, measurement basis has to be explicitly prepared
            for v_qb, p in enumerate(pauli):
                # convert virtual qb (index of p in Pauli) to physical qb in circuit
                p_qb = meas_map(v_qb)
                # prepare measurement basis and possibly insert gate errors
                if p == "X":
                    circ.append("S", p_qb)
                    if self.depolarization_model is not None:
                        gate = ParametrizedRZClifford(p_qb).assign(1)
                        p = self.depolarization_model.get_gate_depolarization(gate)
                        if p is not None:
                            circ.append(f"DEPOLARIZE{len(gate.qbs)}", gate.qbs, p)
                    circ.append("SQRT_X", p_qb)
                    if self.depolarization_model is not None:
                        gate = ParametrizedRXClifford(p_qb).assign(1)
                        p = self.depolarization_model.get_gate_depolarization(gate)
                        if p is not None:
                            circ.append(f"DEPOLARIZE{len(gate.qbs)}", gate.qbs, p)
                elif p == "Y":
                    circ.append("SQRT_X", p_qb)
                    if self.depolarization_model is not None:
                        gate = ParametrizedRXClifford(p_qb).assign(1)
                        p = self.depolarization_model.get_gate_depolarization(gate)
                        if p is not None:
                            circ.append(f"DEPOLARIZE{len(gate.qbs)}", gate.qbs, p)
                elif p == "Z":
                    pass
                else:
                    # no measurement for other characters in Pauli string
                    continue
                # if readout errors on this physical qubit, insert bit flip
                if self.readout_errors is not None and p_qb in self.readout_errors:
                    circ.append("X_ERROR", p_qb, self.readout_errors[p_qb])
                # measure in computational basis
                targets.append(stim.target_z(p_qb))
                targets.append(stim.target_combiner())
        # add combined Pauli measurement to stim circ
        circ.append("MPP", targets[:-1])
    def stim_circuit(self, pauli: str = None):
        """
        Convert pcirc to stim circuit, gates are converted to stim gates
        corresponding to their current parameter k. Includes noise and
        potentially Pauli measurement.
        """
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
        """
        Compute stim circuit and store internally.
        Enables less re-computation during optimization.
        """
        self.circ_snapshot = self.stim_circuit(pauli)
        return self
    def snapshot_noiseless(self, pauli: str = None):
        """
        Compute stim circuit without noise and store internally.
        Enables less re-computation during optimization.
        """
        depol_model = self.depolarization_model
        self.remove_depolarization()
        self.circ_snapshot_noiseless = self.stim_circuit(pauli)
        self.add_depolarization_model(depol_model)
        return self
    def define_parameter_map(self, param_map_dict: dict[int, int] | None):
        """
        Define mapping function between indices in parameter vectors that are given
        to assign() and indices of parametrized gates in pcirc. Relevant when
        interacting with a parametrized qiskit circuit where the parameter order
        does not match the order in pcirc.

        Args:
            param_map: {circuit_param_idx: pcirc_param_idx}
        """
        # compute param map function and inverse map function
        if param_map_dict is None:
            # if None, reset maps to None - corresponds to identity
            self.parameter_map = None
            self.inverse_parameter_map = None
        else:
            # compute map and inverse map from provided dict
            def param_map(i):
                if i in param_map_dict:
                    return param_map_dict[i]
                else:
                    raise Exception(f"circuit param idx {i} was not defined in custom map")
            inv_param_map_dict = {v: k for (k, v) in param_map_dict.items()}
            def inv_param_map(i):
                if i in inv_param_map_dict:
                    return inv_param_map_dict[i]
                else:
                    raise Exception(f"internal pcirc param idx {i} was not defined in custom map")
            self.parameter_map = param_map
            self.inverse_parameter_map = inv_param_map
        return self
    def get_parameter_map(self):
        """Read from parameter map."""
        if self.parameter_map is None:
            return lambda i: i
        else:
            return self.parameter_map
    def get_inverse_parameter_map(self):
        """Read from inverted parameter map."""
        if self.inverse_parameter_map is None:
            return lambda i: i
        else:
            return self.inverse_parameter_map
    def has_custom_parameter_map(self):
        return self.parameter_map is not None
    def define_measurement_map(self, meas_map_dict: dict[int, int] | None):
        """
        Define mapping function from virtual qubit (index of a term in a Pauli 
        string) to a physical qubit in pcirc.
        Example: 
            pauli = "XY"
            phys_qb = meas_map(0)   # want to map first virtual qubit to measure "X"
            # add measurement of X on qubit phys_qb in pcirc

        Args:
            meas_map_dict = {virtual_qb: physical_qb}
        """
        # compute meas map function and inverse map function
        if meas_map_dict is None:
            # if None, reset maps to None - corresponds to identity
            self.measurement_map = None
            self.inverse_measurement_map = None
        else:
            # compute map and inverse map from provided dict
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
        """Read from measurement map."""
        if self.measurement_map is None:
            return lambda i: i
        else:
            return self.measurement_map
    def get_inverse_measurement_map(self):
        """Read from inverted measurement map."""
        if self.inverse_measurement_map is None:
            return lambda i: i
        else:
            return self.inverse_measurement_map
    def has_custom_measurement_map(self):
        return self.measurement_map is not None
    def add_depolarization_model(self, depol_model):
        """Addd gate depolarization model or set to None (no errors)."""
        assert isinstance(depol_model, DepolarizationModel) or depol_model is None
        self.depolarization_model = depol_model
        return self
    def has_depolarization(self):
        return self.depolarization_model is not None
    def remove_depolarization(self):
        """Reset gate errors to None (no errors)."""
        self.depolarization_model = None
        return self
    def add_readout_errors(self, r_dict):
        """
        Add measurement errors, i.e. stochastic bit flips with chance r_error.
        
        Args:
            r_dict = {physical_qb: r_error}
        """
        self.readout_errors = r_dict
        return self
    def has_readout_errors(self):
        return self.readout_errors is not None
    def remove_readout_errors(self):
        """Reset measurement errors to None (no errors)."""
        self.readout_errors = None
        return self
    def has_errors(self):
        return self.has_depolarization() or self.has_readout_errors()
    def number_parametrized_gates(self):
        return sum([1 for gate in self.gates if not gate.is_fixed()])
    def parameter_dimensions(self):
        """Return dimension vector."""
        return [gate.param_dim for gate in self.gates if not gate.is_fixed()]
    def parameter_space(self):
        """Return all possible parameter values."""
        return [list(range(d)) for d in self.parameter_dimensions()]
    def is_fixed(self):
        """Return if pcirc is fixed (if all gates fixed)."""
        return self.number_parametrized_gates() == 0
    def idc_param_2qb(self):
        """Return indices of 2Q gates in pcirc, respecting parameter map."""
        inv_param_map = self.get_inverse_parameter_map()
        return [inv_param_map(i) for i, gate in enumerate(self.gates) if isinstance(gate, Parametrized2QClifford) and not gate.is_fixed()]