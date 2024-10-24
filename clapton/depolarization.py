import numpy as np
import stim
# gate objects are ParametrizedClifford but cannot import as would cause cyclic import

class CliffordNoiseModel:
    # empty noise model
    def __init__(self):
        pass
    def append_noise(self, circ, gate):
        # circ is stim circuit
        pass


class DepolarizationModel(CliffordNoiseModel):
    def __init__(self):
        pass
    def get_gate_depolarization(self, gate):
        pass
    def append_noise(self, circ, gate):
        p = self.get_gate_depolarization(gate)
        if p is not None:
            circ.append(f"DEPOLARIZE{len(gate.qbs)}", gate.qbs, p)


class GateSpecificDepolarizationModel(DepolarizationModel):
    def __init__(
            self, 
            params: dict[str, dict[tuple[int, int], float | None]] = {}
        ):
        self.params = params
    def set_gate_depolarization(
            self, 
            gate_id: str,
            qbs: tuple[int] | tuple[int,int],
            p: float
        ):
        if not gate_id in self.params:
            self.params[gate_id] = {qbs: p}
        else:
            self.params[gate_id][qbs] = p
    def get_gate_depolarization(self, gate):
        gate_id = gate.get_stim_id()
        if not gate_id in self.params:
            return None
        else:
            return self.params[gate_id].get(gate.qbs)


class GateGeneralDepolarizationModel(DepolarizationModel):
    def __init__(
            self, 
            p1: float | None = None, 
            p2: float | None = None
        ):
        self.p1 = p1
        self.p2 = p2
    def get_gate_depolarization(self, gate):
        if len(gate.qbs) == 1:
            p = self.p1
        elif len(gate.qbs) == 2:
            p = self.p2
        else:
            raise Exception(f"weird number of qbs: {gate.qbs}")
        return p
    

class DecoherenceModel(CliffordNoiseModel):
    def __init__(self):
        pass
    def get_gate_time(self, gate):
        pass
    def _get_ps(self, time):
        T1 = self.T1 if self.T1 is not None else np.inf
        T2 = self.T2 if self.T2 is not None else np.inf
        pxy = (1 - np.exp(-time/T1))/4
        pz = (1 - np.exp(-time/T2))/2 - pxy
        return pxy, pz
    def get_gate_decoherence(self, gate):
        if self.T1 is None and self.T2 is None:
            return None
        else:
            time = self.get_gate_time(gate)
            return self._get_ps(time)
    def _append_noise(self, circ, qb, pxy, pz):
        circ.append("CORRELATED_ERROR", [stim.target_x(qb)], pxy)
        circ.append("ELSE_CORRELATED_ERROR", [stim.target_y(qb)], pxy / (1 - pxy))
        circ.append("ELSE_CORRELATED_ERROR", [stim.target_z(qb)], pz / (1 - 2*pxy))
    def append_noise(self, circ, gate, qb=None):
        if gate == "MEAS":
            assert qb is not None
            qbs = (qb, )
        else:
            qbs = gate.qbs
        dec = self.get_gate_decoherence(gate)
        if dec is not None:
            pxy, pz = dec
            for qb in qbs:
                self._append_noise(circ, qb, pxy, pz)
                

class GateSpecificDecoherenceModel(DecoherenceModel):
    def __init__(
            self, 
            T1: float | None = None,
            T2: float | None = None,
            params: dict[str, float] = {}
        ):
        self.T1 = T1
        self.T2 = T2
        self.params = params
    def set_gate_time(
            self, 
            gate_id: str,
            time: float
        ):
        self.params[gate_id] = time
    def get_gate_time(self, gate):
        if gate == "MEAS":
            gate_id = "MEAS"
        else:
            gate_id = gate.get_stim_id()
        if not gate_id in self.params:
            return 0.
        else:
            return self.params[gate_id]
        

class GateGeneralDecoherenceModel(DecoherenceModel):
    def __init__(
            self, 
            T1: float | None = None,
            T2: float | None = None,
            time1: float = 0.,
            time2: float = 0.,
            time_meas: float = 0.
        ):
        self.T1 = T1
        self.T2 = T2
        self.time1 = time1
        self.time2 = time2
        self.time_meas = time_meas
    def get_gate_time(self, gate):
        if gate == "MEAS":
            return self.time_meas
        if len(gate.qbs) == 1:
            time = self.time1
        elif len(gate.qbs) == 2:
            time = self.time2
        else:
            raise Exception(f"weird number of qbs: {gate.qbs}")
        return time