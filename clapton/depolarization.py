# gate objects are ParametrizedClifford but cannot import as would cause cyclic import

class DepolarizationModel:
    def __init__(self):
        pass
    def get_gate_depolarization(self, gate):
        pass


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
        self.params[gate_id][qbs] = p
    def get_gate_depolarization(self, gate):
        gate_id = gate.get_stim_id()
        if not gate_id in self.params:
            return None
        else:
            return self.params[gate_id][gate.qbs]


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