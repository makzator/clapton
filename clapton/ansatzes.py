from clapton.clifford import ParametrizedCliffordCircuit


### Ansatzes
def linear_ansatz(N, reps=1, fix_2q=False):
    pcirc = ParametrizedCliffordCircuit()
    for _ in range(reps):
        for i in range(N):
            pcirc.RY(i)
        for i in range(N):
            pcirc.RZ(i)
        for i in range(N-1):
            control = i
            target = i+1
            if fix_2q:
                pcirc.Q2(control, target).fix(1)
            else:
                pcirc.Q2(control, target)
    for i in range(N):
        pcirc.RY(i)
    for i in range(N):
        pcirc.RZ(i)
    return pcirc


def circular_ansatz(N, reps=1, fix_2q=False):
    pcirc = ParametrizedCliffordCircuit()
    for _ in range(reps):
        for i in range(N):
            pcirc.RY(i)
        for i in range(N):
            pcirc.RZ(i)
        for i in range(N):
            control = (i-1) % N
            target = i
            if fix_2q:
                pcirc.Q2(control, target).fix(1)
            else:
                pcirc.Q2(control, target)
    for i in range(N):
        pcirc.RY(i)
    for i in range(N):
        pcirc.RZ(i)
    return pcirc


def circular_ansatz_mirrored(N, reps=1, fix_2q=False):
    pcirc = ParametrizedCliffordCircuit()
    for _ in range(reps):
        for i in range(N):
            pcirc.RY(i)
        for i in range(N):
            pcirc.RZ(i)
        for i in range(N):
            control = (i-1) % N
            target = i
            if fix_2q:
                pcirc.Q2(control, target).fix(1)
            else:
                pcirc.Q2(control, target)
        for i in range(N):
            pcirc.RY(i)
        for i in range(N):
            pcirc.RZ(i)
        for i in range(N-1, -1, -1):
            control = (i-1) % N
            target = i
            if fix_2q:
                pcirc.Q2(control, target).fix(1)
            else:
                pcirc.Q2(control, target)
    for i in range(N):
        pcirc.RY(i)
    for i in range(N):
        pcirc.RZ(i)
    return pcirc  


def full_ansatz(N, reps=1, fix_2q=False):
    pcirc = ParametrizedCliffordCircuit()
    for _ in range(reps):
        for i in range(N):
            pcirc.RY(i)
        for i in range(N):
            pcirc.RZ(i)
        for i in range(N-1):
            for j in range(i+1, N):
                if fix_2q:
                    pcirc.Q2(i, j).fix(1)
                else:
                    pcirc.Q2(i, j)
    for i in range(N):
        pcirc.RY(i)
    for i in range(N):
        pcirc.RZ(i)
    return pcirc  


def full_ansatz_C1(N, reps=1):
    pcirc = ParametrizedCliffordCircuit()
    for _ in range(reps):
        for i in range(N):
            pcirc.C1(i)
        for i in range(N-1):
            for j in range(i+1, N):
                pcirc.Q2(i, j)
    for i in range(N):
        pcirc.C1(i)
    return pcirc  


def ansatz_from_instructions(
        instructions: list[tuple[str, list[int], bool, int]], 
        meas_map_dict=None
    ):
    """
    instructions: [(lbl, [qbs], is_parametrized, param)]
    lbl: RX, RY, RZ, 2Q
    is_parametrized: True, False
    param: if is_parametrized: parameter index in original input
                         else: fixed parameter
    """
    def ansatz(N=None):
        pcirc = ParametrizedCliffordCircuit()
        param_map = {}
        param_idx = 0
        for ins in instructions:
            if ins[0] == "RX":
                gate = pcirc.RX(*ins[1])
            elif ins[0] == "RY":
                gate = pcirc.RY(*ins[1])
            elif ins[0] == "RZ":
                gate = pcirc.RZ(*ins[1])
            else:
                gate = pcirc.Q2(*ins[1])
            if ins[2]:
                # parametrized gate
                param_map[param_idx] = ins[3]
                param_idx += 1
            else:
                gate.fix(ins[3])
        pcirc.define_parameter_map(param_map)
        pcirc.define_measurement_map(meas_map_dict)
        return pcirc
    return ansatz