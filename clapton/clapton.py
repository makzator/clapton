import numpy as np
import pygad
import multiprocessing as mp
from clapton.clifford import ParametrizedCliffordCircuit
from clapton.evaluation import transform_paulis, get_energy, weighted_relative_pauli_weight
from clapton.utils import n_to_dits
from clapton.mp_helpers import SignalHandler


### Clapton
def loss_func(
        x: list[int], 
        paulis: list[str], 
        coeffs: list[float], 
        vqe_pcirc: ParametrizedCliffordCircuit, 
        trans_pcirc: ParametrizedCliffordCircuit | None = None, 
        alpha: float | None = None, 
        return_sublosses: bool = False, 
        **energy_kwargs
    ):
    if trans_pcirc is None:
        vqe_pcirc.assign(x)
        vqe_pcirc.snapshot()
        vqe_pcirc.snapshot_noiseless()
        energy = get_energy(
                    vqe_pcirc, 
                    paulis, 
                    coeffs, 
                    **energy_kwargs
                    )
        energy_noiseless = get_energy(
                            vqe_pcirc, 
                            paulis, 
                            coeffs, 
                            get_noiseless=True,
                            **energy_kwargs
                            )
        pauli_weight_loss = 0.
        loss = energy + energy_noiseless
    else:
        trans_circ = trans_pcirc.assign(x).stim_circuit()
        paulis_trans, signs = transform_paulis(trans_circ, paulis)
        coeffs_trans = np.multiply(signs, coeffs)
        # assume vqe_pcirc has stim circuit snapshot with all 0 parameters
        energy = get_energy(
                    vqe_pcirc, 
                    paulis_trans, 
                    coeffs_trans, 
                    **energy_kwargs
                    )
        energy_noiseless = get_energy(
                            vqe_pcirc, 
                            paulis_trans, 
                            coeffs_trans, 
                            get_noiseless=True,
                            **energy_kwargs
                            )
        if alpha is not None:
            pauli_weight_loss = alpha * weighted_relative_pauli_weight(paulis_trans, np.abs(coeffs))
        else:
            pauli_weight_loss = 0.
        loss = energy + energy_noiseless + pauli_weight_loss
    if return_sublosses:
        return loss, energy, energy_noiseless, pauli_weight_loss
    else:
        return loss
    

def eval_xs_terms(
        xs: list[list[int]], 
        paulis: list[str], 
        coeffs: list[float],
        vqe_pcirc: ParametrizedCliffordCircuit,
        trans_pcirc: ParametrizedCliffordCircuit | None = None,
        p_start_idx: int = 0, 
        p_end_idx: int | None = None, 
        result_queue = None,
        result_id: int | None = None,
        **loss_kwargs
    ):
    S = len(xs)
    P = len(paulis)
    if p_end_idx is None:
        p_end_idx = P - 1
    idx1 = p_start_idx
    idx2 = P - 1
    partial_losses = []
    for s in range(S-1):
        partial_losses.append(loss_func(
            xs[s], 
            paulis[idx1:idx2+1], 
            coeffs[idx1:idx2+1], 
            vqe_pcirc,
            trans_pcirc,
            **loss_kwargs
            ))
        idx1 = 0
    idx2 = p_end_idx
    partial_losses.append(loss_func(
            xs[S-1], 
            paulis[idx1:idx2+1], 
            coeffs[idx1:idx2+1], 
            vqe_pcirc,
            trans_pcirc,
            **loss_kwargs
            ))
    if result_queue is None:
        return partial_losses
    else:
        result_queue.put((result_id, partial_losses))


def handle_out_data(
        x: list[int], 
        losses: list[float], 
        out_data: list | None = None, 
        callback = None
    ):
        if out_data is not None:
            out_data[0] += 1
            out_data[1] = losses
            out_data[2] = x
            if callback is not None:
                callback(out_data)


def loss_func_mp(
        xs: list[list[int]], 
        paulis: list[str], 
        coeffs: list[float],
        vqe_pcirc: ParametrizedCliffordCircuit,
        trans_pcirc: ParametrizedCliffordCircuit | None = None,
        n_proc: int = 1, 
        out_data: list | None = None, 
        callback = None, 
        **loss_kwargs
    ):
    S = len(xs)
    P = len(paulis)
    SP = S * P
    ntasks_per_P = int(np.ceil(SP / n_proc))
    sp_start_idc = [n_to_dits(c*ntasks_per_P, [S, P]) for c in range(n_proc)] # c is process / core idx
    sp_end_idc = [n_to_dits((c+1)*ntasks_per_P-1, [S, P]) for c in range(n_proc-1)]
    sp_end_idc.append(np.array([S-1, P-1], dtype=int))
    processes = []
    result_queue = mp.Manager().Queue()
    loss_kwargs["return_sublosses"] = True
    # start n_proc - 1 other subprocesses
    for i in range(1, n_proc):
        process = mp.Process(
            target=eval_xs_terms,
            args=(
                xs[sp_start_idc[i][0]:sp_end_idc[i][0]+1],
                paulis,
                coeffs,
                vqe_pcirc,
                trans_pcirc,
                sp_start_idc[i][1],
                sp_end_idc[i][1],
                result_queue,
                i
            ),
            kwargs=loss_kwargs
        )
        processes.append(process)
        process.start()

    partial_losses = eval_xs_terms(
        xs[sp_start_idc[0][0]:sp_end_idc[0][0]+1],
        paulis,
        coeffs,
        vqe_pcirc,
        trans_pcirc,
        sp_start_idc[0][1],
        sp_end_idc[0][1],
        **loss_kwargs
    )
    losses = np.zeros((S, len(partial_losses[0])))
    ss = range(sp_start_idc[0][0], sp_end_idc[0][0]+1)
    for s_idx, s in enumerate(ss):
        losses[s] += partial_losses[s_idx]

    # Block until others finished
    for process in processes:
        process.join()

    while not result_queue.empty():
        item = result_queue.get()
        i = item[0]
        partial_losses = item[1]
        ss = range(sp_start_idc[i][0], sp_end_idc[i][0]+1)
        for s_idx, s in enumerate(ss):
            losses[s] += partial_losses[s_idx]
    best_idx = np.argmin(losses[:,0])
    handle_out_data(xs[best_idx], losses[best_idx], out_data, callback)
    return losses[:,0]


def claptonize(
        paulis: list[str],
        coeffs: list[float],
        vqe_pcirc: ParametrizedCliffordCircuit,
        trans_pcirc: ParametrizedCliffordCircuit | None = None,
        n_proc: int = 10,
        n_starts: int = 10,
        n_rounds: int | None =None,
        n_retry_rounds: int = 0,
        return_n_rounds: bool = False,
        mix_best_pop_frac: float = 0.2,
        **optimizer_and_loss_kwargs
    ):
    sig_handler = SignalHandler()

    assert vqe_pcirc.num_physical_qubits == len(paulis[0])
    if trans_pcirc is not None:
        assert trans_pcirc.num_physical_qubits == len(paulis[0])
        # take snapshot for more efficient sim in cost function (is initialized to params all 0)
        vqe_pcirc.snapshot()
        vqe_pcirc.snapshot_noiseless()
    
    n_proc = n_proc // n_starts
    if n_proc == 0:
        n_proc = 1
    initial_populations = [None] * n_starts
    out_data = [-1, [np.inf]*3, None]
    optimizer_and_loss_kwargs["n_proc"] = n_proc
    optimizer_and_loss_kwargs["return_best_pop_frac"] = mix_best_pop_frac
    optimizer_and_loss_kwargs["out_data"] = out_data
    
    r_idx = 0
    r_idx_last_change = 0
    last_best_energy_ideal = np.inf
    while True:
        print(f"STARTING ROUND {r_idx}\n\n")
        # start parallelization
        master_processes = []
        master_queue = mp.Manager().Queue()
        # start n_starts - 1 other master processes
        for m in range(1, n_starts):
            optimizer_and_loss_kwargs["initial_population"] = initial_populations[m]
            master_process = mp.Process(
                                target=genetic_algorithm, 
                                args=(
                                    paulis,
                                    coeffs,
                                    vqe_pcirc,
                                    trans_pcirc,
                                    master_queue,
                                    m
                                ),
                                kwargs=optimizer_and_loss_kwargs)
            master_processes.append(master_process)
            master_process.start()

        # this is also a master process
        optimizer_and_loss_kwargs["initial_population"] = initial_populations[0]
        xs, losses = genetic_algorithm(
            paulis,
            coeffs,
            vqe_pcirc,
            trans_pcirc,
            **optimizer_and_loss_kwargs
        )
        best_count = len(xs)

        # wait until others are finished
        for master_process in master_processes:
            master_process.join()
        # fetch others
        while not master_queue.empty():
            item = master_queue.get()
            xs = np.vstack((xs, item[1]))
            losses = np.concatenate((losses, item[2]))
        num_xs = xs.shape[0]
        assert num_xs == n_starts * best_count
        
        # create new initial populations for next round
        rand_shuffled_idc = np.random.choice(range(num_xs), size=num_xs, replace=False)
        for i in range(n_starts):
            idc = rand_shuffled_idc[i*best_count:(i+1)*best_count]
            initial_populations[i] = xs[idc]

        best_idx = np.argmin(losses)
        x_best = xs[best_idx]

        _, energy_noisy, energy_ideal, _ = loss_func(
                                                x_best, 
                                                paulis, 
                                                coeffs, 
                                                vqe_pcirc, 
                                                trans_pcirc,
                                                alpha=optimizer_and_loss_kwargs.get("alpha"),
                                                return_sublosses=True
                                                )

        if n_rounds is None:
            if energy_ideal < last_best_energy_ideal:
                r_idx_last_change = r_idx
                last_best_energy_ideal = energy_ideal
                r_idx += 1
            else:
                if r_idx == r_idx_last_change + 1 + n_retry_rounds:
                    # no change within n_retry_rounds
                    r_idx += 1
                    break
                else:
                    r_idx += 1
        else:
            r_idx += 1
            if r_idx == n_rounds:
                break
    
    sig_handler.restore_handlers()
    if return_n_rounds:
        return list(x_best), energy_noisy, energy_ideal, r_idx
    else:
        return list(x_best), energy_noisy, energy_ideal


### Solvers
def genetic_algorithm(
        paulis: list[str], 
        coeffs: list[str],
        vqe_pcirc: ParametrizedCliffordCircuit,
        trans_pcirc: ParametrizedCliffordCircuit | None,
        master_queue = None,
        master_id: int | None = None,
        n_proc: int = 1,
        out_data: list | None = None,
        callback = None,
        budget: int = 100,
        population_size: int = 100,
        return_best_pop_frac: int = 0.2,
        initial_population: np.ndarray = None,
        init_no_2qb: bool = True,
        keep_elitism: bool = None,
        num_parents_mating: int = None,
        parent_selection_type: str = "tournament", #"sss"
        keep_parents: int = -1,
        crossover_type: str = "single_point",
        crossover_probability: float = 0.9,
        mutation_type: str = "adaptive",
        mutation_probability: tuple[float, float] =(0.25, 0.01), #(0.25, 0.05)  
        **loss_kwargs
    ):
    print(f"started GA at id {master_id} with {n_proc} procs\n")
    if trans_pcirc is None:
        gene_space = vqe_pcirc.parameter_space()
        idc_param_2qb = vqe_pcirc.idc_param_2qb()
    else:
        gene_space = trans_pcirc.parameter_space()
        idc_param_2qb = trans_pcirc.idc_param_2qb()
    num_params = len(gene_space)
    num_generations = budget
    num_genes = num_params
    if keep_elitism is None:
        keep_elitism = population_size // 10
    if num_parents_mating is None:
        num_parents_mating = 2 * population_size // 10
    best_count = int(population_size * return_best_pop_frac)

    def fitness_func(ga_instance, solutions, solutions_idc):
        return -loss_func_mp(
            solutions, 
            paulis,
            coeffs,
            vqe_pcirc,
            trans_pcirc,
            n_proc,
            out_data,
            callback,
            **loss_kwargs
            )
    ga_instance = pygad.GA(
                    num_generations=num_generations,
                    num_parents_mating=num_parents_mating,
                    fitness_func=fitness_func,
                    sol_per_pop=population_size,
                    num_genes=num_genes,
                    parent_selection_type=parent_selection_type,
                    keep_parents=keep_parents,
                    crossover_type=crossover_type,
                    mutation_type=mutation_type,
                    gene_space=gene_space,
                    gene_type=[int]*num_params,
                    crossover_probability=crossover_probability,
                    mutation_probability=mutation_probability,
                    keep_elitism=keep_elitism,
                    fitness_batch_size=population_size
                    )
    if initial_population is not None:
        initial_population = np.asarray(initial_population)
        assert len(initial_population.shape) == 2
        assert initial_population.shape[1] == num_params
        num_fixed_pops = initial_population.shape[0]
        ga_instance.initial_population[:num_fixed_pops] = initial_population[:population_size]
        ga_instance.population[:num_fixed_pops] = initial_population[:population_size].copy()
    else:
        if init_no_2qb:
            ga_instance.initial_population[:,idc_param_2qb] = 0 
            ga_instance.population[:,idc_param_2qb] = 0 
    
    ga_instance.run()
    last_losses = -ga_instance.last_generation_fitness
    best_idc = np.argsort(last_losses)[:best_count]
    best_losses = last_losses[best_idc]
    best_xs = ga_instance.population[best_idc,:]

    if master_queue is None:
        return best_xs, best_losses
    else:
        master_queue.put((master_id, best_xs, best_losses))