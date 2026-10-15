import sys, os, time, json
import numpy as np
import argparse
import qulacs
from qulacs import QuantumState, CompressedQuantumState
from qulacs.circuit import QuantumCircuitOptimizer

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'benchmark'))
from circuit_factory import get_circuit

if qulacs.check_build_for_mpi():
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    nranks = MPI.COMM_WORLD.Get_size()
else:
    print("Qulacs module was build without USE_MPI.")
    exit()

class RunResult:
    def __init__(self):
        self.times: dict = {}
        self.sv_MiB: float = 0.0
        self.fidelity: float | None = None

def has_not_supported_fused_swap(circuit: qulacs.QuantumCircuit, nranks: int) -> bool:
    global_qubits = int(np.log2(nranks))
    local_qubits = circuit.get_qubit_count() - global_qubits

    found = False

    for gate_idx in range(circuit.get_gate_count()):
        gate = circuit.get_gate(gate_idx)

        if gate.get_name() == "FusedSWAP":
            target_indices = gate.get_target_index_list()
            width = len(target_indices) // 2

            lower_indices = target_indices[:width]
            upper_indices = target_indices[width:]

            if all([idx < local_qubits for idx in lower_indices]) and \
               all([idx >= local_qubits for idx in upper_indices]):
                continue  # skip valid FusedSWAP gates between local and global qubits

            if all([idx < local_qubits for idx in target_indices]):
                # FusedSWAP gate on local qubits
                found = True
                continue

            if all([idx >= local_qubits for idx in target_indices]):
                # FusedSWAP gate on global qubits
                found = True
                continue

            # other cases
            found = True

    return found

def run_sim(simulator: str, circuit_name: str, nqubits: int, outdir: str | None = None, 
            append: bool = False,  draw_circuit: bool = False, 
            block_size: int | None = None, inner_size: int | None = None):

    def write_result(result: RunResult, outdir: str | None, outfile: str | None):
        overall_time = result.times["overall"]
        print(f"Simulation time: {overall_time:.3f} seconds")
        print(f"State vector size: {result.sv_MiB:.3f} MiB")
        if result.fidelity is not None:
            print(f"Fidelity: {result.fidelity:.6f}")

        if outdir is not None:            
            if not os.path.exists(outfile) or append:
                with open(outfile, 'a') as f:
                    json.dump(vars(result), f)
                    f.write('\n')
            print(outfile)
        else:
            print(vars(result))
        print()

    if simulator == "qompsim" or simulator == "fidelity":
        if inner_size is None:
            inner_size = 4
        if block_size is None:
            block_size = nqubits - max(int(np.log2(nranks)), 1) - inner_size - 1
            block_size = min(block_size, 26)  # limit maximum block size to 26

    circuit = get_circuit(circuit_name, nqubits)
    QCO = QuantumCircuitOptimizer(sv_block_size=block_size or 0)
    if nranks == 1:
        QCO.optimize(circuit, block_size=2)
        # QCO.optimize_light(circuit)
    else:
        QCO.optimize(circuit, block_size=2, swap_level=2)
        # QCO.optimize_light(circuit, swap_level=2)
    if draw_circuit and rank == 0:
        from qulacsvis import circuit_drawer
        circuit_drawer(circuit)

    if simulator != "qulacs" and has_not_supported_fused_swap(circuit, nranks):
        print("This circuit contains FusedSWAP gate(s), which is not supported in qompsim.")
        return

    qs = None
    cqs = None
    outfiles = {}
    state_vectors = {}

    if simulator == "qulacs" or simulator == "fidelity":
        if outdir is not None:
            outfiles['qulacs'] = f'{outdir}/qulacs_{circuit_name}_{nqubits}qubits_{nranks}ranks.json'
            if os.path.exists(outfiles['qulacs']) and not append:
                print(f"{outfiles['qulacs']} already exists. Use --append to append results.")
                return
        else:
            outfiles['qulacs'] = None
            
        qulacs_result = RunResult()    
        if rank == 0:
            print("======== Qulacs ========")

        MPI.COMM_WORLD.Barrier()
        ts_init = time.time()
        use_multi_cpu = True if nranks > 1 else False
        qs = QuantumState(nqubits, use_multi_cpu=use_multi_cpu)        
        MPI.COMM_WORLD.Barrier()
        te_init = time.time()

        MPI.COMM_WORLD.Barrier()
        ts_run = time.time()
        circuit.update_quantum_state(qs)
        MPI.COMM_WORLD.Barrier()
        te_run = time.time()
        
        if rank == 0:
            qulacs_result.times["init"] = te_init - ts_init          
            qulacs_result.times["overall"] = te_run - ts_run
            qulacs_result.times["fused_swap"] = qs.get_fused_swap_time()
            qulacs_result.sv_MiB = 2**(nqubits + 4) / (1024**2)
            write_result(qulacs_result, outdir, outfiles['qulacs'])

        state_vectors['qulacs'] = qs.get_vector() if simulator == "fidelity" else None


    if simulator == "qompsim" or simulator == "fidelity":
        assert block_size + inner_size + int(np.log2(nranks)) < nqubits, \
            f"{block_size=} and {inner_size=} are too large for the number of qubits and ranks."

        if outdir is not None:
            outfiles['qompsim'] = f'{outdir}/qompsim_{circuit_name}_{nqubits}qubits_{nranks}ranks_bs{block_size}_inner{inner_size}.json'
            if os.path.exists(outfiles['qompsim']) and not append:
                print(f"{outfiles['qompsim']} already exists. Use --append to append results.")
                return
        else:
            outfiles['qompsim'] = None
                
        # sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pysrc/BMQSim'))
        # from circuit_partitioning import partition_and_register_circuit
        qompsim_result = RunResult()
        if rank == 0:
            print("======== QompSim ========")
            print(f"block_size: {block_size}, inner_size: {inner_size}")

        MPI.COMM_WORLD.Barrier()
        ts_init = time.time()
        cqs = CompressedQuantumState(nqubits, block_size)
        # partition_and_register_circuit(cqs, block_size, inner_size, circuit, verbose=1)
        cqs.partition_and_register_circuit(circuit, inner_size, verbose=1)
        MPI.COMM_WORLD.Barrier()
        te_init = time.time()

        MPI.COMM_WORLD.Barrier()
        ts_run = time.time()
        cqs.apply_circuit()
        MPI.COMM_WORLD.Barrier()
        te_run = time.time()

        if rank == 0:
            qompsim_result.times["init"] = te_init - ts_init
            qompsim_result.times["overall"] = te_run - ts_run
            total_times = cqs.get_total_times()
            for name, t in total_times.items():
                qompsim_result.times[name] = t
            qompsim_result.sv_MiB = cqs.get_max_SV_bytes() / (1024**2)
            if simulator == "qompsim":
                write_result(qompsim_result, outdir, outfiles['qompsim'])

        if simulator == "fidelity":
            if rank == 0:
                print("Calculating fidelity between Qulacs and QompSim...")
            state_vectors["qompsim"] = cqs.get_vector()
            assert len(state_vectors["qulacs"]) == len(state_vectors["qompsim"]), \
                "Length mismatch between Qulacs and QompSim state vectors"
            if rank == 0:
                print(" Qulacs  sv[0]: ", state_vectors['qulacs'][0])
                print("QompSim  sv[0]: ", state_vectors["qompsim"][0])
            if rank == nranks - 1:
                print(" Qulacs sv[-1]: ", state_vectors['qulacs'][-1])
                print("QompSim sv[-1]: ", state_vectors["qompsim"][-1])      

            # Optimized fidelity calculation using vdot
            local_inner_product = np.vdot(state_vectors["qulacs"], state_vectors["qompsim"])
            total_inner_product = MPI.COMM_WORLD.reduce(local_inner_product, op=MPI.SUM, root=0)
            if rank == 0:                
                fidelity = np.abs(total_inner_product)
                qompsim_result.fidelity = float(fidelity)
                write_result(qompsim_result, outdir, outfiles["qompsim"])
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulator", type=str, choices=["qulacs", "qompsim", "fidelity"],
                       help="Name of simulator (qulacs or qompsim), or fidelity calc mode")
    parser.add_argument("-c", "--circuit_name", type=str, 
                        choices=["hadamard", "seq_hadamard", "ghz", "qft", "ising", "hea",
                                 "grover", "cc", "bv", "qsvm", "qaoa", "wstate"],
                        help="Name of gates to use")
    parser.add_argument("-n", "--nqubits", type=int, help="Number of qubits")
    parser.add_argument("-o", "--outdir", type=str, default=None,
                        help="Output directory for result file")
    parser.add_argument("-a", "--append", action='store_true', default=False, 
                        help="Append to existing result file (default: False)")
    parser.add_argument("-d", "--draw_circuit", action='store_true', default=False, 
                        help="Draw the quantum circuit (default: False)")
    parser.add_argument("-b", "--block_size", type=int, default=None, 
                        help="Block size for qompsim (default: None)")
    parser.add_argument("-i", "--inner_size", type=int, default=None, 
                        help="Inner size for qompsim (default: None)")
    
    args = parser.parse_args()
    kwargs = args.__dict__
    run_sim(**kwargs)