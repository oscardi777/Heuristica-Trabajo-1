import os
import bisect
import math
import time
import pandas as pd

# ─────────────────────────────────────────────
# PARÁMETROS - AJUSTADOS PARA VELOCIDAD
# ─────────────────────────────────────────────
INSTANCES_DIR = "NWJSSP Instances"
OUTPUT_FILE   = "resultados/NWJSSP_OADG_NEH(Constructivo).xlsx"

TIME_LIMIT_PER_BLOCK = 0.015     # ← más estricto que antes

INSTANCES = [
    "ft06.txt", "ft06r.txt", "ft10.txt", "ft10r.txt", "ft20.txt", "ft20r.txt",
    "tai_j10_m10_1.txt", "tai_j10_m10_1r.txt",
    "tai_j100_m10_1.txt", "tai_j100_m10_1r.txt",
    "tai_j100_m100_1.txt", "tai_j100_m100_1r.txt",
    "tai_j1000_m10_1.txt", "tai_j1000_m10_1r.txt",
    "tai_j1000_m100_1.txt", "tai_j1000_m100_1r.txt",
]


# ─────────────────────────────────────────────
# ESTRUCTURAS (igual)
# ─────────────────────────────────────────────
class Operation:
    def __init__(self, machine, processing_time):
        self.machine = machine
        self.p = processing_time

class Job:
    def __init__(self, operations, release):
        self.operations = operations
        self.release = release

class Machine:
    __slots__ = ("id", "begins", "ends", "max_end_prefix")

    def __init__(self, id: int):
        self.id = id
        self.begins: list[int] = []
        self.ends: list[int] = []
        self.max_end_prefix: list[int] = []

    def add(self, b: int, e: int) -> None:
        idx = bisect.bisect_right(self.begins, b)
        self.begins.insert(idx, b)
        self.ends.insert(idx, e)
        if idx == len(self.begins) - 1:
            prev = self.max_end_prefix[idx - 1] if idx > 0 else 0
            self.max_end_prefix.append(max(prev, e))
        else:
            self.max_end_prefix = []
            running = 0
            for ek in self.ends:
                running = max(running, ek)
                self.max_end_prefix.append(running)

    def max_end_before(self, threshold: int) -> int:
        idx = bisect.bisect_left(self.begins, threshold)
        return self.max_end_prefix[idx - 1] if idx > 0 else 0


# ─────────────────────────────────────────────
# LECTURA Y OFFSETS PRECOMPUTADOS
# ─────────────────────────────────────────────
def read_instance(file: str):
    with open(file) as f:
        n, m = map(int, f.readline().split())
        jobs = []
        for _ in range(n):
            data = list(map(int, f.readline().split()))
            operations = [Operation(data[2*i], data[2*i+1]) for i in range(m)]
            jobs.append(Job(operations, data[-1]))
    return jobs, m

def precompute_offsets(jobs):
    return [[0] * len(job.operations) if len(job.operations) <= 1 else 
            [0] + [sum(op.p for op in job.operations[:u+1]) for u in range(len(job.operations)-1)]
            for job in jobs]


# ─────────────────────────────────────────────
# find_start y schedule_job (mismos que V4 + offsets)
# ─────────────────────────────────────────────
def find_start(job, machines, offsets):
    start = job.release
    while True:
        max_candidate = start
        feasible = True
        for u, op in enumerate(job.operations):
            b_op = start + offsets[u]
            e_op = b_op + op.p
            max_ek = machines[op.machine].max_end_before(e_op)
            if max_ek > b_op:
                feasible = False
                candidate = max_ek - offsets[u]
                max_candidate = max(max_candidate, candidate)
        if feasible:
            return start
        start = max_candidate


def schedule_job(job, machines, job_id, schedule, offsets):
    start = find_start(job, machines, offsets)
    completion = 0
    for u, op in enumerate(job.operations):
        begin = start + offsets[u]
        finish = begin + op.p
        machines[op.machine].add(begin, finish)
        if schedule is not None:
            schedule.append({"job": job_id, "machine": machines[op.machine].id,
                             "operation": u, "start": begin, "finish": finish})
        completion = finish
    return completion


# evaluate_sequence y evaluate_insertion (igual que V5)
def evaluate_sequence(sequence, jobs, m, offsets_list, save_schedule=False):
    machines = [Machine(i) for i in range(m)]
    total_flow = 0
    schedule = [] if save_schedule else None
    for j in sequence:
        total_flow += schedule_job(jobs[j], machines, j, schedule, offsets_list[j])
    return (total_flow, schedule) if save_schedule else total_flow


def evaluate_insertion(sequence, j, pos, jobs, m, offsets_list):
    machines = [Machine(i) for i in range(m)]
    total_flow = 0
    for idx in range(pos):
        total_flow += schedule_job(jobs[sequence[idx]], machines, sequence[idx], None, offsets_list[sequence[idx]])
    total_flow += schedule_job(jobs[j], machines, j, None, offsets_list[j])
    for idx in range(pos, len(sequence)):
        total_flow += schedule_job(jobs[sequence[idx]], machines, sequence[idx], None, offsets_list[sequence[idx]])
    return total_flow


# find_best_insertion con early termination más agresivo
def find_best_insertion(sequence, j, jobs, m, block_size, time_limit, offsets_list):
    n_pos = len(sequence) + 1
    best_pos = 0
    best_value = float("inf")

    no_improve_limit = max(5, int(math.sqrt(len(sequence) + 10)))   # más agresivo
    no_improve_streak = 0

    pos = 0
    while pos < n_pos:
        end_block = min(pos + block_size, n_pos)
        t_bloque = time.time()

        for p in range(pos, end_block):
            if time.time() - t_bloque > time_limit:
                break
            value = evaluate_insertion(sequence, j, p, jobs, m, offsets_list)
            if value < best_value:
                best_value = value
                best_pos = p
                no_improve_streak = 0
            else:
                no_improve_streak += 1
                if no_improve_streak >= no_improve_limit:
                    return best_pos, best_value

        pos = end_block
    return best_pos, best_value


# construct_solution
def construct_solution(jobs, m):
    n = len(jobs)
    offsets_list = precompute_offsets(jobs)

    block_size = max(30, int(n ** 0.55))        # ← clave: más grande para n=100

    order = sorted(range(n), key=lambda j: jobs[j].release + sum(op.p for op in jobs[j].operations), reverse=True)

    sequence = []
    for j_idx, j in enumerate(order):
        best_pos, _ = find_best_insertion(sequence, j, jobs, m, block_size, TIME_LIMIT_PER_BLOCK, offsets_list)
        sequence.insert(best_pos, j)

    return sequence


# write_results_to_excel y main (igual que antes)
def write_results_to_excel(results, output_file):
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    writer_kwargs = dict(engine="openpyxl", mode="a", if_sheet_exists="replace") if os.path.exists(output_file) else dict(engine="openpyxl", mode="w")
    with pd.ExcelWriter(output_file, **writer_kwargs) as writer:
        for sheet_name, (total_flow, compute_time_ms, job_start_times) in results.items():
            df = pd.DataFrame([[total_flow, compute_time_ms], job_start_times])
            df.to_excel(writer, sheet_name=sheet_name, header=False, index=False)
    print(f"\nResultados guardados en: {output_file}")


def main():
    results = {}
    for inst in INSTANCES:
        filepath = os.path.join(INSTANCES_DIR, inst)
        if not os.path.exists(filepath):
            print(f"[SKIP] {inst}")
            continue

        jobs, m = read_instance(filepath)
        n = len(jobs)

        t0 = time.time()
        sequence = construct_solution(jobs, m)
        offsets_list = precompute_offsets(jobs)
        total_flow, schedule = evaluate_sequence(sequence, jobs, m, offsets_list, save_schedule=True)
        compute_time_ms = round((time.time() - t0) * 1000)

        job_start_times = [None] * n
        for op in schedule:
            if op["operation"] == 0:
                job_start_times[op["job"]] = op["start"]

        sheet_name = inst.replace(".txt", "")
        results[sheet_name] = (total_flow, compute_time_ms, job_start_times)
        print(f"[OK] {inst:<30} Z={total_flow:>10}  tiempo={compute_time_ms:>6} ms")

    if results:
        write_results_to_excel(results, OUTPUT_FILE)


if __name__ == "__main__":
    main()