import os
import math
import random
import time
import pandas as pd


# ─────────────────────────────────────────────
# PARÁMETROS
# ─────────────────────────────────────────────
INSTANCES_DIR = "NWJSSP Instances"
OUTPUT_FILE   = "resultados\\NWJSSP_OADG_NEH(SimpleNoise).xlsx"

R = 0.15  # Ruido: fracción del tiempo total de procesamiento mínimo (±15%)
N_ITER = 10  # Construcciones independientes por instancia
SEED = 42  # Semilla para reproducibilidad
TIME_LIMIT_PER_BLOCK = 0.01 # Tiempo máximo en segundos por bloque de posiciones

INSTANCES = [
    "ft06.txt",           "ft06r.txt",
    "ft10.txt",           "ft10r.txt",
    "ft20.txt",           "ft20r.txt",
    "tai_j10_m10_1.txt",    "tai_j10_m10_1r.txt",
    "tai_j100_m10_1.txt",   "tai_j100_m10_1r.txt",
    "tai_j100_m100_1.txt",  "tai_j100_m100_1r.txt",
    "tai_j1000_m10_1.txt",  "tai_j1000_m10_1r.txt",
    "tai_j1000_m100_1.txt", "tai_j1000_m100_1r.txt"
]


# ─────────────────────────────────────────────
# ESTRUCTURAS DE DATOS
# ─────────────────────────────────────────────
class Operation:
    def __init__(self, machine, processing_time):
        self.machine = machine
        self.p = processing_time

class Job:
    def __init__(self, operations, release):
        self.operations = operations
        self.release = release


# ─────────────────────────────────────────────
# LECTURA DE INSTANCIAS
# ─────────────────────────────────────────────
def read_instance(file):
    """
    Lee una instancia NWJSSP desde archivo de texto, para extraer los datos de la instancia.
    Asumiendo formato del Anexo 2
    """
    with open(file) as f:
        n, m = map(int, f.readline().split())
        jobs = []
        for _ in range(n):
            data = list(map(int, f.readline().split()))
            operations = [
                Operation(data[2*i], data[2*i + 1])
                for i in range(m)
            ]
            jobs.append(Job(operations, release=data[-1]))
            
    return jobs, m

# ─────────────────────────────────────────────
# OFFSETS NO-WAIT
# ─────────────────────────────────────────────
def compute_offsets(job):
    """
    Calcula el desplazamiento acumulado de cada operación respecto al
    inicio del trabajo. offset[u] = sum de tiempos de op. 0..u-1.
    Bajo No-Wait, la operación u comienza exactamente en start + offset[u].
    """
    offsets = [0] * len(job.operations)
    total = 0
    for u, op in enumerate(job.operations[:-1]):
        total += op.p
        offsets[u + 1] = total

    return offsets

# ─────────────────────────────────────────────
# INICIO FACTIBLE MÍNIMO BAJO NO-WAIT
# ─────────────────────────────────────────────
def find_start(job, machine_available, offsets):
    """
    Calcula el inicio factible mínimo bajo la restricción No-Wait:
    start = max(r_j, max_u{ machine_available[machine_u] - offset[u] })
    """
    start = job.release
    for u, op in enumerate(job.operations):
        required = machine_available[op.machine] - offsets[u]
        if required > start:
            start = required

    return start

# ─────────────────────────────────────────────
# PROGRAMAR UN TRABAJO
# ─────────────────────────────────────────────
def schedule_job(job, machine_available, job_id, schedule):
    """
    Inserta job_id en la programación actual.
    Actualiza machine_available y, si schedule no es None, registra ops.
    Retorna el tiempo de finalización del trabajo (Cj).
    """
    offsets = compute_offsets(job)
    start = find_start(job, machine_available, offsets)
    completion = 0
    for u, op in enumerate(job.operations):
        begin  = start + offsets[u]
        finish = begin + op.p
        machine_available[op.machine] = finish
        if schedule is not None:
            schedule.append({
                "job": job_id, "machine": op.machine,
                "operation": u, "start": begin, "finish": finish
            })
        completion = finish

    return completion

# ─────────────────────────────────────────────
# EVALUAR SECUENCIA COMPLETA
# ─────────────────────────────────────────────
def evaluate_sequence(sequence, jobs, m, save_schedule=False):
    """
    Evalúa una secuencia completa y calcula el Total Flow Time (suma Cj).
    Si save_schedule=True devuelve también el schedule completo.
    """
    machine_available = [0] * m
    total_flow = 0
    schedule = [] if save_schedule else None
    for j in sequence:
        Cj = schedule_job(jobs[j], machine_available, j, schedule)
        total_flow += Cj

    return (total_flow, schedule) if save_schedule else total_flow


# ─────────────────────────────────────────────
# EVALUAR INSERCIÓN
# ─────────────────────────────────────────────
def evaluate_insertion(sequence, j, pos, jobs, m):
    """
    Evalúa el Total Flow Time de insertar j en la posición pos de
    sequence, recorriendo los tres segmentos sin construir lista temporal:
        sequence[0..pos-1]  →  j  →  sequence[pos..end]
    """
    machine_available = [0] * m
    total_flow = 0

    for idx in range(pos):
        total_flow += schedule_job(
            jobs[sequence[idx]], machine_available, sequence[idx], None
        )

    total_flow += schedule_job(jobs[j], machine_available, j, None)

    for idx in range(pos, len(sequence)):
        total_flow += schedule_job(
            jobs[sequence[idx]], machine_available, sequence[idx], None
        )

    return total_flow

# ─────────────────────────────────────────────
# MEJOR POSICIÓN CON BÚSQUEDA POR BLOQUES TEMPORIZADOS
# ─────────────────────────────────────────────
def find_best_insertion(sequence, j, jobs, m, block_size, time_limit):
    """
    Encuentra la mejor posición para insertar j en sequence usando
    búsqueda por bloques con límite de tiempo por bloque:
      - La secuencia se divide en bloques consecutivos de tamaño block_size.
      - Cada bloque tiene un time_limit segundos.
        Si se agota el tiempo dentro del bloque, se interrumpe ese bloque
        y se avanza al siguiente.
      - Al finalizar se retorna la mejor posición encontrada en cualquier bloque.
    """
    n_pos = len(sequence) + 1
    best_pos = 0
    best_value = float("inf")

    pos = 0
    while pos < n_pos:
        end_block = min(pos + block_size, n_pos)
        t_bloque = time.time()

        for p in range(pos, end_block):
            # Tiempo agotado en este bloque → saltar al siguiente
            if time.time() - t_bloque > time_limit:
                break
            value = evaluate_insertion(sequence, j, p, jobs, m)
            if value < best_value:
                best_value = value
                best_pos = p

        pos = end_block

    return best_pos, best_value


# ─────────────────────────────────────────────
# ORDENAMIENTO CON RUIDO
# ─────────────────────────────────────────────
def noisy_order(jobs, r):
    """
    Genera el orden inicial NEH con ruido aditivo uniforme en los tiempos de procesamiento.
    Para cada trabajo j:
        w_j       = r_j + sum(p_ju)
        ρ_j       ~ U(-r·w_j, +r·w_j)
        w_j^ruido = w_j + ρ_j
    Se ordena de mayor a menor por w_j^ruido. Con r=0 el resultado
    es idéntico al ordenamiento NEH puro (sin aleatoriedad).
    """
    weights = []
    for j, job in enumerate(jobs):
        w = job.release + sum(op.p for op in job.operations)
        noise = random.uniform(-r * w, r * w)
        weights.append((j, w + noise))
    weights.sort(key=lambda x: x[1], reverse=True)

    return [j for j, _ in weights]

# ─────────────────────────────────────────────
# CONSTRUCCIÓN NEH CON RUIDO — una iteración
# ─────────────────────────────────────────────
def construct_noisy_solution(jobs, m, r, block_size, time_limit):
    """
    Una iteración de NEH con ruido:
    1. Ordena con pesos perturbados por ruido uniforme proporcional.
    2. Inserta cada trabajo en la mejor posición via búsqueda por bloques
       con límite de tiempo por bloque.
    El ruido afecta solo al ordenamiento inicial; la inserción es el mismo metodo que con NEH.
    """
    order = noisy_order(jobs, r)
    sequence = []

    for j in order:
        best_pos, _ = find_best_insertion(
            sequence, j, jobs, m, block_size, time_limit
        )
        sequence.insert(best_pos, j)

    return sequence

# ─────────────────────────────────────────────
# MULTI-START CON RUIDO
# ─────────────────────────────────────────────
def solve(jobs, m, n_iter=N_ITER, r=0, time_limit=0.01):
    """
    Ejecuta n_iter construcciones NEH con ruido y retorna la mejor solución.
    La iteración 0 usa r=0 (NEH puro) como ancla de reference.
    block_size = max(10, sqrt(n)) — adaptativo al tamaño de la instancia.
    """
    n = len(jobs)
    block_size = max(10, int(math.sqrt(n)))
    best_seq = None
    best_value = float("inf")

    for i in range(n_iter):
        s     = 0.0 if i == 0 else r
        seq   = construct_noisy_solution(jobs, m, s, block_size, time_limit)
        value = evaluate_sequence(seq, jobs, m)
        if value < best_value:
            best_value = value
            best_seq   = seq

    return best_seq, best_value


# ─────────────────────────────────────────────
# EXPORTAR RESULTADOS A EXCEL
# ─────────────────────────────────────────────
def write_results_to_excel(results, output_file):
    """
    Exporta los resultados de todas las instancias a un archivo Excel.
    Cada hoja lleva el nombre de la instancia (sin extensión .txt).
    Fila 1: [Z, tiempo_ms] | Fila 2: [start_job_0, start_job_1, ...]
    Si el archivo ya existe las hojas se reemplazan individualmente.
    """
    os.makedirs(
        os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
        exist_ok=True
    )

    writer_kwargs = (
        dict(engine="openpyxl", mode="a", if_sheet_exists="replace")
        if os.path.exists(output_file)
        else dict(engine="openpyxl", mode="w")
    )

    with pd.ExcelWriter(output_file, **writer_kwargs) as writer:
        for sheet_name, (total_flow, compute_time_ms, job_start_times) in results.items():
            df = pd.DataFrame([
                [total_flow, compute_time_ms],
                job_start_times
            ])
            df.to_excel(writer, sheet_name=sheet_name, header=False, index=False)

    print(f"\nResultados guardados en: {output_file}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    random.seed(SEED)
    results = {}

    for inst in INSTANCES:
        filepath = os.path.join(INSTANCES_DIR, inst)

        if not os.path.exists(filepath):
            print(f"[SKIP] {inst} — archivo no encontrado")
            continue

        jobs, m = read_instance(filepath)
        n = len(jobs)

        t0 = time.time()
        sequence, _ = solve(jobs, m, n_iter=N_ITER, sigma=R, time_limit=TIME_LIMIT_PER_BLOCK)
        total_flow, schedule = evaluate_sequence(sequence, jobs, m, save_schedule=True)
        compute_time_ms = round((time.time() - t0) * 1000)

        job_start_times = [None] * n
        for op in schedule:
            if op["operation"] == 0:
                job_start_times[op["job"]] = op["start"]

        sheet_name = inst.replace(".txt", "")
        results[sheet_name] = (total_flow, compute_time_ms, job_start_times)
        print(
            f"[OK] {inst:<30} Z={total_flow:>10}  "
            f"iters={N_ITER}  R={R}  tiempo={compute_time_ms:>6} ms"
        )

    if results:
        write_results_to_excel(results, OUTPUT_FILE)
    else:
        print("No se procesó ninguna instancia.")


if __name__ == "__main__":
    main()