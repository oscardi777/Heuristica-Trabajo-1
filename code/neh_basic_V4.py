import os
import bisect
import math
import time
import pandas as pd


# ─────────────────────────────────────────────
# PARÁMETROS
# ─────────────────────────────────────────────
INSTANCES_DIR = "NWJSSP Instances"
OUTPUT_FILE   = "resultados/NWJSSP_OADG_NEH(Constructivo).xlsx"

TIME_LIMIT_PER_BLOCK = 0.01  # Tiempo máximo en segundos por bloque de posiciones

INSTANCES = [
    "ft06.txt",             "ft06r.txt",
    "ft10.txt",             "ft10r.txt",
    "ft20.txt",             "ft20r.txt",
    "tai_j10_m10_1.txt",    "tai_j10_m10_1r.txt",
    "tai_j100_m10_1.txt",   "tai_j100_m10_1r.txt",
    "tai_j100_m100_1.txt",  "tai_j100_m100_1r.txt",
    "tai_j1000_m10_1.txt",  "tai_j1000_m10_1r.txt",
    "tai_j1000_m100_1.txt", "tai_j1000_m100_1r.txt",
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


class Machine:
    """
    Registra los intervalos ocupados de una máquina como listas ordenadas
    por tiempo de inicio, junto con un arreglo de prefijo de máximos de fin
    (max_end_prefix) que permite consultar en O(log k) el máximo tiempo de
    fin entre todos los intervalos que empiezan antes de un umbral dado.

    Esto habilita find_start en O(while_rondas × m × log k) en lugar de
    O(while_rondas × m × k) como en V2/V3, donde k es el número de
    intervalos acumulados en la máquina. La ganancia es especialmente
    significativa en instancias grandes (n ≥ 100).
    """

    __slots__ = ("id", "begins", "ends", "max_end_prefix")

    def __init__(self, id: int):
        self.id = id
        self.begins: list[int] = []          # b_k, siempre ordenado
        self.ends: list[int] = []            # e_k correspondiente a cada b_k
        self.max_end_prefix: list[int] = []  # max_end_prefix[i] = max(ends[0..i])

    def add(self, b: int, e: int) -> None:
        """
        Inserta el intervalo [b, e] manteniendo begins ordenado.
        En NEH los trabajos se programan de forma casi monótona (start crece),
        por lo que la inserción al final es el caso común → O(1) amortizado.
        La reconstrucción parcial del sufijo de max_end_prefix solo ocurre
        cuando hay inserción en medio (instancias con r_j > 0).
        """
        idx = bisect.bisect_right(self.begins, b)
        self.begins.insert(idx, b)
        self.ends.insert(idx, e)

        if idx == len(self.begins) - 1:
            # Inserción al final: solo añadir un elemento al prefijo.
            prev = self.max_end_prefix[idx - 1] if idx > 0 else 0
            self.max_end_prefix.append(max(prev, e))
        else:
            # Inserción en medio: reconstruir desde cero (raro con r_j ≥ 0).
            self.max_end_prefix = []
            running = 0
            for ek in self.ends:
                running = max(running, ek)
                self.max_end_prefix.append(running)

    def max_end_before(self, threshold: int) -> int:
        """
        Retorna el máximo tiempo de fin entre todos los intervalos cuyo
        inicio es estrictamente menor que threshold. Búsqueda en O(log k).
        Devuelve 0 si no existe ninguno.
        """
        idx = bisect.bisect_left(self.begins, threshold)
        return self.max_end_prefix[idx - 1] if idx > 0 else 0


# ─────────────────────────────────────────────
# LECTURA DE INSTANCIAS
# ─────────────────────────────────────────────
def read_instance(file: str) -> tuple[list[Job], int]:
    """
    Lee una instancia NWJSSP desde archivo de texto.
    Asume el formato del Anexo 2: primera línea n m, luego n líneas
    con (machine, proc_time) × m seguidas del release time r_j.
    """
    with open(file) as f:
        n, m = map(int, f.readline().split())
        jobs = []
        for _ in range(n):
            data = list(map(int, f.readline().split()))
            operations = [Operation(data[2 * i], data[2 * i + 1]) for i in range(m)]
            jobs.append(Job(operations, release=data[-1]))

    return jobs, m


# ─────────────────────────────────────────────
# OFFSETS NO-WAIT
# ─────────────────────────────────────────────
def compute_offsets(job: Job) -> list[int]:
    """
    Calcula el desplazamiento acumulado de cada operación respecto al inicio
    del trabajo: offset[u] = sum(p_0, ..., p_{u-1}).

    Bajo la restricción No-Wait, si el trabajo inicia en `start`, la
    operación u ocupa exactamente [start + offset[u], start + offset[u] + p_u].
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
def find_start(job: Job, machines: list[Machine], offsets: list[int]) -> int:
    """
    Encuentra el menor start ≥ r_j tal que ninguna de las operaciones del
    trabajo solape con los intervalos ya registrados en las máquinas.

    En cada iteración del while se toma el máximo de todos los candidatos
    e_k − offset_u de todos los solapamientos detectados (estrategia
    "max-step"), lo que garantiza el menor número posible de rondas.

    La consulta max_end_before(e_op) en cada máquina corre en O(log k),
    siendo k el número de intervalos acumulados en esa máquina.
    """
    start = job.release
    while True:
        max_candidate = start
        feasible = True

        for u, op in enumerate(job.operations):
            b_op = start + offsets[u]
            e_op = b_op + op.p

            # El mayor e_k con b_k < e_op: si supera b_op, hay solapamiento.
            max_ek = machines[op.machine].max_end_before(e_op)
            if max_ek > b_op:
                feasible = False
                candidate = max_ek - offsets[u]
                if candidate > max_candidate:
                    max_candidate = candidate

        if feasible:
            return start
        start = max_candidate


# ─────────────────────────────────────────────
# PROGRAMAR UN TRABAJO
# ─────────────────────────────────────────────
def schedule_job(
    job: Job,
    machines: list[Machine],
    job_id: int,
    schedule: list | None,
) -> int:
    """
    Inserta job_id en la programación actual a su tiempo de inicio mínimo
    factible, actualiza los intervalos de cada máquina y, si schedule no es
    None, registra cada operación. Retorna el tiempo de finalización C_j.
    """
    offsets = compute_offsets(job)
    start = find_start(job, machines, offsets)
    completion = 0

    for u, op in enumerate(job.operations):
        begin  = start + offsets[u]
        finish = begin + op.p
        machines[op.machine].add(begin, finish)
        if schedule is not None:
            schedule.append({
                "job": job_id, "machine": machines[op.machine].id,
                "operation": u, "start": begin, "finish": finish,
            })
        completion = finish

    return completion


# ─────────────────────────────────────────────
# EVALUAR SECUENCIA COMPLETA
# ─────────────────────────────────────────────
def evaluate_sequence(
    sequence: list[int],
    jobs: list[Job],
    m: int,
    save_schedule: bool = False,
) -> int | tuple[int, list]:
    """
    Evalúa una secuencia completa y calcula el Total Flow Time (∑ C_j).
    Si save_schedule=True, devuelve también el detalle de operaciones.
    """
    machines = [Machine(i) for i in range(m)]
    total_flow = 0
    schedule = [] if save_schedule else None

    for j in sequence:
        total_flow += schedule_job(jobs[j], machines, j, schedule)

    return (total_flow, schedule) if save_schedule else total_flow


# ─────────────────────────────────────────────
# EVALUAR INSERCIÓN
# ─────────────────────────────────────────────
def evaluate_insertion(
    sequence: list[int],
    j: int,
    pos: int,
    jobs: list[Job],
    m: int,
) -> int:
    """
    Evalúa el Total Flow Time de insertar j en la posición pos de sequence
    recorriendo los tres segmentos en orden, sin construir lista temporal:
        sequence[0..pos-1]  →  j  →  sequence[pos..end]
    """
    machines = [Machine(i) for i in range(m)]
    total_flow = 0

    for idx in range(pos):
        total_flow += schedule_job(jobs[sequence[idx]], machines, sequence[idx], None)

    total_flow += schedule_job(jobs[j], machines, j, None)

    for idx in range(pos, len(sequence)):
        total_flow += schedule_job(jobs[sequence[idx]], machines, sequence[idx], None)

    return total_flow


# ─────────────────────────────────────────────
# MEJOR POSICIÓN CON BÚSQUEDA POR BLOQUES TEMPORIZADOS
# ─────────────────────────────────────────────
def find_best_insertion(
    sequence: list[int],
    j: int,
    jobs: list[Job],
    m: int,
    block_size: int,
    time_limit: float,
) -> tuple[int, int]:
    """
    Recorre las n+1 posiciones de inserción divididas en bloques consecutivos
    de tamaño block_size. Cada bloque dispone de time_limit segundos; si se
    agota, se avanza directamente al siguiente bloque sin evaluar las posiciones
    restantes de ese bloque.

    Este mecanismo acota el tiempo de inserción en instancias grandes donde
    una sola evaluate_insertion puede superar time_limit, sacrificando
    cobertura de posiciones de manera uniforme en toda la secuencia.
    """
    n_pos      = len(sequence) + 1
    best_pos   = 0
    best_value = float("inf")

    pos = 0
    while pos < n_pos:
        end_block = min(pos + block_size, n_pos)
        t_bloque  = time.time()

        for p in range(pos, end_block):
            if time.time() - t_bloque > time_limit:
                break
            value = evaluate_insertion(sequence, j, p, jobs, m)
            if value < best_value:
                best_value = value
                best_pos   = p

        pos = end_block

    return best_pos, best_value


# ─────────────────────────────────────────────
# HEURÍSTICA CONSTRUCTIVA NEH
# ─────────────────────────────────────────────
def construct_solution(jobs: list[Job], m: int) -> list[int]:
    """
    Heurística NEH adaptada al NWJSSP:
    1. Ordena los trabajos de mayor a menor por r_j + ∑ p_{ju} (regla NEH).
    2. Inserta cada trabajo en la mejor posición encontrada mediante
       búsqueda por bloques con límite de tiempo por bloque.
       block_size = max(10, ⌊√n⌋) — adaptativo al tamaño de la instancia.
    """
    n = len(jobs)
    block_size = max(10, int(math.sqrt(n)))

    order = sorted(
        range(n),
        key=lambda j: jobs[j].release + sum(op.p for op in jobs[j].operations),
        reverse=True,
    )

    sequence = []
    for j in order:
        best_pos, _ = find_best_insertion(
            sequence, j, jobs, m, block_size, TIME_LIMIT_PER_BLOCK
        )
        sequence.insert(best_pos, j)

    return sequence


# ─────────────────────────────────────────────
# EXPORTAR RESULTADOS A EXCEL
# ─────────────────────────────────────────────
def write_results_to_excel(results: dict, output_file: str) -> None:
    """
    Exporta los resultados de todas las instancias a un archivo Excel
    siguiendo el formato del Anexo 3.
    """
    os.makedirs(
        os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
        exist_ok=True,
    )
    writer_kwargs = (
        dict(engine="openpyxl", mode="a", if_sheet_exists="replace")
        if os.path.exists(output_file)
        else dict(engine="openpyxl", mode="w")
    )
    with pd.ExcelWriter(output_file, **writer_kwargs) as writer:
        for sheet_name, (total_flow, compute_time_ms, job_start_times) in results.items():
            df = pd.DataFrame([[total_flow, compute_time_ms], job_start_times])
            df.to_excel(writer, sheet_name=sheet_name, header=False, index=False)

    print(f"\nResultados guardados en: {output_file}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main() -> None:
    results = {}

    for inst in INSTANCES:
        filepath = os.path.join(INSTANCES_DIR, inst)

        if not os.path.exists(filepath):
            print(f"[SKIP] {inst} — archivo no encontrado")
            continue

        jobs, m = read_instance(filepath)
        n = len(jobs)

        t0 = time.time()
        sequence = construct_solution(jobs, m)
        total_flow, schedule = evaluate_sequence(sequence, jobs, m, save_schedule=True)
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
    else:
        print("No se procesó ninguna instancia.")


if __name__ == "__main__":
    main()
