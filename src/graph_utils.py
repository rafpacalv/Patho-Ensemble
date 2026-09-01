"""
Grafos de vecindad espacial a partir de los `coords` de los .h5 de PARADIS.

Los .h5 de features traen `coords (N,2) int64` alineado fila a fila con
`features (N,D)`, en píxeles de nivel 0, y `coords.attrs['patch_size_level0']`
da el paso (448 en 20x_224px_0px_overlap). Medido sobre las 112 slides de
cptac_brca/TP53_mutation: los coords caen en una retícula EXACTA y sin
duplicados, así que la vecindad de Moore se resuelve con aritmética entera y
una búsqueda binaria — no hace falta un k-NN métrico ni una librería de grafos.

Grado medido en las slides mínima / mediana / máxima (N = 1135 / 7186 / 29622):
E/N = 7.21 / 7.64 / 7.81, mediana de grado 8, CERO nodos aislados. El tejido es
localmente contiguo; la ocupación de 0.198 del bounding box es un artefacto de
la caja, no del vecindario.

La agregación tiene DOS implementaciones y se despacha por dispositivo, porque
medirlas dio lo contrario de lo que predecía el modelo de FLOPs.

  - `index_add_`: materializa un tensor (E, D) — 455 MB en la slide mayor de
    uni_v2 (N=29622, E≈231k, D=512) más el buffer de gradiente, frente a los
    61 MB de la activación que produce. En CUDA usa atomicAdd, cuyo orden de
    suma varía entre ejecuciones.
  - SpMM sobre un operador CSR normalizado: no materializa el intermedio.

El modelo de FLOPs decía que SpMM sería gratis (+0.5 %). Medido en la slide
mayor a D=512, en CPU con 16 hilos: index_add_ 26.4 ms, SpMM COO 43.5 ms, SpMM
CSR 55.6 ms. SpMM es 2.1x MÁS LENTO. La razón es que el modelo de FLOPs es el
modelo equivocado para esta operación: la SpMM tiene una intensidad aritmética
pésima y va limitada por ancho de banda con accesos irregulares, mientras que
el GEMM dominante (la proyección 1536->512) corre cerca del pico. Contar FLOPs
compara dos operaciones que viven en regímenes distintos.

De ahí el despacho por dispositivo:
  - CPU  -> index_add_. Es el más rápido Y es determinista (suma secuencial),
            así que la objeción del atomicAdd no aplica.
  - CUDA -> SpMM. Ahí sí aplica: el protocolo de análisis de este repo cuenta
            EMPATES como estadístico de primera clase (con 22 slides de test
            por fold, en kappa llegaron a 22 de 50), y un argmax que baila en
            la frontera cambia la matriz de confusión. Un recuento de empates
            irreproducible corrompe justo la cifra que el protocolo existe para
            reportar.

El test de Etapa 1 fija `allclose` entre ambas rutas, así que la elección de
backend no puede cambiar el resultado, sólo el tiempo.
"""
import hashlib

import numpy as np
import torch


# Vecindad de Moore (8 vecinos). El nodo NO se incluye: la señal propia la
# conserva el residual de GraphLayer, que es más barato que inflar E un 12.5 %.
_MOORE8 = ((-1, -1), (-1, 0), (-1, 1),
           (0, -1),           (0, 1),
           (1, -1),  (1, 0),  (1, 1))


# ───────────────────────── construcción de aristas ─────────────────────────
def lattice_edges(coords, stride=448, offsets=_MOORE8):
    """Aristas de la retícula exacta. Devuelve edge_index (2,E) int64 [src, dst].

    Falla explícitamente si los coords no caen en la retícula, en vez de
    devolver un grafo vacío. Con un edge_index vacío la agregación da 0 y, por
    el residual, SpatialABMIL queda numéricamente IDÉNTICO a ABMIL: produciría
    un nulo perfectamente limpio e indistinguible de uno real. Es el fallo más
    caro de diagnosticar de todo el diseño, así que se falla alto.

    Args:
        coords: (N,2) array de coordenadas en píxeles de nivel 0.
        stride: paso de la retícula (patch_size_level0 del .h5).
        offsets: desplazamientos de vecindad en celdas. Por defecto Moore-8.

    Returns:
        edge_index: (2,E) int64. edge_index[0]=src, edge_index[1]=dst.

    Raises:
        ValueError: si los coords no están en la retícula o hay duplicados.
    """
    c = np.asarray(coords, dtype=np.int64)
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError(f"coords debe ser (N,2), recibido {c.shape}")
    n = len(c)
    if n == 0:
        raise ValueError("coords vacío")

    rel = c - c.min(0)
    if (rel % stride).any():
        off = int((rel % stride).any(axis=1).sum())
        raise ValueError(
            f"{off}/{n} coords fuera de la retícula de paso {stride}. "
            "Usa knn_edges() para datasets no reticulados."
        )
    g = rel // stride

    # Clave lineal fila-mayor. El +2 deja una columna guardia vacía: sin ella,
    # la sonda dy=+1 en el borde derecho generaría la clave de la fila
    # siguiente y devolvería un vecino falso.
    width = int(g[:, 1].max()) + 2
    key = g[:, 0] * width + g[:, 1]

    order = np.argsort(key, kind="stable")
    ks = key[order]
    if (np.diff(ks) == 0).any():
        raise ValueError("coords duplicados: la búsqueda binaria devolvería sólo uno")

    src, dst = [], []
    for dx, dy in offsets:
        nk = (g[:, 0] + dx) * width + (g[:, 1] + dy)
        pos = np.clip(np.searchsorted(ks, nk), 0, n - 1)
        hit = ks[pos] == nk
        dst.append(np.nonzero(hit)[0])       # el nodo i
        src.append(order[pos[hit]])          # su vecino

    return np.stack([np.concatenate(src), np.concatenate(dst)]).astype(np.int64)


def knn_edges(coords, k=8):
    """Fallback k-NN para datasets que no estén reticulados. Devuelve (2,E) int64.

    Se simetriza: `kneighbors_graph` no lo es, y una arista dirigida rompe la
    simetría de la media de vecinos sin que nada lo señale.
    """
    from sklearn.neighbors import kneighbors_graph

    c = np.asarray(coords, dtype=np.float64)
    a = kneighbors_graph(c, min(k, len(c) - 1), mode="connectivity")
    a = a.maximum(a.T).tocoo()
    return np.stack([a.col, a.row]).astype(np.int64)


def self_loop_edges(n):
    """Sólo self-loops. Es el control negativo `spatial_self`.

    Con la media de vecinos, self-loops dan mean_neigh(h) = h, así que en modo
    `smooth` la capa colapsa a h + alpha*(h - h) = h: SpatialABMIL queda
    EXACTAMENTE en ABMIL. Eso lo convierte además en una tercera prueba de
    inercia del cableado. En los modos sage sí es un control con contenido:
    mantiene los pesos y elimina el vecindario.
    """
    i = np.arange(n, dtype=np.int64)
    return np.stack([i, i])


# ───────────────────────────── control negativo ─────────────────────────────
def seed_from_slide_id(slide_id):
    """Semilla estable derivada del slide_id, vía sha256.

    NO se usa hash() de Python: está aleatorizado por proceso para str, así que
    la permutación cambiaría entre folds y entre train/test dentro del mismo
    experimento. El control tiene que ser fijo por slide e idéntico en todos los
    splits — si se redibuja, deja de ser un control y pasa a ser data
    augmentation fuerte, y habrías cambiado dos cosas a la vez.
    """
    return int.from_bytes(
        hashlib.sha256(str(slide_id).encode()).digest()[:8], "little"
    ) % (2 ** 32)


# ───────────────────────────── API principal ─────────────────────────────
def build_graph(coords, stride=448, k=8, mode="lattice", shuffle_seed=None):
    """Construye el grafo de una slide. Devuelve (edge_index, deg, info).

    `info` registra por qué camino se construyó y cuántos nodos quedaron sin
    vecinos. Esa contabilidad no es decorativa: la extracción `_monai` de
    ctranspath resultó tener un 0.96 % de nodos aislados y algunas slides fuera
    de retícula, y sin `info` el fallback a k-NN pasaba desapercibido.

    Args:
        coords: (N,2) coordenadas de nivel 0.
        stride: paso de la retícula.
        k: vecinos para el fallback k-NN.
        mode: 'lattice' (retícula exacta, con fallback a k-NN), 'knn' o 'self'.
        shuffle_seed: si se indica, activa el CONTROL NEGATIVO por relabelado.

    El control relabela los nodos con una permutación fija (`ei = perm[ei]`).

    El grafo resultante es ISOMORFO al real, así que grado, número de aristas,
    coste de cómputo y número de parámetros son idénticos POR CONSTRUCCIÓN, no
    por suerte. Lo único que se destruye es el emparejamiento
    feature <-> posición, que es exactamente lo que se quiere aislar: separa la
    información ESPACIAL de la capacidad añadida.

    Se relabela en vez de "barajar coords y reconstruir" precisamente por eso:
    reconstruir daría otro grafo, con otro grado y otro E, y el contraste
    mezclaría dos efectos.
    """
    n = len(coords)
    method = mode
    if mode == "self":
        ei = self_loop_edges(n)
    elif mode == "knn":
        ei = knn_edges(coords, k)
    else:
        try:
            ei = lattice_edges(coords, stride)
            method = "lattice"
        except ValueError:
            # El fallback se REGISTRA. Silencioso, convierte "el stride es otro"
            # en un grafo distinto que entrena igual y no avisa.
            ei = knn_edges(coords, k)
            method = "knn_fallback"

    if shuffle_seed is not None:
        perm = np.random.default_rng(shuffle_seed).permutation(n)
        ei = perm[ei]                        # relabelado -> isomorfo

    ei = torch.from_numpy(np.ascontiguousarray(ei))

    # Nodos sin vecinos: se les da un self-loop para que la media de vecinos sea
    # su propio valor y la capa sea la IDENTIDAD sobre ellos. Sin esto, deg se
    # clampaba a 1 y la agregación daba 0, con lo que 'smooth' calculaba
    # h + alpha*(0 - h) = (1-alpha)*h: encoge el nodo hacia cero en vez de
    # dejarlo quieto. uni_v2 no tiene aislados y no lo destapó; ctranspath sí.
    counts = torch.bincount(ei[1], minlength=n)
    isolated = (counts == 0).nonzero().flatten()
    if len(isolated):
        ei = torch.cat([ei, torch.stack([isolated, isolated])], dim=1)

    deg = torch.bincount(ei[1], minlength=n).clamp(min=1).float()
    info = {"method": method, "n_isolated": int(len(isolated)), "n": n,
            "mean_deg": float(ei.shape[1] - len(isolated)) / n}
    return ei, deg, info


def make_operator(edge_index, deg, n):
    """Operador de agregación media, disperso y normalizado por filas.

    A[i,j] = 1/deg[i] si j es vecino de i. Entonces (A @ h)[i] es la media de
    los vecinos de i, en un solo SpMM y sin materializar el intermedio (E,D).

    edge_index viene como [src, dst]; una matriz dispersa quiere [fila, col] =
    [dst, src], de ahí el flip.
    """
    rows, cols = edge_index[1], edge_index[0]
    vals = (1.0 / deg)[rows]
    a = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, (n, n)
    ).coalesce()
    return a.to_sparse_csr()


class SlideGraph:
    """El grafo de una slide, con las dos rutas de agregación y despacho.

    Es lo que se cachea por slide y lo que el motor mueve con `.to(device)`,
    igual que hace hoy con el tensor de features. Encapsular el despacho aquí
    evita que la elección de backend se filtre a SpatialABMIL.

    El operador CSR se construye de forma perezosa: en CPU no se usa nunca, así
    que pagarlo en la carga sería tirar memoria y tiempo por 112 slides.
    """

    __slots__ = ("edge_index", "deg", "n", "backend", "_op")

    def __init__(self, edge_index, deg, n, backend="auto"):
        self.edge_index = edge_index
        self.deg = deg
        self.n = n
        self.backend = backend
        self._op = None

    def to(self, device):
        if self.edge_index.device == torch.device(device):
            return self
        g = SlideGraph(
            self.edge_index.to(device), self.deg.to(device), self.n, self.backend
        )
        return g

    def _use_spmm(self):
        if self.backend == "spmm":
            return True
        if self.backend == "index_add":
            return False
        # auto: en CUDA el atomicAdd de index_add_ no es determinista y el
        # protocolo cuenta empates; en CPU index_add_ es determinista y 2.1x
        # más rápido que la SpMM (medido).
        return self.edge_index.is_cuda

    def aggregate(self, h):
        """Media de los vecinos de cada nodo. (N,D) -> (N,D)."""
        if self._use_spmm():
            if self._op is None:
                self._op = make_operator(self.edge_index, self.deg, self.n)
            return torch.sparse.mm(self._op, h)
        return _aggregate_mean(h, self.edge_index, self.deg)


def slide_graph(coords, stride=448, k=8, mode="lattice", shuffle_seed=None,
                backend="auto"):
    """build_graph envuelto en SlideGraph. Es lo que se cachea por slide.

    Devuelve (SlideGraph, info) — info se agrega en load_graphs_cached para
    poder validar el conjunto, no slide a slide.
    """
    ei, deg, info = build_graph(coords, stride, k, mode, shuffle_seed)
    return SlideGraph(ei, deg, len(coords), backend), info


# ───────────────────────── referencia y validación ─────────────────────────
def _aggregate_mean(h, edge_index, deg):
    """Media de vecinos por index_select + index_add_. Referencia legible.

    NO se usa en el camino de entrenamiento (ver el docstring del módulo:
    tensor (E,D) de 455 MB y atomicAdd no determinista). El test de Etapa 1 fija
    `allclose(sparse_mm, _aggregate_mean)` para que el SpMM siga siendo
    auditable contra algo que se lee de un vistazo.
    """
    src, dst = edge_index[0], edge_index[1]
    agg = torch.zeros_like(h)
    agg.index_add_(0, dst, h.index_select(0, src))
    return agg / deg.unsqueeze(1)


def validate_graph(edge_index, deg, n, lo=6.0, hi=8.0, name=""):
    """Comprueba que el grafo es plausible. Aserción de arranque.

    Existe por el modo de fallo silencioso: un grafo degenerado no lanza nada,
    entrena igual, y devuelve un nulo que parece un resultado.
    """
    mean_deg = float(len(edge_index[1])) / n
    n_isolated = int((torch.bincount(edge_index[1], minlength=n) == 0).sum())
    if not (lo <= mean_deg <= hi):
        raise ValueError(
            f"{name}: grado medio {mean_deg:.2f} fuera de [{lo}, {hi}] — "
            "el grafo no es la vecindad de Moore que se espera"
        )
    return mean_deg, n_isolated
