#!/usr/bin/env python3
"""
Puerta de la Etapa 1 del brazo espacial. No toca el pipeline ni entrena nada:
carga .h5 reales, construye los grafos y comprueba las propiedades de las que
depende todo lo demás.

Se ejecuta sobre las slides mínima / mediana / máxima de las 112 de
cptac_brca/TP53_mutation, no sobre slides sintéticas: las cifras esperadas
(grado medio 7.2-7.9, cero aislados) se midieron sobre esos ficheros y un
sintético no las verificaría.

    python test_spatial_gnn.py [--work_dir ...] [--model uni_v2]
"""
import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

import graph_utils as G
from abmil_engine import ABMIL
from spatial_abmil import ARCHS, SpatialABMIL, attention_stats, make_factory
from utils import get_features_dir

DEFAULT_WORK_DIR = "/shared/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
STRIDE = 448

_fails = []


def check(cond, msg):
    print(f"  {'OK  ' if cond else 'FALLO'}  {msg}")
    if not cond:
        _fails.append(msg)
    return cond


def pick_slides(work_dir, train_source, task_name, model):
    """Las slides mínima, mediana y máxima por número de tiles."""
    df = pd.read_csv(f"{work_dir}/{train_source}/{task_name}/k=all.tsv", sep="\t")
    fdir = get_features_dir(work_dir, train_source, model)
    sizes = []
    for s in df["slide_id"]:
        p = f"{fdir}/{s}.h5"
        if Path(p).exists():
            with h5py.File(p, "r") as f:
                sizes.append((f["coords"].shape[0], s))
    sizes.sort()
    picks = [sizes[0], sizes[len(sizes) // 2], sizes[-1]]
    print(f"  {len(sizes)} slides en la tarea; elegidas min/mediana/max: "
          f"N = {', '.join(str(n) for n, _ in picks)}")
    return fdir, picks


def load(fdir, slide):
    with h5py.File(f"{fdir}/{slide}.h5", "r") as f:
        return f["features"][:], f["coords"][:], dict(f["coords"].attrs)


# ─────────────────────────────── 1. grafos ───────────────────────────────
def test_graphs(fdir, picks):
    print("\n[1] Construcción del grafo sobre slides reales")
    for n_exp, slide in picks:
        feats, coords, attrs = load(fdir, slide)
        # ctranspath_monai no trae attrs en coords: se cae al default de 448,
        # que es lo que hace load_graphs_cached. La comprobación real de que el
        # stride es el correcto es que el grafo salga por retícula, abajo.
        stride = int(attrs.get("patch_size_level0", STRIDE))
        check(stride == STRIDE,
              f"{slide[:14]}: stride = {stride}"
              f"{'' if 'patch_size_level0' in attrs else ' (por defecto; sin attrs)'}")
        check(len(coords) == len(feats),
              f"{slide[:14]}: coords y features alineados ({len(coords)} filas)")

        t0 = time.perf_counter()
        ei, deg, info = G.build_graph(coords, stride)
        dt = time.perf_counter() - t0

        # El grado legítimo depende de lo agresiva que fuese la extracción al
        # filtrar parches, no del código: sobre la MISMA retícula de 448,
        # uni_v2 da 7.75 y ctranspath_monai 5.34, porque conserva 1.7x menos
        # parches. Así que aquí se comprueba que el grafo no está ROTO
        # (construido por retícula, con vecindario real), no una banda fija.
        check(info["method"] == "lattice",
              f"{slide[:14]}: construido por retícula (no fallback k-NN)")
        check(1.0 <= info["mean_deg"] <= 8.0,
              f"{slide[:14]}: N={n_exp:6d} E={ei.shape[1]:7d} "
              f"grado medio={info['mean_deg']:.2f} "
              f"(aislados {info['n_isolated']})")
        check(dt < 5.0, f"{slide[:14]}: grafo construido en {dt * 1000:.0f} ms")


def test_offlattice_raises(fdir, picks):
    print("\n[2] Coords fuera de retícula fallan alto (no dan un grafo vacío)")
    _, slide = picks[0]
    _, coords, _ = load(fdir, slide)
    bad = coords.copy()
    bad[0, 0] += 1                      # un solo tile desalineado
    try:
        G.lattice_edges(bad, STRIDE)
        check(False, "lattice_edges NO lanzó con coords desalineados")
    except ValueError as e:
        check(True, f"lattice_edges lanza ValueError: {str(e)[:60]}")

    dup = np.vstack([coords, coords[:1]])
    try:
        G.lattice_edges(dup, STRIDE)
        check(False, "lattice_edges NO lanzó con coords duplicados")
    except ValueError as e:
        check(True, f"duplicados detectados: {str(e)[:60]}")


# ───────────────────────── 3. equivalencia del kernel ─────────────────────────
def test_kernel_equivalence(fdir, picks):
    print("\n[3] SpMM equivale a index_select + index_add_, y propaga gradiente")
    for n_exp, slide in picks:
        _, coords, _ = load(fdir, slide)
        ei, deg, _ = G.build_graph(coords, STRIDE)
        n = len(coords)
        a_norm = G.make_operator(ei, deg, n)

        h = torch.randn(n, 64, dtype=torch.float32)
        ref = G._aggregate_mean(h, ei, deg)
        got = torch.sparse.mm(a_norm, h)
        check(torch.allclose(ref, got, atol=1e-5),
              f"{slide[:14]}: allclose(sparse_mm, index_add_) "
              f"maxdiff={float((ref - got).abs().max()):.2e}")

    h = torch.randn(n, 64, requires_grad=True)
    torch.sparse.mm(a_norm, h).sum().backward()
    check(h.grad is not None and torch.isfinite(h.grad).all(),
          "el gradiente fluye a través de torch.sparse.mm (CSR)")


# ───────────────────────── 4. control negativo ─────────────────────────
def test_shuffle_control(fdir, picks):
    print("\n[4] El control shuffle es isomorfo (relabelado, no reconstrucción)")
    _, slide = picks[1]
    _, coords, _ = load(fdir, slide)
    n = len(coords)

    ei, deg, _ = G.build_graph(coords, STRIDE)
    seed = G.seed_from_slide_id(slide)
    ei_s, deg_s, _ = G.build_graph(coords, STRIDE, shuffle_seed=seed)

    check(ei.shape == ei_s.shape, f"mismo E: {ei.shape[1]} vs {ei_s.shape[1]}")
    check(torch.equal(deg.sort().values, deg_s.sort().values),
          "misma secuencia de grados ordenada -> grafos isomorfos")
    check(not torch.equal(ei, ei_s), "pero el emparejamiento feature<->posición cambia")

    again = G.seed_from_slide_id(slide)
    check(seed == again, f"seed_from_slide_id estable dentro del proceso ({seed})")
    ei_s2, _, _ = G.build_graph(coords, STRIDE, shuffle_seed=again)
    check(torch.equal(ei_s, ei_s2), "misma semilla -> mismo relabelado")

    # self-loops: media de vecinos = h, así que 'smooth' colapsa a la identidad
    ei_self, deg_self, _ = G.build_graph(coords, STRIDE, mode="self")
    a_self = G.make_operator(ei_self, deg_self, n)
    h = torch.randn(n, 32)
    check(torch.allclose(torch.sparse.mm(a_self, h), h, atol=1e-6),
          "mode='self': la agregación es la identidad (control spatial_self)")

    # Nodos sin vecinos. uni_v2 no tiene ninguno, así que esto se construye a
    # propósito: un tile aislado a 100 celdas del resto. Antes del arreglo,
    # deg se clampaba a 1 y la agregación daba 0, con lo que 'smooth' encogía
    # el nodo — (1-alpha)*h — en vez de dejarlo quieto.
    print("\n[4b] Un nodo sin vecinos conserva su valor (no se encoge a cero)")
    _, _, info_0 = G.build_graph(coords, STRIDE)
    lonely = np.vstack([coords, coords.max(0) + 100 * STRIDE])
    ei_l, deg_l, info_l = G.build_graph(lonely, STRIDE)
    check(info_l["n_isolated"] == info_0["n_isolated"] + 1,
          f"aislados {info_0['n_isolated']} -> {info_l['n_isolated']} "
          "al añadir un tile suelto")
    a_l = G.make_operator(ei_l, deg_l, len(lonely))
    hl = torch.randn(len(lonely), 16)
    agg = torch.sparse.mm(a_l, hl)
    check(torch.allclose(agg[-1], hl[-1], atol=1e-6),
          "la agregación sobre el nodo aislado devuelve su propio valor")


# ───────────────────────── 5. modelo ─────────────────────────
def test_model(fdir, picks, device):
    print(f"\n[5] SpatialABMIL en {device}")
    n_exp, slide = picks[1]
    feats, coords, _ = load(fdir, slide)
    n, d = feats.shape
    ei, deg, _ = G.build_graph(coords, STRIDE)
    a_norm = G.SlideGraph(ei, deg, n).to(device)
    x = torch.tensor(feats, dtype=torch.float32, device=device)

    torch.manual_seed(0)
    base = ABMIL(d, 2).to(device).eval()
    with torch.no_grad():
        lo_b, at_b = base(x)

    for arch in ARCHS:
        torch.manual_seed(0)
        m = SpatialABMIL(d, 2, mode=arch, n_layers=2).to(device).eval()
        with torch.no_grad():
            lo, at = m(x, a_norm)
        p = sum(q.numel() for q in m.parameters())
        pb = sum(q.numel() for q in base.parameters())
        check(lo.shape == lo_b.shape and at.shape == at_b.shape,
              f"{arch:8s}: formas de salida iguales a ABMIL "
              f"{tuple(lo.shape)}, {tuple(at.shape)}")
        check(torch.isfinite(lo).all() and torch.isfinite(at).all(),
              f"{arch:8s}: salidas finitas")
        print(f"          parámetros: {p:,} vs ABMIL {pb:,}  (+{100 * (p - pb) / pb:.1f} %)")

    # smooth arranca en alpha=0 -> el grafo no debe poder cambiar la salida
    print("\n[6] 'smooth' con alpha=0 es exactamente ABMIL en la inicialización")
    torch.manual_seed(0)
    m = SpatialABMIL(d, 2, mode="smooth", n_layers=2).to(device).eval()
    check(m.alphas() == [0.0, 0.0], f"alphas iniciales = {m.alphas()}")
    ei_s, deg_s, _ = G.build_graph(coords, STRIDE, shuffle_seed=123)
    a_shuf = G.SlideGraph(ei_s, deg_s, n).to(device)
    with torch.no_grad():
        lo1, _ = m(x, a_norm)
        lo2, _ = m(x, a_shuf)
    check(torch.equal(lo1, lo2), "la salida es invariante al grafo con alpha=0")

    with torch.no_grad():
        m.layers[0].alpha.fill_(0.7)
        m.layers[1].alpha.fill_(0.3)
        lo3, _ = m(x, a_norm)
        ei_self, deg_self, _ = G.build_graph(coords, STRIDE, mode="self")
        lo4, _ = m(x, G.SlideGraph(ei_self, deg_self, n).to(device))
    check(not torch.allclose(lo1, lo3, atol=1e-6),
          "con alpha>0 el grafo SÍ cambia la salida")
    check(torch.allclose(lo1, lo4, atol=1e-5),
          "con self-loops vuelve a ABMIL sea cual sea alpha")

    ess_b, top_b = attention_stats(at_b)
    print(f"\n      atención ABMIL: ESS={ess_b:.1f} de N={n}, masa top-1 %={top_b:.3f}")


# ───────────────────────── 7. coste ─────────────────────────
def test_cost(fdir, picks, device):
    print(f"\n[7] Coste del forward en la slide mayor ({device})")
    n_exp, slide = picks[-1]
    feats, coords, _ = load(fdir, slide)
    n, d = feats.shape
    ei, deg, _ = G.build_graph(coords, STRIDE)
    a_norm = G.SlideGraph(ei, deg, n).to(device)
    x = torch.tensor(feats, dtype=torch.float32, device=device)

    def timeit(fn, reps=3):
        with torch.no_grad():
            fn()
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            if device == "cuda":
                torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps

    torch.manual_seed(0)
    base = ABMIL(d, 2).to(device).eval()
    t_base = timeit(lambda: base(x))
    print(f"      N={n}, E={ei.shape[1]}, D={d}")
    print(f"      ABMIL           {t_base * 1000:8.1f} ms   (referencia)")

    ratios = {}
    for arch in ARCHS:
        torch.manual_seed(0)
        m = SpatialABMIL(d, 2, mode=arch, n_layers=2).to(device).eval()
        t = timeit(lambda: m(x, a_norm))
        ratios[arch] = t / t_base
        print(f"      {arch:14s}  {t * 1000:8.1f} ms   x{t / t_base:.2f}")

    # El modelo de FLOPs predecía x1.004 para 'smooth' y falla: la agregación
    # va limitada por ancho de banda, no por FLOPs (ver el docstring de
    # graph_utils). La puerta se pone sobre lo que decide de verdad si el
    # experimento es asequible — el coste relativo — y no sobre la predicción.
    # El ratio depende de D y NO es una constante del método: la agregación
    # cuesta lo mismo (opera en proj_dim=512) mientras que la proyección de
    # entrada escala con D. Con ctranspath (768D) la proyección es 2x más
    # barata que con uni_v2 (1536D), así que el mismo grafo pesa el doble en
    # términos relativos. Poner una puerta fija sobre el ratio fue el mismo
    # error de calibración que la banda de grado; se informa y se acota flojo.
    check(ratios["smooth"] < 5.0,
          f"'smooth' cuesta x{ratios['smooth']:.2f} (el ratio depende de D; "
          f"la cifra que decide es el tiempo real de las 50 folds)")
    # NO se comprueba monotonía coste-capacidad: es falsa, y medirla lo demostró.
    # En CPU sage_bn (x1.62) sale MÁS BARATO que smooth (x1.70) pese a tener
    # 132 k parámetros más por capa, porque agrega en la dimensión del cuello de
    # botella (128) en vez de en 512. Con la agregación limitada por ancho de
    # banda, eso es 4x menos tráfico, y compensa de sobra los GEMM extra. En GPU
    # sí ordena (x1.22 < 1.40 < 1.50) porque ahí dominan los densos.
    # Capacidad y coste son ejes distintos; sólo coinciden si el ancho de banda
    # no es el cuello de botella.
    check(ratios["sage"] > ratios["smooth"],
          f"'sage' (x{ratios['sage']:.2f}) cuesta más que 'smooth' "
          f"(x{ratios['smooth']:.2f}) — el extremo caro de la escalera lo es")
    check(max(ratios.values()) < 8.0,
          f"ningún brazo se dispara (máx x{max(ratios.values()):.2f})")

    backend = "SpMM (CSR)" if a_norm._use_spmm() else "index_add_"
    ei_mb = ei.numel() * 8 / 1e6
    print(f"      backend elegido en {device}: {backend}")
    print(f"      edge_index: {ei_mb:.1f} MB frente a {n * d * 4 / 1e6:.1f} MB de features")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default=DEFAULT_WORK_DIR)
    ap.add_argument("--train_source", default="cptac_brca")
    ap.add_argument("--task_name", default="TP53_mutation")
    ap.add_argument("--model", default="uni_v2")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("Etapa 1 — puerta del brazo espacial (grafo + SpatialABMIL)")
    print("=" * 78)
    print(f"  modelo={a.model}  device={device}  torch={torch.__version__}")

    fdir, picks = pick_slides(a.work_dir, a.train_source, a.task_name, a.model)

    test_graphs(fdir, picks)
    test_offlattice_raises(fdir, picks)
    test_kernel_equivalence(fdir, picks)
    test_shuffle_control(fdir, picks)
    test_model(fdir, picks, device)
    test_cost(fdir, picks, device)

    print("\n" + "=" * 78)
    if _fails:
        print(f"PUERTA NO SUPERADA — {len(_fails)} comprobación(es) fallidas:")
        for f in _fails:
            print(f"  - {f}")
        print("=" * 78)
        sys.exit(1)
    print("PUERTA SUPERADA — Etapa 1 lista para el cableado del motor")
    print("=" * 78)


if __name__ == "__main__":
    main()
