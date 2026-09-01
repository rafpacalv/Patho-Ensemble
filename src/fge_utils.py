"""
Helper compartido para Fast Geometric Ensembles (FGE, Garipov et al. ICLR 2018).

Usado tanto por src/abmil_fge.py (modelos base ABMIL) como por
FGEMLPMetaClassifier en src/meta_models.py (meta-learner), para no duplicar
la fórmula de la tasa de aprendizaje cíclica ni acoplar ambos módulos entre sí.

A diferencia de Snapshot Ensemble (Huang et al. 2017, cosine annealing con
reinicios, ciclos largos que exploran nuevos mínimos desde cero), FGE parte
de un modelo YA CONVERGIDO y usa ciclos cortos con una LR piecewise-linear
(Fig. 2 del paper) para recorrer una región plana ("curva") cercana al mínimo
y muestrear puntos diversos con poco coste adicional de entrenamiento.

Ambos schedules conviven aquí (`fge_cycle_lr` y `se_cycle_lr`, con el despacho
`cycle_lr`) porque la comparación entre ellos es justamente el experimento: se
diferencian sólo en la forma de la LR dentro del ciclo, así que compartir el
resto del bucle de entrenamiento es lo que hace la comparación limpia.
"""
import math


def fge_cycle_lr(epoch_in_cycle, cycle_length, lr_1, lr_2):
    """
    LR piecewise-linear de un ciclo FGE (Garipov et al., Eq. 2 / Fig. 2).

    Dentro de cada ciclo, la LR sube linealmente de lr_2 a lr_1 en la primera
    mitad y baja linealmente de lr_1 a lr_2 en la segunda mitad. lr_1 es el
    techo del ciclo (LR más alta, donde se explora), lr_2 es el piso (LR más
    baja, donde se toma el snapshot al final del ciclo).

    El denominador usa (cycle_length - 1), no cycle_length: con pasos
    discretos por época, dividir entre cycle_length hace que la última época
    del ciclo (donde se toma el snapshot) nunca llegue exactamente a t=1
    (p.ej. con cycle_length=3, la última época quedaría en t=0.667, a un
    33% del pico en vez de en el piso) — el snapshot quedaría tomado con LR
    todavía alta, sin consolidar un punto realmente distinto. Con
    (cycle_length - 1), la última época cae exactamente en t=1 -> lr_2.
    Nota: cycle_length < 3 es degenerado (no hay margen para alcanzar el
    techo lr_1 entre dos pisos consecutivos).

    Args:
        epoch_in_cycle (int): índice de época global (0-based); se reduce
            internamente módulo cycle_length.
        cycle_length (int): número de épocas por ciclo (típicamente 3-4 para
            FGE, mucho más corto que los ciclos de Snapshot Ensemble).
        lr_1 (float): LR techo del ciclo.
        lr_2 (float): LR piso del ciclo (valor al inicio y al final).

    Returns:
        float: LR a usar en esa época.
    """
    denom = max(cycle_length - 1, 1)
    t = (epoch_in_cycle % cycle_length) / denom
    if t <= 0.5:
        return lr_2 + (lr_1 - lr_2) * (t / 0.5)
    return lr_1 - (lr_1 - lr_2) * ((t - 0.5) / 0.5)


def se_cycle_lr(epoch_in_cycle, cycle_length, restart_lr, eta_min=None):
    """
    LR de un ciclo de Snapshot Ensembles (Huang et al., ICLR 2017, Eq. 2).

    A diferencia de `fge_cycle_lr` —que sube y baja linealmente en torno a un
    mínimo ya encontrado—, aquí cada ciclo REINICIA en `restart_lr` y desciende
    con un coseno hasta casi cero:

        α(t) = eta_min + (restart_lr - eta_min) * (1 + cos(pi * t)) / 2

    con t = (época dentro del ciclo) / cycle_length. El snapshot se toma al
    final del ciclo, donde la LR es mínima.

    La diferencia de fondo entre ambos schedules es cuánta diversidad generan:
    el reinicio a LR alta busca sacar al modelo de su mínimo hacia otro
    distinto, a costa de que cada snapshot sea individualmente peor. FGE hace
    lo contrario. Ese es exactamente el eje que se quiere medir.

    Nota sobre el denominador: aquí se divide entre `cycle_length` y no entre
    (cycle_length - 1) como en fge_cycle_lr. Es deliberado: en el coseno la
    última época del ciclo debe quedar cerca del mínimo pero SIN completar el
    medio periodo, de modo que la siguiente época reinicie en el techo. Usar
    (cycle_length - 1) haría que la última época valiese exactamente eta_min y
    el reinicio duplicaría ese valor.

    Args:
        epoch_in_cycle (int): índice de época global (0-based); se reduce
            internamente módulo cycle_length.
        cycle_length (int): épocas por ciclo. Típicamente largo (15-50) en SE,
            frente a las 3-4 de FGE.
        restart_lr (float): LR al inicio de cada ciclo (α0 en el paper).
        eta_min (float): LR mínima al final del ciclo. Si es None, usa
            restart_lr * 1e-3 (mismo criterio que SnapshotMLPMetaClassifier).

    Returns:
        float: LR a usar en esa época.
    """
    if eta_min is None:
        eta_min = restart_lr * 1e-3
    t = (epoch_in_cycle % cycle_length) / max(cycle_length, 1)
    return eta_min + (restart_lr - eta_min) * (1 + math.cos(math.pi * t)) / 2


def cycle_lr(schedule, epoch_in_cycle, cycle_length, lr_1, lr_2):
    """
    Despacha al schedule pedido, para que quien entrena no dependa de cuál sea.

    Args:
        schedule (str): "fge" (piecewise-linear, Garipov et al.) o "se"
            (coseno con reinicios, Huang et al.).
        epoch_in_cycle, cycle_length: igual que en las funciones concretas.
        lr_1 (float): LR techo del ciclo. En "se" es el valor del reinicio (α0).
        lr_2 (float): LR piso del ciclo.

    Returns:
        float: LR a usar en esa época.
    """
    if schedule == "fge":
        return fge_cycle_lr(epoch_in_cycle, cycle_length, lr_1, lr_2)
    if schedule == "se":
        return se_cycle_lr(epoch_in_cycle, cycle_length, lr_1, eta_min=lr_2)
    raise ValueError(f"schedule desconocido: {schedule!r} (esperado 'fge' o 'se')")


def set_lr(optimizer, lr):
    """Setea la LR de todos los param_groups de un optimizador (in-place)."""
    for g in optimizer.param_groups:
        g["lr"] = lr
