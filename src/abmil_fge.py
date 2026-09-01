"""
Ensembles por snapshots para ABMIL base: FGE (Garipov et al. ICLR 2018) y
Snapshot Ensembles (Huang et al. ICLR 2017), seleccionables con `schedule`.

Parte de un modelo ABMIL YA CONVERGIDO (el `best_state` que produce
abmil_engine.train_abmil()) y hace fine-tuning con ciclos de LR, tomando un
snapshot al final de cada ciclo (mínimo de LR). No reescribe carga de datos,
splits ni la arquitectura ABMIL — reutiliza abmil_engine.py en su totalidad.

Los dos métodos se diferencian sólo en la forma de la LR dentro del ciclo
(src/fge_utils.py): FGE usa ciclos cortos (2-4 épocas) con una rampa
piecewise-linear de amplitud pequeña alrededor del mínimo ya encontrado;
Snapshot Ensembles usa ciclos largos (15+ épocas) que reinician en una LR alta
y descienden con un coseno, buscando salir hacia otro mínimo. Todo lo demás
—snapshots por ciclo, early stopping inter-ciclo, include_base— es común, que
es lo que permite atribuir cualquier diferencia al schedule y no al montaje.

Medido en cptac_brca (50 folds): con `schedule="fge"` los snapshots del mismo
modelo correlacionan r = 0.93-0.98, frente a r = 0.797 entre modelos
fundacionales distintos. El eje de diversidad relevante para el ensemble ya lo
ocupa el multi-modelo, así que la comparación FGE/SE es sobre el margen que
queda.
"""
import copy
import random

import numpy as np
import torch
import torch.nn as nn

from abmil_engine import ABMIL, _grouped_val_split, _auc, _RECIPES, ABMIL_RECIPE
from fge_utils import cycle_lr, set_lr


def train_abmil_fge(
    feats, stems, y, groups, in_dim, n_classes, device, seed,
    warm_start_state,
    n_cycles=6, cycle_length=4, lr_1=2e-4, lr_2=2e-5,
    proj_dim=512, dropout=0.25, wd=1e-4, val_frac=0.2, cycle_patience=2,
    include_base=False, schedule="fge",
):
    """Fine-tunea un ABMIL convergido con ciclos FGE cortos, tomando snapshots.

    Args:
        feats, stems, y, groups: mismos argumentos que abmil_engine.train_abmil
            (dict de features, IDs de bolsa, labels, IDs de paciente/case_id).
        in_dim, n_classes, device, seed: igual que train_abmil.
        warm_start_state (dict): state_dict del modelo YA convergido (el
            best_state que devuelve abmil_engine.train_abmil() para este
            mismo fold/seed). Es el punto de partida de los ciclos FGE.
        n_cycles (int): número de ciclos FGE. Default: 6.
        cycle_length (int): épocas por ciclo (corto, a diferencia de Snapshot
            Ensemble). Default: 4.
        lr_1 (float): LR techo del ciclo piecewise-linear. Default: 2e-4
            (misma escala que la receta "ours" de abmil_engine).
        lr_2 (float): LR piso del ciclo (valor en el snapshot). Default: 2e-5.
        proj_dim, dropout, wd, val_frac: igual que abmil_engine.train_abmil
            (deben coincidir con los usados para producir warm_start_state).
        cycle_patience (int): si el AUC de validación del snapshot no mejora
            en `cycle_patience` ciclos consecutivos, se descarta el último
            snapshot y se detiene el entrenamiento (early stopping
            inter-ciclo, mismo patrón que SnapshotMLPMetaClassifier).
        include_base (bool): si True, el propio modelo convergido
            (warm_start_state) se devuelve como primer "snapshot". Medido en
            cptac_brca, los snapshots FGE quedaron ~0.04 AUC por debajo del
            modelo convergido en el 70-76 % de los folds; incluirlo hace que
            el promedio sea una ampliación del base y no un reemplazo.
            Default: False (comportamiento original).
        schedule (str): forma de la LR dentro del ciclo — "fge"
            (piecewise-linear, Garipov et al.) o "se" (coseno con reinicios,
            Huang et al. ICLR 2017). Con "se", `lr_1` es la LR del reinicio
            (α0) y `lr_2` la LR mínima al final del ciclo. Lo único que cambia
            entre ambos es esta fórmula: el resto del bucle —snapshots por
            ciclo, early stopping inter-ciclo, include_base— es idéntico, que
            es lo que hace comparables los dos métodos. Default: "fge".

    Returns:
        snapshots (list[dict]): state_dicts, uno por ciclo completado (menos
            si hubo early stopping; más el modelo base si include_base=True).
        val_aucs (list[float]): AUC de validación de cada snapshot conservado.
    """
    stems = list(stems)
    y = list(y)
    groups = np.asarray(groups)

    # Mismo split que generó warm_start_state (mismo seed -> mismo resultado
    # de _grouped_val_split), para que los AUC de snapshot sean comparables
    # al best_auc del modelo convergido original.
    if val_frac:
        tri, vai = _grouped_val_split(groups, y, val_frac, seed)
    else:
        from sklearn.model_selection import GroupKFold
        tri, vai = next(GroupKFold(5).split(stems, y, groups))

    sub = [stems[i] for i in tri]
    suby = [y[i] for i in tri]
    val = [stems[i] for i in vai]
    valy = [y[i] for i in vai]

    R = _RECIPES[ABMIL_RECIPE]

    model = ABMIL(in_dim, n_classes, proj_dim=proj_dim, dropout=dropout).to(device)
    model.load_state_dict(warm_start_state)

    opt = torch.optim.AdamW(model.parameters(), lr=lr_1, weight_decay=wd)

    cnt = np.bincount(np.asarray(suby), minlength=n_classes).astype(np.float64)
    cw = torch.tensor(
        len(suby) / (n_classes * np.clip(cnt, 1.0, None)),
        dtype=torch.float32, device=device
    )
    crit = nn.CrossEntropyLoss(
        label_smoothing=R["smooth"], weight=cw, reduction="sum"
    )

    order = list(range(len(sub)))
    rnd = random.Random(seed + 1)  # offset del seed del modelo convergido

    snapshots, val_aucs = [], []
    best_auc, bad_cycles = -1.0, 0
    total_epochs = n_cycles * cycle_length

    if include_base:
        # El modelo convergido entra como miembro 0 del ensemble. Su AUC no
        # participa en el early stopping: el contador debe medir si los ciclos
        # mejoran entre sí, no si superan a un base que casi siempre es mejor
        # (si no, bad_cycles se dispararía en el ciclo 1 y cortaría de inmediato).
        snapshots.append(copy.deepcopy(model.state_dict()))
        val_aucs.append(_auc(model, feats, val, valy, device))

    for epoch in range(total_epochs):
        lr = cycle_lr(schedule, epoch, cycle_length, lr_1, lr_2)
        set_lr(opt, lr)

        model.train()
        rnd.shuffle(order)
        for j in order:
            opt.zero_grad()
            lo, _ = model(feats[sub[j]].to(device))
            loss = crit(lo.unsqueeze(0), torch.tensor([suby[j]], device=device))
            loss.backward()
            opt.step()

        # Snapshot al final de cada ciclo (mínimo de LR = lr_2)
        if (epoch + 1) % cycle_length == 0:
            va = _auc(model, feats, val, valy, device)
            snapshots.append(copy.deepcopy(model.state_dict()))
            val_aucs.append(va)

            if va > best_auc:
                best_auc = va
                bad_cycles = 0
            else:
                bad_cycles += 1
                if bad_cycles >= cycle_patience:
                    snapshots.pop()
                    val_aucs.pop()
                    break

    return snapshots, val_aucs
