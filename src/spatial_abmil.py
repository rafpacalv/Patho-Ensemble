"""
ABMIL con message passing sobre el grafo espacial de tiles, ANTES del pooling
de atención — es decir, grafo *además* del mecanismo de atención, no en su
lugar.

No toca `abmil_engine.ABMIL`: los checkpoints ya entrenados siguen cargando con
la clase original y sin condicionales de versión. `GatedAttention` sí se
reutiliza por import, para que cualquier diferencia frente a ABMIL sea
atribuible al bloque de grafo y no al pooling.

La escalera de capacidad es deliberada y va de MENOS a MÁS, igual que la de
meta_models.py. El hallazgo mejor documentado del repo es que subir capacidad
ha perjudicado de forma monótona (7 meta-learners, de +0.0003 a −0.2283 AUC, y
ninguna de 39 configuraciones significativa), así que arrancar con un
GraphSAGE completo sería empezar justo en el régimen que ya ha fallado cuatro
veces:

  - `smooth`   : h + alpha*(media_vecinos(h) - h). UN escalar por capa, +0.4 %
                 de coste. Contesta "¿ayuda el promediado espacial?" sin ningún
                 confound de capacidad.
  - `sage_bn`  : cuello de botella 512->128->512.   ~132 k parámetros por capa.
  - `sage`     : GraphSAGE-mean, W_self + W_neigh.  ~525 k parámetros por capa.

`smooth` arranca con alpha = 0, así que en la inicialización SpatialABMIL es
NUMÉRICAMENTE IDÉNTICO a ABMIL y sólo se aleja si el gradiente lo pide. Eso
convierte el alpha aprendido en un diagnóstico directo: si converge a ~0, la
lectura es "no hay señal espacial que extraer", que es mucho más interpretable
que un delta de AUC dentro del ruido.

Profundidad por defecto K=2. El over-smoothing preocupa menos de lo que sugiere
la literatura: son retículas 2-D de diámetro ~sqrt(N) (unos 85 nodos para la
mediana de 7174 tiles), y K=3 mezcla sobre una ventana 7x7, 49 de 7174 nodos
(0.7 %). La restricción real es física: 448 px de nivel 0 a 40x son ~112 um por
tile, así que K=2 da un contexto de ~560 um — la escala de nido tumoral y banda
estromal. K>=4 empieza a promediar entre compartimentos de tejido, que es un
error de modelado antes que uno de suavizado.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from abmil_engine import GatedAttention


ARCHS = ("smooth", "sage_bn", "sage")


class GraphLayer(nn.Module):
    """Una ronda de message passing con media de vecinos y residual.

    El residual sustituye al self-loop: el nodo conserva su señal sin inflar el
    número de aristas un 12.5 %.

    Args:
        dim: dimensión de trabajo (proj_dim de SpatialABMIL).
        mode: 'smooth' | 'sage_bn' | 'sage'.
        bottleneck: dimensión interna del modo sage_bn.
        dropout: dropout del residual (no aplica a 'smooth').
    """

    def __init__(self, dim, mode="smooth", bottleneck=128, dropout=0.25,
                 alpha_init=0.0):
        super().__init__()
        if mode not in ARCHS:
            raise ValueError(f"mode debe ser uno de {ARCHS}, recibido {mode!r}")
        self.mode = mode

        if mode == "smooth":
            # alpha_init=0 -> capa identidad en la inicialización, que es lo que
            # hace demostrable la inercia del cableado.
            #
            # Pero arrancar en 0 tiene un coste que sólo se ve al leer el
            # resultado: si el alpha aprendido se queda en ~0, eso NO distingue
            # "el contexto espacial no aporta" de "el gradiente nunca llegó a
            # moverlo". Arrancar en un valor sustantivo (0.3) y ver si el
            # entrenamiento lo empuja de vuelta a 0 sí separa las dos hipótesis:
            # un descenso desde 0.3 es evidencia activa de que el modelo
            # rechaza el promediado, no ausencia de evidencia.
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        elif mode == "sage_bn":
            self.down = nn.Linear(dim, bottleneck)
            self.up = nn.Linear(bottleneck, dim)
            self.norm = nn.LayerNorm(dim)
            self.drop = nn.Dropout(dropout)
        else:
            self.w_self = nn.Linear(dim, dim)
            self.w_neigh = nn.Linear(dim, dim, bias=False)
            self.norm = nn.LayerNorm(dim)
            self.drop = nn.Dropout(dropout)

    def forward(self, h, graph):
        """h (N,dim); graph: SlideGraph, que despacha la agregación por device."""
        if self.mode == "smooth":
            return h + self.alpha * (graph.aggregate(h) - h)

        if self.mode == "sage_bn":
            z = self.up(F.gelu(graph.aggregate(self.down(h))))
        else:
            z = self.w_self(h) + self.w_neigh(graph.aggregate(h))

        return h + self.drop(F.gelu(self.norm(z)))


class SpatialABMIL(nn.Module):
    """projection -> K x GraphLayer -> GatedAttention -> classifier.

    Devuelve `(logits (C,), attention (N,))`, la MISMA firma de salida que
    `ABMIL.forward`. Eso es lo que permite que `_auc` y `abmil_predict` del
    motor sirvan sin cambios.

    proj_dim se queda en 512: el grafo opera en el mismo espacio que ya consume
    la atención, así que no aparece un hiperparámetro de anchura nuevo que
    barrer.
    """

    def __init__(self, in_dim, num_classes, hidden_dim=256, proj_dim=512,
                 dropout=0.25, n_layers=2, mode="smooth", bottleneck=128,
                 alpha_init=0.0):
        super().__init__()
        if proj_dim <= 0:
            raise ValueError("SpatialABMIL requiere proj_dim > 0")

        self.mode = mode
        self.n_layers = n_layers
        self.alpha_init = alpha_init

        self.projection = nn.Sequential(
            nn.Linear(in_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU()
        )
        self.layers = nn.ModuleList(
            GraphLayer(proj_dim, mode, bottleneck, dropout, alpha_init)
            for _ in range(n_layers)
        )
        self.attention = GatedAttention(proj_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(proj_dim, num_classes)
        )

    def forward(self, h, graph):
        h = self.projection(h)
        for layer in self.layers:
            h = layer(h, graph)
        a = self.attention(h)
        z = (a * h).sum(0, keepdim=True)
        return self.classifier(z).squeeze(0), a.squeeze(1)

    def alphas(self):
        """Los alpha aprendidos, en modo 'smooth'. Diagnóstico principal:
        alpha ~ 0 significa que el modelo ha descartado el promediado espacial."""
        if self.mode != "smooth":
            return None
        return [float(l.alpha) for l in self.layers]


def make_factory(arch, n_layers=2, bottleneck=128, alpha_init=0.0):
    """Devuelve un constructor con la firma que espera `train_abmil` del motor:
    `factory(in_dim, n_classes, proj_dim=..., dropout=...)`."""
    def factory(in_dim, num_classes, proj_dim=512, dropout=0.25):
        return SpatialABMIL(
            in_dim, num_classes, proj_dim=proj_dim, dropout=dropout,
            n_layers=n_layers, mode=arch, bottleneck=bottleneck,
            alpha_init=alpha_init,
        )
    return factory


def attention_stats(a):
    """(ESS, masa del top-1 %) de un vector de atención.

    El riesgo real de este diseño no es el over-smoothing sino el COLAPSO DE
    ATENCIÓN: promediar vecindarios hace los tiles más parecidos, el softmax
    sobre el eje de parches se aplana y el pooling degenera hacia una media
    simple — perdiendo justo la propiedad "encuentra los pocos tiles
    discriminativos" que hace funcionar a ABMIL. Con N=29622 el peso uniforme
    es 3.4e-5. Si el ESS sube mucho con el AUC plano, ese es el mecanismo, y el
    arreglo es menos capas o alpha menor, no más capacidad.
    """
    a = a.detach().float().flatten()
    ess = float(1.0 / (a ** 2).sum())
    k = max(1, int(round(0.01 * len(a))))
    top = float(a.sort(descending=True).values[:k].sum())
    return ess, top
