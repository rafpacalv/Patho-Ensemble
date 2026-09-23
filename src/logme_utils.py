"""LogME (You, Zhang, Hu, Wang, Long. "LogME: Practical Assessment of
Pre-trained Models for Transfer Learning", ICML 2021) — evidencia marginal
logarítmica de un modelo lineal bayesiano `y ~ f`, calculable en forma
cerrada sin entrenar nada. Sirve como criterio de selección *a priori*: si
un foundational model va a rendir bien en esta tarea, estimado directamente
sobre sus embeddings congelados y las etiquetas, sin pasar por el ABMIL.

Vendorizado (sin la dependencia de `numba`, que no está en environment.yml;
a esta escala — decenas de folds, cientos de filas, 512 D — la versión JIT
no aporta nada medible) desde el repo oficial thuml/LogME
(https://github.com/thuml/LogME, MIT license), método `_fit_fixed_point`
("Ranking and Tuning Pre-trained Models: A New Paradigm of Exploiting Model
Hubs", https://arxiv.org/abs/2110.10545). Lógica sin modificar; solo se
quitaron los decoradores `@njit` y las llamadas de precompilación.
"""
import warnings

import numpy as np


def _truncated_svd(x):
    u, s, vh = np.linalg.svd(x.transpose() @ x)
    s = np.sqrt(s)
    u_times_sigma = x @ vh.transpose()
    k = np.sum((s > 1e-10) * 1)  # rank de f
    s = s.reshape(-1, 1)
    s = s[:k]
    vh = vh[:k]
    u = u_times_sigma[:, :k] / s.reshape(1, -1)
    return u, s, vh


class LogME(object):
    def __init__(self, regression=False):
        """:param regression: si es un problema de regresión (o clasificación)."""
        self.regression = regression
        self.fitted = False
        self.reset()

    def reset(self):
        self.num_dim = 0
        self.alphas = []
        self.betas = []
        self.ms = []

    def _fit_fixed_point(self, f: np.ndarray, y: np.ndarray):
        N, D = f.shape  # k = min(N, D)
        if N > D:
            u, s, vh = _truncated_svd(f)
        else:
            u, s, vh = np.linalg.svd(f, full_matrices=False)
        s = s.reshape(-1, 1)
        sigma = (s ** 2)

        evidences = []
        self.num_dim = y.shape[1] if self.regression else int(y.max() + 1)
        for i in range(self.num_dim):
            y_ = y[:, i] if self.regression else (y == i).astype(np.float64)
            y_ = y_.reshape(-1, 1)
            x = u.T @ y_
            x2 = x ** 2
            res_x2 = (y_ ** 2).sum() - x2.sum()

            alpha, beta = 1.0, 1.0
            t = alpha / beta
            for _ in range(11):
                t = alpha / beta
                gamma = (sigma / (sigma + t)).sum()
                m2 = (sigma * x2 / ((t + sigma) ** 2)).sum()
                res2 = (x2 / ((1 + sigma / t) ** 2)).sum() + res_x2
                alpha = gamma / (m2 + 1e-5)
                beta = (N - gamma) / (res2 + 1e-5)
                t_ = alpha / beta
                if abs(t_ - t) / t <= 1e-3:
                    break
            evidence = D / 2.0 * np.log(alpha) \
                       + N / 2.0 * np.log(beta) \
                       - 0.5 * np.sum(np.log(alpha + beta * sigma)) \
                       - beta / 2.0 * res2 \
                       - alpha / 2.0 * m2 \
                       - N / 2.0 * np.log(2 * np.pi)
            evidence /= N
            m = 1.0 / (t + sigma) * s * x
            m = (vh.T @ m).reshape(-1)
            evidences.append(evidence)
            self.alphas.append(alpha)
            self.betas.append(beta)
            self.ms.append(m)
        self.ms = np.stack(self.ms)
        return np.mean(evidences)

    _fit = _fit_fixed_point

    def fit(self, f: np.ndarray, y: np.ndarray):
        """:param f: [N, D] embeddings. :param y: [N] enteros de clase (o [N, C] si regression)."""
        if self.fitted:
            warnings.warn("re-fitting para nuevos datos; se limpian los parámetros anteriores.")
            self.reset()
        else:
            self.fitted = True
        f = f.astype(np.float64)
        if self.regression:
            y = y.astype(np.float64)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
        return self._fit(f, y)
