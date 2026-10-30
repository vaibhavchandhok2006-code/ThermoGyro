"""
models.py -- every model used in ThermoGyro, written from scratch in NumPy so
each one maps directly onto the course notes.

  SinglePointCal      : what most hobby code does (subtract one bias measured
                        at start-up). Module 1 "baseline".
  PolyRegression      : hypothesis h(T) = theta_0 + theta_1 T + ... + theta_d T^d,
                        solved with the normal equation W = (X^T X + lambda I')^-1 X^T Y
                        (Module 2 for d=1, lambda=0; Module 3 for polynomial + ridge).
                        Optional dT/dt features (multivariable regression).
  gradient_descent    : batch gradient descent on the same cost, to compare with
                        the closed form (Module 3/4 Sec. 4.2).
  MLP                 : multilayer perceptron, forward pass + backpropagation by
                        hand + Adam optimiser (Module 4).
  HybridModel         : physics/linear backbone + MLP that learns only the
                        residual the backbone misses.

Inputs are standardised (z-scored) using TRAINING statistics only -- the
feature-scaling lesson of Module 4 Sec. 4.2.8. This also keeps X^T X well
conditioned for high polynomial degrees.
"""
import numpy as np


# ----------------------------------------------------------------- utilities
def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def r2(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    sst = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - yhat) ** 2) / sst) if sst > 0 else float("nan")


class Scaler:
    def fit(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mu) / self.sd


# ------------------------------------------------------------ baselines
class SinglePointCal:
    """Subtract one constant bias, measured during a short still period at a
    reference temperature (default 25 degC). This is the 'calibrate once at
    room temperature' approach used by typical MPU-6050 sketches."""
    name = "Single-point calibration"

    def __init__(self, T_ref=25.0, window=2.0):
        self.T_ref, self.window = T_ref, window

    def fit(self, T, y, dTdt=None):
        m = np.abs(T - self.T_ref) <= self.window
        if not m.any():                                  # nearest points instead
            m = np.argsort(np.abs(T - self.T_ref))[:30]
        self.b = float(np.mean(y[m]))
        return self

    def predict(self, T, dTdt=None):
        return np.full_like(np.asarray(T, float), self.b)


# ------------------------------------------------- polynomial / ridge regression
def design_matrix(Tz, dz=None, degree=1, use_rate=False):
    cols = [np.ones_like(Tz)] + [Tz ** k for k in range(1, degree + 1)]
    if use_rate and dz is not None:
        cols += [dz, dz * Tz]          # heating/cooling term and its T interaction
    return np.column_stack(cols)


class PolyRegression:
    def __init__(self, degree=1, lam=0.0, use_rate=False):
        self.degree, self.lam, self.use_rate = degree, lam, use_rate

    @property
    def name(self):
        s = f"Poly deg {self.degree}"
        if self.lam > 0:
            s += f", ridge lambda={self.lam:g}"
        if self.use_rate:
            s += " + dT/dt"
        return s

    def _X(self, T, dTdt):
        Tz = (np.asarray(T, float) - self.Tm) / self.Ts
        dz = None if dTdt is None else (np.asarray(dTdt, float) - self.dm) / self.ds
        return design_matrix(Tz, dz, self.degree, self.use_rate)

    def fit(self, T, y, dTdt=None):
        T = np.asarray(T, float)
        self.Tm, self.Ts = T.mean(), T.std() or 1.0
        if dTdt is not None:
            self.dm, self.ds = float(np.mean(dTdt)), float(np.std(dTdt) or 1.0)
        X = self._X(T, dTdt)
        I_ = np.eye(X.shape[1])
        I_[0, 0] = 0.0                                   # never penalise the intercept
        A = X.T @ X + self.lam * I_                      # (X^T X + lambda I')
        self.W = np.linalg.solve(A, X.T @ np.asarray(y, float))
        self.cond = float(np.linalg.cond(X.T @ X))
        return self

    def predict(self, T, dTdt=None):
        return self._X(T, dTdt) @ self.W


def gradient_descent(X, y, alpha, iters, W0=None):
    """Batch gradient descent on J = 1/(2N) ||XW - Y||^2.
    Returns final W and the cost history (Module 4, Eq. 4.2.9)."""
    N = len(y)
    W = np.zeros(X.shape[1]) if W0 is None else W0.copy()
    hist = []
    for _ in range(iters):
        E = X @ W - y
        hist.append(float(E @ E / (2 * N)))
        if not np.isfinite(hist[-1]) or hist[-1] > 1e12:
            break
        W = W - alpha / N * (X.T @ E)                    # simultaneous update
    return W, np.array(hist)


# ----------------------------------------------------------------------- MLP
class MLP:
    """Fully connected network: inputs -> [hidden]*L (ReLU or tanh) -> 1 linear
    output. Trained on J = 1/2 mean (o - y)^2 with Adam (Module 4, Sec. 4.14.5).
    Backpropagation follows Eq. 4.8.7-4.8.9 exactly."""

    def __init__(self, n_in, hidden=(16, 16), act="tanh", lr=3e-3, epochs=3000,
                 l2=1e-4, seed=0, batch=None):
        self.sizes = [n_in, *hidden, 1]
        self.act, self.lr, self.epochs, self.l2 = act, lr, epochs, l2
        self.batch = batch
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(self.sizes) - 1):
            fan_in = self.sizes[i]
            gain = 2.0 if act == "relu" else 1.0          # He vs Xavier-style init
            self.W.append(rng.normal(0, np.sqrt(gain / fan_in), (self.sizes[i + 1], fan_in)))
            self.b.append(np.full(self.sizes[i + 1], 0.01 if act == "relu" else 0.0))
        self.rng = rng

    def _f(self, z):
        return np.maximum(0, z) if self.act == "relu" else np.tanh(z)

    def _df(self, z, a):
        return (z > 0).astype(float) if self.act == "relu" else 1 - a ** 2

    def forward(self, X):
        A, Z = [X.T], []
        for l in range(len(self.W)):
            z = self.W[l] @ A[-1] + self.b[l][:, None]
            Z.append(z)
            A.append(z if l == len(self.W) - 1 else self._f(z))
        return Z, A

    def fit(self, X, y, X_val=None, y_val=None, patience=400):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.xs = Scaler().fit(X)
        self.ym, self.ysd = y.mean(), y.std() or 1.0
        Xs, ys = self.xs.transform(X), (y - self.ym) / self.ysd
        if X_val is not None:
            Xv, yv = self.xs.transform(np.asarray(X_val, float)), np.asarray(y_val, float)
        mW = [np.zeros_like(w) for w in self.W]; vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]; vb = [np.zeros_like(b) for b in self.b]
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        self.loss_hist, self.val_hist = [], []
        best, best_state, wait = np.inf, None, 0
        N = len(ys)
        for ep in range(self.epochs):
            idx = self.rng.permutation(N) if self.batch else np.arange(N)
            bs = self.batch or N
            for s in range(0, N, bs):
                bi = idx[s:s + bs]
                Z, A = self.forward(Xs[bi])
                n = len(bi)
                delta = (A[-1] - ys[bi][None, :])          # dJ/dz at linear output = o - y
                gW, gb = [None] * len(self.W), [None] * len(self.W)
                for l in reversed(range(len(self.W))):
                    gW[l] = delta @ A[l].T / n + self.l2 * self.W[l]
                    gb[l] = delta.mean(1)
                    if l > 0:                              # delta^(l) = (W^T delta) * f'(z)
                        delta = (self.W[l].T @ delta) * self._df(Z[l - 1], A[l])
                t += 1
                for l in range(len(self.W)):               # Adam update
                    mW[l] = b1 * mW[l] + (1 - b1) * gW[l]; vW[l] = b2 * vW[l] + (1 - b2) * gW[l] ** 2
                    mb[l] = b1 * mb[l] + (1 - b1) * gb[l]; vb[l] = b2 * vb[l] + (1 - b2) * gb[l] ** 2
                    self.W[l] -= self.lr * (mW[l] / (1 - b1 ** t)) / (np.sqrt(vW[l] / (1 - b2 ** t)) + eps)
                    self.b[l] -= self.lr * (mb[l] / (1 - b1 ** t)) / (np.sqrt(vb[l] / (1 - b2 ** t)) + eps)
            if ep % 10 == 0 or ep == self.epochs - 1:
                self.loss_hist.append(rmse(ys, self.forward(Xs)[1][-1][0]) * self.ysd)
                if X_val is not None:
                    v = rmse(yv, self._pred_scaled(Xv))
                    self.val_hist.append(v)
                    if v < best - 1e-6:
                        best, wait = v, 0
                        best_state = ([w.copy() for w in self.W], [b.copy() for b in self.b])
                    else:
                        wait += 10
                        if wait >= patience:
                            break
        if best_state is not None:                         # early stopping restore
            self.W, self.b = best_state
        self.epochs_run = ep + 1
        return self

    def _pred_scaled(self, Xs):
        return self.forward(Xs)[1][-1][0] * self.ysd + self.ym

    def predict(self, X):
        return self._pred_scaled(self.xs.transform(np.asarray(X, float)))

    def dead_units(self, X):
        """Count ReLU units that output 0 for EVERY input (dying-ReLU check)."""
        if self.act != "relu":
            return 0
        _, A = self.forward(self.xs.transform(np.asarray(X, float)))
        return int(sum((a.max(1) <= 0).sum() for a in A[1:-1]))

    def export(self):
        return {"act": self.act, "sizes": self.sizes,
                "W": [w.tolist() for w in self.W], "b": [b.tolist() for b in self.b],
                "x_mu": self.xs.mu.tolist(), "x_sd": self.xs.sd.tolist(),
                "y_mu": self.ym, "y_sd": self.ysd}


# ------------------------------------------------------------- hybrid model
class HybridModel:
    """Backbone (low-degree polynomial in T, interpretable, extrapolates
    sensibly) + small MLP trained on the backbone's residual using [T, dT/dt].

    Range guard (Module 5, "regime awareness"): the MLP correction is used
    only for temperatures inside the training range (plus `margin` degC).
    Outside it the model returns the backbone alone and flags the prediction
    as extrapolated. The correction is also clipped to the residual range
    seen in training."""

    def __init__(self, degree=2, lam=0.0, mlp_kwargs=None, guard=True, margin=0.0):
        self.backbone = PolyRegression(degree, lam)
        self.mlp_kwargs = mlp_kwargs or {}
        self.guard, self.margin = guard, margin
        self.name = f"Hybrid (poly{degree} + MLP residual)"

    def in_range(self, T):
        T = np.asarray(T, float)
        return (T >= self.Tlo - self.margin) & (T <= self.Thi + self.margin)

    def fit(self, T, y, dTdt, T_val=None, y_val=None, d_val=None):
        # The physics backbone sees ALL training data (train + early-stopping
        # blocks). The validation blocks are used only to early-stop the MLP.
        if T_val is not None:
            self.backbone.fit(np.r_[T, T_val], np.r_[y, y_val])
        else:
            self.backbone.fit(T, y)
        res = y - self.backbone.predict(T)
        self.rlo, self.rhi = float(res.min()), float(res.max())
        X = np.column_stack([T, dTdt])
        kw = dict(n_in=2, hidden=(16, 16), act="tanh", lr=3e-3, epochs=3000)
        kw.update(self.mlp_kwargs)
        self.mlp = MLP(**kw)
        if T_val is not None:
            rv = y_val - self.backbone.predict(T_val)
            self.mlp.fit(X, res, np.column_stack([T_val, d_val]), rv)
        else:
            self.mlp.fit(X, res)
        Tall = T if T_val is None else np.r_[T, T_val]
        self.Tlo, self.Thi = float(np.min(Tall)), float(np.max(Tall))
        return self

    def predict(self, T, dTdt):
        T = np.asarray(T, float)
        corr = np.clip(self.mlp.predict(np.column_stack([T, dTdt])), self.rlo, self.rhi)
        if self.guard:
            corr = np.where(self.in_range(T), corr, 0.0)
        return self.backbone.predict(T) + corr
