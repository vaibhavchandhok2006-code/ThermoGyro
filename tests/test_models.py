"""Sanity tests -- run with:  python -m pytest -q tests/
1. Backprop gradients match central finite differences (Module 4, Sec 4.8.7).
2. Normal equation reproduces the Module-2 spring example (theta = 0.10, 1.96).
3. Gradient descent converges to the normal-equation answer.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models import MLP, PolyRegression, gradient_descent, design_matrix  # noqa: E402


def _loss(net, X, y):
    o = net.forward(X)[1][-1][0]
    return 0.5 * np.mean((o - y) ** 2)


def test_backprop_gradient_check():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(20, 2)), rng.normal(size=20)
    for act in ("tanh", "relu"):
        net = MLP(2, hidden=(5, 4), act=act, seed=3)
        Z, A = net.forward(X)
        delta = A[-1] - y[None, :]
        grads = [None] * len(net.W)
        for l in reversed(range(len(net.W))):
            grads[l] = delta @ A[l].T / len(y)
            if l > 0:
                delta = (net.W[l].T @ delta) * net._df(Z[l - 1], A[l])
        eps = 1e-6
        for l in range(len(net.W)):
            for (i, j) in [(0, 0), (net.W[l].shape[0] - 1, net.W[l].shape[1] - 1)]:
                old = net.W[l][i, j]
                net.W[l][i, j] = old + eps; jp = _loss(net, X, y)
                net.W[l][i, j] = old - eps; jm = _loss(net, X, y)
                net.W[l][i, j] = old
                num = (jp - jm) / (2 * eps)
                assert abs(num - grads[l][i, j]) < 1e-6, (act, l, num, grads[l][i, j])


def test_normal_equation_spring():
    x = np.array([1, 2, 3, 4.0]); y = np.array([2.1, 3.9, 6.1, 7.9])
    X = np.column_stack([np.ones(4), x])
    W = np.linalg.solve(X.T @ X, X.T @ y)
    assert np.allclose(W, [0.10, 1.96], atol=1e-9)


def test_gd_matches_closed_form():
    rng = np.random.default_rng(0)
    T = rng.uniform(-1, 1, 200); y = 0.3 + 1.2 * T - 0.5 * T ** 2 + 0.01 * rng.normal(size=200)
    X = design_matrix(T, degree=2)
    W_ne = np.linalg.solve(X.T @ X, X.T @ y)
    W_gd, _ = gradient_descent(X, y, alpha=0.5, iters=20000)
    assert np.allclose(W_gd, W_ne, atol=1e-6)


if __name__ == "__main__":
    test_backprop_gradient_check(); test_normal_equation_spring(); test_gd_matches_closed_form()
    print("all tests passed")
