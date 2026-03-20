import os
import numpy as np
import matplotlib.pyplot as plt

def make_dataset(n=500, seed=42):
    rng = np.random.default_rng(seed)
    X = 100.0 * rng.random((n, 1))
    y = 50.0 + 1.2 * X + rng.normal(0, 5, size=(n, 1))
    return X, y

def normalize_zscore(X):
    return (X - X.mean()) / X.std()

def add_bias(X_norm):
    n = X_norm.shape[0]
    return np.hstack([np.ones((n, 1)), X_norm])

def minibatch_gd(X_b, y, alpha=0.5, n_epochs=100, batch_size=32, seed=42, tol=None):
    rng = np.random.default_rng(seed)
    m, n = X_b.shape
    theta = rng.normal(size=(n, 1))
    history = []
    prev = np.inf

    for epoch in range(n_epochs):
        idx = rng.permutation(m)
        Xs, ys = X_b[idx], y[idx]

        for i in range(0, m, batch_size):
            xi = Xs[i:i+batch_size]
            yi = ys[i:i+batch_size]
            grad = (2.0 / xi.shape[0]) * xi.T @ (xi @ theta - yi)
            theta -= alpha * grad

        loss = np.mean((X_b @ theta - y) ** 2)
        history.append(loss)

        if tol is not None and abs(loss - prev) < tol:
            return theta, np.array(history), epoch + 1
        prev = loss

    return theta, np.array(history), n_epochs

if __name__ == "__main__":
    X, y = make_dataset()
    Xn = normalize_zscore(X)
    Xb = add_bias(Xn)

    alpha = 0.5
    n_epochs = 100
    batch_size = 32
    tol = None

    theta, history, epochs_run = minibatch_gd(Xb, y, alpha=alpha, n_epochs=n_epochs, batch_size=batch_size, tol=tol)

    plt.plot(history, linewidth=2)
    plt.title(f"Convergencia (alpha={alpha}, batch={batch_size})")
    plt.xlabel("Épocas")
    plt.ylabel("Coste J(theta)")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/loss_alpha_{alpha}_batch_{batch_size}.png", dpi=200)
    plt.show()
    print("Épocas ejecutadas:", epochs_run)
