"""
Práctica 9 - Few-shot Learning con MNIST
Código base para alumnos

Objetivo:
    Simular un escenario few-shot en el que el modelo se entrena sin ver
    la clase 7 y posteriormente intenta reconocerla usando pocos ejemplos
    mediante prototipos en un espacio de embeddings.
"""

import os
os.environ["KERAS_BACKEND"] = "jax"

import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import keras
from keras import layers, models
from keras.src.backend import random as keras_random


SEED = 42
np.random.seed(SEED)
random.seed(SEED)
keras.utils.set_random_seed(SEED)


@dataclass
class FewShotData:
    X_train_known: np.ndarray
    y_train_known: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    known_classes: np.ndarray
    novel_class: int


def load_mnist_world_without_sevens() -> FewShotData:
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    mask_known = y_train != 7
    X_train_known = X_train[mask_known]
    y_train_known = y_train[mask_known]

    known_classes = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9])

    return FewShotData(
        X_train_known=X_train_known,
        y_train_known=y_train_known,
        X_test=X_test,
        y_test=y_test,
        known_classes=known_classes,
        novel_class=7,
    )


def build_classifier(num_classes: int) -> keras.Model:
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu", name="embedding"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def remap_known_labels(y: np.ndarray, known_classes: np.ndarray) -> np.ndarray:
    mapping = {orig: i for i, orig in enumerate(known_classes)}
    return np.array([mapping[label] for label in y])


def create_feature_extractor(classifier: keras.Model) -> keras.Model:
    # Build the extractor from the classifier's layers (avoiding .input/.output)
    inp = layers.Input(shape=(28, 28, 1))
    x = inp
    for layer in classifier.layers:
        x = layer(x)
        if layer.name == "embedding":
            break
    return models.Model(inputs=inp, outputs=x)


def sample_support_set(X: np.ndarray, y: np.ndarray, class_label: int, n_shots: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.where(y == class_label)[0]
    selected = np.random.choice(indices, size=n_shots, replace=False)
    return X[selected], y[selected]


def compute_prototypes(feature_extractor: tf.keras.Model, X_support: np.ndarray, y_support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    embeddings = feature_extractor.predict(X_support, verbose=0)

    unique_classes = np.unique(y_support)
    prototypes = []
    prototype_labels = []

    for cls in unique_classes:
        mask = y_support == cls
        proto = embeddings[mask].mean(axis=0)
        prototypes.append(proto)
        prototype_labels.append(cls)

    return np.array(prototypes), np.array(prototype_labels)


def classify_by_nearest_prototype(feature_extractor: tf.keras.Model, X_query: np.ndarray, prototypes: np.ndarray, prototype_labels: np.ndarray) -> np.ndarray:
    query_embeddings = feature_extractor.predict(X_query, verbose=0)

    distances = np.linalg.norm(query_embeddings[:, np.newaxis, :] - prototypes[np.newaxis, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)

    return prototype_labels[nearest]


def build_fewshot_episode(data: FewShotData, n_shots: int, n_query_per_class: int = 100):
    X_support_list, y_support_list = [], []
    X_query_list, y_query_list = [], []

    X_train = data.X_train_known
    y_train = data.y_train_known
    X_test = data.X_test
    y_test = data.y_test

    for cls in range(10):
        if cls in data.known_classes:
            sup_X, sup_y = sample_support_set(X_train, y_train, cls, n_shots)
        else:
            test_indices = np.where(y_test == cls)[0]
            chosen = np.random.choice(test_indices, size=n_shots, replace=False)
            sup_X = X_test[chosen]
            sup_y = y_test[chosen]

        X_support_list.append(sup_X)
        y_support_list.append(sup_y)

        test_indices = np.where(y_test == cls)[0]
        qry_indices = np.random.choice(test_indices, size=n_query_per_class, replace=False)
        X_query_list.append(X_test[qry_indices])
        y_query_list.append(y_test[qry_indices])

    X_support = np.concatenate(X_support_list, axis=0)
    y_support = np.concatenate(y_support_list, axis=0)
    X_query = np.concatenate(X_query_list, axis=0)
    y_query = np.concatenate(y_query_list, axis=0)

    return X_support, y_support, X_query, y_query


def plot_accuracy_comparison(results: dict[str, float], output_path: str = "fewshot_accuracy_comparison.png") -> None:
    plt.figure(figsize=(6, 4))
    bars = plt.bar(list(results.keys()), list(results.values()), color=["#ff7f0e", "#1f77b4"])
    for bar, val in zip(bars, results.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=11)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Few-shot classification: 1-shot vs 5-shot")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()


def plot_embeddings_pca(feature_extractor: tf.keras.Model, X: np.ndarray, y: np.ndarray, prototypes: np.ndarray | None = None, prototype_labels: np.ndarray | None = None, output_path: str = "fewshot_embeddings_pca.png") -> None:
    embeddings = feature_extractor.predict(X, verbose=0)

    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=y, cmap="tab10", alpha=0.6, s=10)

    if prototypes is not None:
        proto_2d = pca.transform(prototypes)
        plt.scatter(proto_2d[:, 0], proto_2d[:, 1], c=prototype_labels, cmap="tab10",
                    marker="X", s=200, edgecolors="black", linewidths=1.5, label="Prototypes")

    plt.colorbar(scatter, label="Digit class")
    plt.title("2D PCA of MNIST embeddings (few-shot feature space)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()


def main() -> None:
    data = load_mnist_world_without_sevens()

    y_train_known_remap = remap_known_labels(data.y_train_known, data.known_classes)
    classifier = build_classifier(num_classes=len(data.known_classes))

    classifier.fit(
        data.X_train_known,
        y_train_known_remap,
        validation_split=0.1,
        epochs=3,
        batch_size=128,
        verbose=1,
    )

    feature_extractor = create_feature_extractor(classifier)

    sample_emb = feature_extractor.predict(data.X_test[:1], verbose=0)
    print(f"Embedding dimension: {sample_emb.shape[-1]}")

    results = {}
    for n_shots in [1, 5]:
        X_support, y_support, X_query, y_query = build_fewshot_episode(data, n_shots=n_shots)
        prototypes, prototype_labels = compute_prototypes(feature_extractor, X_support, y_support)
        y_pred = classify_by_nearest_prototype(feature_extractor, X_query, prototypes, prototype_labels)
        acc = accuracy_score(y_query, y_pred)
        results[f"{n_shots}-shot"] = acc
        print(f"{n_shots}-shot accuracy: {acc:.4f}")

    plot_accuracy_comparison(results)

    # Build a combined query set for PCA visualization
    _, _, viz_X, viz_y = build_fewshot_episode(data, n_shots=1, n_query_per_class=50)
    # Build a support set to mark prototype positions on the PCA plot
    sup_X, sup_y, _, _ = build_fewshot_episode(data, n_shots=5, n_query_per_class=0)
    viz_protos, viz_proto_labels = compute_prototypes(feature_extractor, sup_X, sup_y)
    plot_embeddings_pca(feature_extractor, viz_X, viz_y, prototypes=viz_protos, prototype_labels=viz_proto_labels)


if __name__ == "__main__":
    main()
