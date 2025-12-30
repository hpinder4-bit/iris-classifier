import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an Iris Decision Tree classifier.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction (e.g. 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (e.g. 42)")
    args = parser.parse_args()

    # Outputs folder relative to project root
    project_root = Path(__file__).resolve().parents[1]
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Train
    model = DecisionTreeClassifier(random_state=args.random_state)
    model.fit(X_train, y_train)

    # Predict + evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # Confusion matrix -> save figure
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    disp.plot()
    plt.tight_layout()
    out_path = outputs_dir / "confusion_matrix.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix to: {out_path}")


if __name__ == "__main__":
    main()
