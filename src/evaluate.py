import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
import shap


TARGET_NAMES = ["Low", "Moderate", "High"]


def print_metrics(y_true, y_pred, model_name: str = "Model"):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    print(f"\n=== {model_name} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Macro : {f1:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=TARGET_NAMES))
    return {"accuracy": acc, "f1_macro": f1}


def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES, ax=ax)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    return fig


def plot_shap_summary(model, X_test, feature_display_names):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    plt.figure(figsize=(9, 5))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_display_names,
        class_names=TARGET_NAMES,
        plot_type="bar",
        max_display=14,
        show=True
    )
    return shap_values
