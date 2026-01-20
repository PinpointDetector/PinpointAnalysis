from pathlib import Path

import hist
import matplotlib.pyplot as plt
import mplhep
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize


def plot_training_metrics(
    input_path: str | Path = "training_metrics.npz",
    figure_path: str | Path | None = None,
) -> None:
    training_metrics = np.load(input_path)
    train_losses = training_metrics["train_losses"]
    train_accuracies = training_metrics["train_accuracies"]
    val_losses = training_metrics["val_losses"]
    val_accuracies = training_metrics["val_accuracies"]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    # loss
    ax = axs[0]
    ax.plot(train_losses, label="Training")
    ax.plot(val_losses, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    # accuracy
    ax = axs[1]
    ax.plot(train_accuracies, label="Training")
    ax.plot(val_accuracies, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    #
    plt.tight_layout()
    if figure_path is not None:
        plt.savefig(figure_path)
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    normalize: bool = False,
    figure_path: str | Path | None = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_counts = cm.copy()  # Keep original counts

    if normalize:
        cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cbar_kws = {"label": "Norm. Number of Events"}
    else:
        cbar_kws = {"label": "Number of Events"}

    fig, ax = plt.subplots(figsize=(7.5, 6))

    # Create annotations with both normalized and count values
    if normalize:
        # Create custom annotations: "normalized\n(count)"
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm_normalized[i, j]:.2f}\n({cm_counts[i, j]})"

        sns.heatmap(
            cm_normalized,
            annot=annot,
            fmt="",  # Use custom annotations
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws=cbar_kws,
            vmin=vmin,
            vmax=vmax,
        )
    else:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws=cbar_kws,
        )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_aspect("equal")
    plt.tight_layout()
    if figure_path is not None:
        plt.savefig(figure_path)
    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    figure_path: str | Path | None = None,
):
    """Multi-class ROC Curves (One-vs-Rest)"""
    if len(class_names) == 2:
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])[:, :2]
    else:
        y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))

    fpr = {}
    tpr = {}
    roc_auc = {}
    for i, class_name in enumerate(class_names):
        fpr[class_name], tpr[class_name], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[class_name] = auc(fpr[class_name], tpr[class_name])
    # Micro-average ROC curve and AUC
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    fig, ax = plt.subplots(figsize=(6, 4))
    for key in fpr.keys():
        ax.plot(fpr[key], tpr[key], label=f"{key} (AUC = {roc_auc[key]:.2f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("")
    ax.legend()
    plt.tight_layout()
    # if figure_path is not None:
    #     plt.savefig(figure_path)
    plt.show()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    figure_path: str | Path | None = None,
):
    """
    Recall: TP / (TP + FN)
    Precision: TP / (TP + FP)
    """
    precision_dict = {}
    recall_dict = {}
    y_true_bin = label_binarize(y_true, classes=range(3))
    for i, label in enumerate(class_names):
        precision_dict[label], recall_dict[label], _ = precision_recall_curve(
            y_true_bin[:, i], y_prob[:, i]
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    for key in class_names:
        ax.plot(recall_dict[key], precision_dict[key], label=key)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    plt.tight_layout()
    if figure_path is not None:
        plt.savefig(figure_path)
    plt.show()


# DEPRECATED?
# def plot_probabilities(
#     y_true: np.ndarray,
#     y_prob: np.ndarray,
#     class_names: list[str],
#     num_bins: int = 50,
#     figures_path: str | Path | None = None,
# ) -> None:
#     h = hist.Hist(hist.axis.Regular(num_bins, 0, 1))
#     bins = h.axes[0].edges

#     for true_idx, true_label in enumerate(class_names):
#         y_prob_class = y_prob[y_true == true_idx]
#         fig, ax = plt.subplots(figsize=(6, 4))
#         for pred_idx, pred_label in enumerate(class_names):
#             h.reset()
#             h.fill(y_prob_class[:, pred_idx])
#             mplhep.histplot(
#                 h.values(),
#                 yerr=np.sqrt(h.variances()),
#                 bins=bins,
#                 label=pred_label,
#                 flow=None,
#                 ax=ax,
#             )
#         ax.legend()
#         ax.set_xlabel(rf"Prob {true_label}")
#         ax.set_ylabel(r"\# Events")
#         plt.tight_layout()
#         if figures_path is not None:
#             plt.savefig(figures_path)
#         plt.show()


def plot_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    num_bins: int = 50,
    yscale: str = "linear",
    figures_path: str | Path | None = None,
) -> None:
    h = hist.Hist(hist.axis.Regular(num_bins, -0.01, 1.01))
    bins = h.axes[0].edges

    for true_idx, true_label in enumerate(class_names):
        y_prob_class = y_prob[y_true == true_idx]
        fig, ax = plt.subplots(figsize=(6, 4))
        for pred_idx, pred_label in enumerate(class_names):
            h.reset()
            h.fill(y_prob_class[:, pred_idx])
            print(h.values().sum())
            mplhep.histplot(
                h.values(),
                yerr=np.sqrt(h.variances()),
                bins=bins,
                label=pred_label,
                flow=None,
                ax=ax,
            )
        ax.legend()
        ax.set_xlabel(rf"Prob {true_label}")
        ax.set_ylabel(r"\# Events")
        ax.set_yscale(yscale)
        plt.tight_layout()
        if figures_path is not None:
            plt.savefig(figures_path)
        plt.show()


def get_accuracy_dict(y_true, y_pred, class_names: list[str]) -> dict[str, float]:
    acc_dict = {
        class_name: np.sum((y_true == idx) & (y_true == y_pred)) / np.sum(y_true == idx)
        for idx, class_name in enumerate(class_names)
    }
    average = np.mean(list(acc_dict.values()))
    acc_dict["average"] = average
    return acc_dict


def get_fp_fn_tp_tn(y_true: np.ndarray, y_pred: np.ndarray):
    cm = confusion_matrix(y_true, y_pred)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.values.sum() - (fp + fn + tp)
    return fp, fn, tp, tn
