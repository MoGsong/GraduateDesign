
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def plot_confusion_maxtirx(y_true,y_pred,kind):
    '''
    此函数用于绘制混淆矩阵
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param kind: 类别
            eg. kind = ["right", "foot"]
    :return:热力图
    '''
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    confusion_mat = confusion_matrix(y_true, y_pred)
    # 将矩阵转化为 DataFrame
    confusion_mat = pd.DataFrame(confusion_mat, index=kind, columns=kind)
    # 绘制 heatmap
    conf_fig = sn.heatmap(confusion_mat, annot=True, fmt="d", cmap="BuPu")
    # plt.show(block=False)


def plot_confusion_matrix(
    targets, predictions, target_names, title="Confusion matrix", cmap="Blues", normalize=True
):
    """Plot Confusion Matrix."""
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError("Install seaborn to plot confusion matrix")

    cm = confusion_matrix(targets, predictions)
    if normalize:
        # Check for zero values in each row
        row_sums = cm.sum(axis=1)
        cm[row_sums == 0, :] = 0
        row_sums[row_sums == 0] = 1

        # Normalize each row by its sum
        cm = 100 * cm.astype("float") / row_sums[:, np.newaxis]
    # cm = 100 * cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    df = pd.DataFrame(data=cm, columns=target_names, index=target_names)
    g = sns.heatmap(
        df, annot=True, fmt=".1f", linewidths=0.5, vmin=0, vmax=100, cmap=cmap
    )
    g.set_title(title)
    g.set_ylabel("True label")
    g.set_xlabel("Predicted label")
    return g

# def plot_confusion_matrix(cm, classes,
#                           normalize=True,
#                           title="Confusion matrix",
#                           cmap=plt.cm.Blues):
#     if normalize:
#         # Check for zero values in each row
#         row_sums = cm.sum(axis=1)
#         cm[row_sums == 0, :] = 0
#         row_sums[row_sums == 0] = 1
#
#         # Normalize each row by its sum
#         cm = 100 * cm.astype("float") / row_sums[:, np.newaxis]
#
#     plt.imshow(cm, interpolation="nearest", cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt = ".2f" if normalize else "d"
#     thresh = cm.max() / 2.0
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel("True label")
#     plt.xlabel("Predicted label")