import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import warnings
from pathlib import Path

# =========================
# Fake Confusion Matrix Data
# =========================
# Ma trận theo đúng ảnh:
#        True
#       Drone | background FP
# Pred ----------------------
# Drone      0.8 | 1.0
# bg FN      0.2 | 0.0

matrix = np.array([
    [0.8, 1.0],
    [0.2, 0.0]
])

# Class names
names = ['Drone']

# =========================
# Plot Function
# =========================
def plot_confusion_matrix(matrix, normalize=False, save_dir=''):
    try:
        # Chuẩn hóa nếu cần
        array = matrix / ((matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)

        # Ẩn giá trị quá nhỏ
        array[array < 0.005] = np.nan

        # Figure
        fig = plt.figure(figsize=(12, 9), tight_layout=True)

        nc = len(names)

        # Scale font
        sn.set(font_scale=1.0 if nc < 50 else 0.8)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            ax = sn.heatmap(
                array,
                annot=True,
                annot_kws={"size": 8},
                cmap='Blues',
                fmt='.2f',
                square=True,
                vmin=0.0,
                vmax=1.0,
                xticklabels=names + ['background FP'],
                yticklabels=names + ['background FN']
            )

            ax.set_facecolor((1, 1, 1))

        # Axis labels
        fig.axes[0].set_xlabel('True')
        fig.axes[0].set_ylabel('Predicted')

        # Save image
        save_path = Path(save_dir) / 'confusion_matrix.png'
        fig.savefig(save_path, dpi=300)

        plt.show()
        plt.close()

        print(f"Saved to: {save_path}")

    except Exception as e:
        print(f'WARNING: ConfusionMatrix plot failure: {e}')

# =========================
# Run
# =========================
plot_confusion_matrix(matrix)