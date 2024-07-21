import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def calculate_metrics(y_true, y_pred, classes):

    TP = {cls: 0 for cls in classes}
    TN = {cls: 0 for cls in classes}
    FP = {cls: 0 for cls in classes}
    FN = {cls: 0 for cls in classes}

    for cls in classes:
        TP[cls] = np.sum((y_true == cls) & (y_pred == cls))
        TN[cls] = np.sum((y_true != cls) & (y_pred != cls))
        FP[cls] = np.sum((y_true != cls) & (y_pred == cls))
        FN[cls] = np.sum((y_true == cls) & (y_pred != cls))


    recall = {cls: TP[cls] / (TP[cls] + FN[cls]) if (TP[cls] + FN[cls]) != 0 else 0 for cls in classes}
    precision = {cls: TP[cls] / (TP[cls] + FP[cls]) if (TP[cls] + FP[cls]) != 0 else 0 for cls in classes}
    f1_score = {cls: 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls]) if (precision[cls] + recall[cls]) != 0 else 0 for cls in classes}
    rand_index = {cls: (TP[cls] + TN[cls]) / (TP[cls] + TN[cls] + FP[cls] + FN[cls]) if (TP[cls] + TN[cls] + FP[cls] + FN[cls]) != 0 else 0 for cls in classes}

    avg_recall = np.mean(list(recall.values()))
    avg_precision = np.mean(list(precision.values()))
    avg_f1_score = np.mean(list(f1_score.values()))
    avg_rand_index = np.mean(list(rand_index.values()))

    return avg_recall, avg_precision, avg_f1_score, avg_rand_index, recall, precision, f1_score, rand_index

classes = [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140]

# ΑΛΛΑΓΗ ΤΩΝ ΑΡΧΕΙΩΝ ΑΝΑΛΟΓΑ ΜΕ ΤΟ ΜΟΝΤΕΛΟ
y_test = pd.read_csv('y_test_neural.csv')
y_predictions = pd.read_csv('y_predictions_neural.csv')

y_test = y_test.iloc[:, 0].to_numpy()
y_predictions = y_predictions.iloc[:, 0].to_numpy()


avg_recall, avg_precision, avg_f1_score, avg_rand_index, recall, precision, f1_scores, rand_index = calculate_metrics(y_test, y_predictions, classes)

print("Average Recall:", avg_recall)
print("Average Precision:", avg_precision)
print("Average F1-Score:", avg_f1_score)
print("Average Rand Index:", avg_rand_index)
print("Recall per class: ", recall)
print("Precision per class:", precision)
print("F1-Score per class:", f1_scores)
print("Rand Index per class:", rand_index)


# CONFUSION MATRIX
conf_matrix = confusion_matrix(y_test, y_predictions, labels=classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# BARPLOT AVERAGE
metrics_df = pd.DataFrame({
    'Metric': ['Average Precision', 'Average Recall', 'Average F1-Score', 'Average Rand Index'],
    'Score': [avg_precision, avg_recall, avg_f1_score, avg_rand_index]
})

sns.set_palette(sns.color_palette("Blues"))
plt.figure(figsize=(8, 5))
sns.barplot(x='Metric', y='Score', data=metrics_df)
plt.xlabel('Metric')
plt.ylabel('Score')
plt.title('Overall Metrics')
plt.ylim(0, 1)
plt.show()


# BARPLOT FOR EVERY LABEL
num_classes = len(classes)
num_rows = num_classes // 3
if num_classes % 3 != 0:
    num_rows += 1
fig, axes = plt.subplots(num_rows, 3, figsize=(12, 3*num_rows), sharex=True)
axes = axes.flatten()

for i, class_label in enumerate(classes):
    class_precision = precision[class_label]
    class_recall = recall[class_label]
    class_f1_score = f1_scores[class_label]
    class_rand_index = rand_index[class_label]

    sns.barplot(x='Metric', y='Score', data=metrics_df, ax=axes[i])
    axes[i].set_title(f'Metrics for Class {class_label}')
    axes[i].set_ylim(0.0, 1.0)
    axes[i].set_ylabel('Score')

for i in range(num_classes, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()
