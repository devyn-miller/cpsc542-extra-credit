from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def print_classification_report(y_true, y_pred):
    """
    Print the classification report for the true and predicted labels.

    This function utilizes scikit-learn's classification_report to generate and print a text report showing
    the main classification metrics (precision, recall, f1-score, support).

    Parameters:
    y_true (array-like): True labels of the data.
    y_pred (array-like): Predicted labels by the model.

    Returns:
    None: This function only prints the classification report.
    """
    print(classification_report(y_true, y_pred))

def plot_confusion_matrix(y_true, y_pred):
    """
    Plot a confusion matrix using seaborn's heatmap.

    This function calculates the confusion matrix using scikit-learn's confusion_matrix and then plots it using
    seaborn's heatmap function. The plot is displayed with 'Fire' and 'Not Fire' as labels for both axes.

    Parameters:
    y_true (array-like): True labels of the data.
    y_pred (array-like): Predicted labels by the model.

    Returns:
    None: This function only displays the plot and does not return any value.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm,
                annot=True, fmt='d',
                xticklabels=['Fire', 'Not Fire'],
                yticklabels=['Fire', 'Not Fire'],
                cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
