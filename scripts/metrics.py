import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scripts.config import number_to_label


class ModelEvaluator:
    def __init__(self, y_true, y_pred):
        """
        Initializes the evaluator with ground truth and predictions.
        """
        self.y_true = y_true.cpu().numpy()
        self.y_pred = y_pred.cpu().numpy()
        self.labels = sorted(set(self.y_true) | set(self.y_pred))  # Ensure known order of labels
        self.cm = confusion_matrix(self.y_true, self.y_pred)

    def plot_confusion_matrix(self):
        """
        Plots the confusion matrix with labels.
        """
        gestures = [number_to_label(label) for label in self.labels]
        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(self.cm, annot=True, fmt="d", cmap="Greys", 
                         xticklabels=gestures, yticklabels=gestures)

        # Move x-axis labels to the top
        ax.xaxis.set_label_position('top')  
        ax.xaxis.tick_top()

        plt.xlabel("Predicted Gesture")
        plt.ylabel("True Gesture")
        plt.title("Confusion Matrix")
        plt.yticks(rotation=45)
        plt.show()

    def print_classification_report(self):
        """
        Prints the classification report with precision, recall, and F1-score.
        """
        report = classification_report(self.y_true, self.y_pred, labels=self.labels)
        print("Classification Report:\n", report)

    def print_accuracy(self):
        """
        Prints the accuracy score.
        """
        acc = accuracy_score(self.y_true, self.y_pred)
        print(f"Accuracy: {acc:.2f}")

    def evaluate(self):
        """
        Runs all evaluation functions.
        """
        self.plot_confusion_matrix()
        self.print_classification_report()
        self.print_accuracy()


# evaluator = ModelEvaluator(y_true, y_pred)
# evaluator.evaluate()
