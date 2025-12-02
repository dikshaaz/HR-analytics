# ModelTrainingAndEvaluation.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def train_and_evaluate(X, y):
    """
    Train Logistic Regression and Random Forest models.
    Evaluate models using Accuracy, Precision, Recall, F1, Confusion Matrix, ROC Curve.
    Save best model.
    """
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier()
    }

    metrics_dict = {}
    best_model_name = None
    best_score = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = auc(fpr, tpr)

        # Store metrics
        metrics_dict[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "AUC": auc_score
        }

        print(f"\n=== {name} ===")
        print(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc_score:.3f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        # ROC Curve
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.2f})")
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend()
        plt.show()

        # Save best model by F1 score
        if f1 > best_score:
            best_score = f1
            best_model_name = name
            best_model_object = model

    # Save best model
    joblib.dump(best_model_object, "best_attrition_model.pkl")
    print(f"\n✔ Best model '{best_model_name}' saved as best_attrition_model.pkl")

    return best_model_object, metrics_dict, metrics_dict