"""
Step 4: Evaluation
Evaluates the optimal Logistic Regression model on the test split.
"""
import joblib
from sklearn.metrics import accuracy_score, classification_report

def evaluate(test_scaled):
    model = joblib.load("artifacts/model.pkl")
    
    X_test = test_scaled.drop("Transported", axis=1)
    y_test = test_scaled["Transported"]
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"\nEvaluation | Accuracy = {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    return acc

if __name__ == "__main__":
    evaluate(None)