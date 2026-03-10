"""
Step 3: Training
Optimizes and trains a Logistic Regression classifier.
"""
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def train(train_scaled):
    print("Starting hyperparameter optimization for Logistic Regression...")
    X_train = train_scaled.drop("Transported", axis=1)
    y_train = train_scaled["Transported"]

    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear']
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(max_iter=2000, random_state=42),
        param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    joblib.dump(best_model, "artifacts/model.pkl")
    print(f"Model trained with params: {grid_search.best_params_}")
    print("Model saved to artifacts/model.pkl")
    return best_model

if __name__ == "__main__":
    train(None)