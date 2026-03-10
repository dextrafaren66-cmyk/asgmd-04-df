"""
Session 04 – Step 2: Preprocessing
Reads ingested data, splits, scales, and saves the preprocessor artifact.
"""
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        df = X.copy()
        
        df['Deck'] = df['Cabin'].apply(lambda x: str(x).split('/')[0] if pd.notna(x) else 'Unknown')
        df['Cabin_num'] = df['Cabin'].apply(lambda x: str(x).split('/')[1] if pd.notna(x) and len(str(x).split('/'))>1 else -1).astype(float)
        df['Side'] = df['Cabin'].apply(lambda x: str(x).split('/')[2] if pd.notna(x) and len(str(x).split('/'))>2 else 'Unknown')
        
        df['Group'] = df['PassengerId'].apply(lambda x: str(x).split('_')[0] if pd.notna(x) else '0000')
        df['Group_size'] = df.groupby('Group')['Group'].transform('count')
        df['Solo'] = (df['Group_size'] == 1).astype(int)
        
        df['FirstName'] = df['Name'].apply(lambda x: str(x).split()[0] if pd.notna(x) else 'Unknown')
        df['LastName'] = df['Name'].apply(lambda x: str(x).split()[-1] if pd.notna(x) else 'Unknown')
        df['Family_size'] = df.groupby('LastName')['LastName'].transform('count')
        
        # Spending Feauture
        spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        for col in spending_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        df['TotalSpending'] = df[spending_cols].sum(axis=1)
        df['HasSpending'] = (df['TotalSpending'] > 0).astype(int)
        df['NoSpending'] = (df['TotalSpending'] == 0).astype(int)
        
        # Age grouping
        df['Age_group'] = pd.cut(df['Age'], bins=[0, 12, 18, 30, 50, 100], labels=['Child', 'Teen', 'Young_Adult', 'Adult', 'Senior'])
        
        # Missing indicators?
        df['Age_missing'] = df['Age'].isna().astype(int)
        df['CryoSleep_missing'] = df['CryoSleep'].isna().astype(int)
        
        # Force categoricals to string 
        cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Age_group']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(object).fillna('Unknown').astype(str)
                
        # Drop raw text data
        df.drop(['PassengerId', 'Cabin', 'Name', 'Group', 'FirstName', 'LastName'], axis=1, inplace=True, errors='ignore')
        return df

def preprocess():
    os.makedirs("artifacts", exist_ok=True)
    df = pd.read_csv("ingested/train.csv")

    X = df.drop("Transported", axis=1)
    y = df["Transported"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Age_group']
    numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                          'Cabin_num', 'Group_size', 'Solo', 'Family_size', 'TotalSpending',
                          'HasSpending', 'NoSpending', 'Age_missing', 'CryoSleep_missing']

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numerical_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), categorical_features)
    ])

    full_pipeline = Pipeline(steps=[('feature_engineer', FeatureEngineer()), ('preprocessor', preprocessor)])

    X_train_processed = full_pipeline.fit_transform(X_train)
    X_test_processed = full_pipeline.transform(X_test)

    joblib.dump(full_pipeline, "artifacts/preprocessor.pkl")

    all_features = numerical_features + categorical_features
    train_scaled = pd.DataFrame(X_train_processed, columns=all_features)
    train_scaled['Transported'] = y_train.values
    test_scaled = pd.DataFrame(X_test_processed, columns=all_features)
    test_scaled['Transported'] = y_test.values

    print("Preprocessing done. Preprocessor saved to artifacts/preprocessor.pkl")
    return train_scaled, test_scaled

if __name__ == "__main__":
    preprocess()