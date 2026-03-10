"""
Streamlit App
Serve the trained Logistic Regression model via a web UI.
Run with: `streamlit run app_streamlit.py`
"""
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from pre_processing import FeatureEngineer 

# Load preprocessor and model
scaler = joblib.load(Path(__file__).parent / "artifacts/preprocessor.pkl")
model  = joblib.load(Path(__file__).parent / "artifacts/model.pkl")

def main():
    st.title("ASG 04 MD - Dextra Faren - Spaceship Titanic Model Deployment")
    st.write("Enter the 13 passenger features below to predict transportation status.")

    col1, col2 = st.columns(2)

    with col1:
        PassengerId = st.text_input("PassengerId", value="0001_01")
        HomePlanet = st.selectbox("HomePlanet", options=["Europa", "Earth", "Mars"])
        CryoSleep = st.selectbox("CryoSleep", options=[False, True])
        Cabin = st.text_input("Cabin", value="B/0/P")
        Destination = st.selectbox("Destination", options=["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"])
        Age = st.number_input("Age", min_value=0.0, max_value=120.0, value=25.0, step=1.0)
        VIP = st.selectbox("VIP", options=[False, True])

    with col2:
        RoomService = st.number_input("RoomService", min_value=0.0, value=0.0, step=10.0)
        FoodCourt = st.number_input("FoodCourt", min_value=0.0, value=0.0, step=10.0)
        ShoppingMall = st.number_input("ShoppingMall", min_value=0.0, value=0.0, step=10.0)
        Spa = st.number_input("Spa", min_value=0.0, value=0.0, step=10.0)
        VRDeck = st.number_input("VRDeck", min_value=0.0, value=0.0, step=10.0)
        Name = st.text_input("Name", value="Dextra Faren")

    if st.button("Predict"):
        features = pd.DataFrame([{
            'PassengerId': PassengerId, 'HomePlanet': HomePlanet, 'CryoSleep': CryoSleep,
            'Cabin': Cabin, 'Destination': Destination, 'Age': Age, 'VIP': VIP,
            'RoomService': RoomService, 'FoodCourt': FoodCourt, 'ShoppingMall': ShoppingMall,
            'Spa': Spa, 'VRDeck': VRDeck, 'Name': Name
        }])
        
        try:
            processed_data = scaler.transform(features)
            prediction = model.predict(processed_data)[0]
            
            st.divider()
            if prediction == 1:
                st.success("### Prediction: Transported")
            else:
                st.info("### Prediction: Not Transported")
                
        except Exception as e:
            st.error(f"Error during prediction mapping: {e}")

if __name__ == "__main__":
    main()