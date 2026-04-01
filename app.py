import streamlit as st
import pandas as pd 
import numpy as np
import datetime
import joblib
import time
from geopy.geocoders import Nominatim # Used to get latitude and longitude of the property entered by the user 
from geopy.exc import GeocoderTimedOut
from src.visualizations.plots import plot_feature_importance,plot_scatter # Self created file that contains custom plots features 
from src.data_manipulation.feature_enginnering import engineer_features # Self created file that performs feature engineering on user input  

st.markdown("""
    <style>
        .block-container {
            max-width: 900px;
            margin: auto;
            padding-top: 2rem;
            padding-bottom: 2   rem;
        }
    </style>
""", unsafe_allow_html=True)


space = "&nbsp;"

@st.cache_data
def load_rf():
    data = joblib.load("saved_models/random_forest_results.pkl")
    return data

@st.cache_data
def load_ridge():
    data = joblib.load("saved_models/ridge_results.pkl")
    return data

with st.spinner("🚀 Loading data and models..."):
    ridge = load_ridge()
    rf = load_rf()

st.sidebar.markdown(""" # 🧭 Navigation""")
page = st.sidebar.radio(
    "Choose",
    ["About the project","🧪 Test models", "📊 Analyse models"]
)

if page == "About the project":

    st.header("About the Project")
    st.subheader("Predict housing prices in Stockholm using machine learning")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        #### 🧪 Test Models  
        Input property details and get predicted prices.

        **Models:**  
        - Random Forest  
        - Ridge Regression  

        ⚠️ Predictions are generated using **dummy models trained on synthetic data**.  
        No real company data is used in this section.
        """)

    with col2:
        st.markdown("""
        #### 📊 Analyse Models  
        Explore performance of models trained on **real historical data**.

        **Includes:**  
        - MAPE, RMSE, MAE, R²  
        - y_test vs y_pred plots  
        - Feature importance  

        **Purpose:**  
        Compare models and understand housing price patterns.

        ⚠️ Analysis is based on historical data for educational use only.  
        Predictions in *Test Models* remain synthetic.
        """)

elif page == "🧪 Test models":
    
    # Bounding box for Stockholm region
    MIN_LAT, MAX_LAT = 59.2272, 59.4402
    MIN_LON, MAX_LON = 17.7605, 18.2011

    @st.cache_data
    def get_coordinates(address):
        geolocator = Nominatim(user_agent="my_house_app_1234", timeout=10)

        try:
            location = geolocator.geocode(address)
            if location:
                return location.latitude, location.longitude
        except GeocoderTimedOut:
            return None, None

        return None, None

    def is_in_stockholm(lat,long):
        if ( MAX_LAT >= lat >= MIN_LAT ) and (MAX_LON >= long >= MIN_LON):
            return True
        else:
            return False

    st.title(f"{space*12}House prediction in Stockholm 🏠")

    colA, colB = st.columns(2)

    with colA:
        st_Contract_Date = None
        st_Contract_Date = st.date_input("Sale date",
                                        min_value="today",
                                        max_value=datetime.date.today() + datetime.timedelta(days=90),
                                        value=None)

        st_lat = None
        st_long = None

        st.write("")
        user_address = st.text_input("Enter your address")
        if user_address: 
            if len(user_address.strip()) > 5:
                time.sleep(2)
                st_lat, st_long = get_coordinates(user_address)
                if st_lat and st_long:
                    if not is_in_stockholm(st_lat,st_long):
                        st.error("❌ The address is OUTSIDE the Stockholm region.\n"
                                f"Allowed coordinates:\n"
                                f"- Latitude: {MIN_LAT} – {MAX_LAT}\n"
                                f"- Longitude: {MIN_LON} – {MAX_LON}")
                    st.markdown(
                        f"📍 Longitude: `{st_long:.4f}`  |  Latitude: `{st_lat:.4f}`"
                        )
                else:   
                    st.error("Could not find this address.")
            else:
                st.error("Please enter your full address to get coordinates.")
                
        st_Housing_Type = st.radio("House type",["Apartment","House","Other"],horizontal=True)


        sub_col2_1, sub_col2_2, sub_col2_3 =  st.columns(3)

        with sub_col2_1:
            st_Balcony = st.radio("Balcony",["Yes","No"],) 

        with sub_col2_2:
            st_Elevator = st.radio("Elevator", ["Yes","No"])

        with sub_col2_3:
            st_Heating_Included = st.radio("Heating included",["Yes","No"])

    with colB: 
        st_Living_Area_sqm = st.number_input("Living area",
                                      min_value=1,
                                      max_value=400,
                                      value=None,
                                      help="m²")
        
        st_Year_Built = st.select_slider("Year bulit",range(1850,2027))
        
        sub_col1, sub_col2 = st.columns(2)

        with sub_col1: 
            st_Floor =st.selectbox("Floor number",
                         range(0,26),
                         help="The specific floor where the apartment is located. Use 0 for ground floor")
        with sub_col2: 
            st_Total_Floors = st.selectbox("Total floors",
                         range(st_Floor,36),
                         help="The total number of floors in the building")

        st_Monthly_Fee = st.number_input("Monthly fee",min_value=0,max_value=25000,value=None)

        st_Annual_Fee_per_sqm = None
        if st_Living_Area_sqm and st_Monthly_Fee:
            st_Annual_Fee_per_sqm = int(round((st_Monthly_Fee * 12) / st_Living_Area_sqm , 0))


        sub_col3_1 , sub_col3_2 = st.columns([1,2])

        with sub_col3_1:
            st_New_Construction = st.radio("New Construction", ["Yes","No"],index = 1)

        with sub_col3_2:
            st_Rooms = st.select_slider("Number of rooms",
                                range(1,11),
                                value=1)
            
    col1, col2,col3 = st.columns([1,2,1])

    data_to_predict = pd.DataFrame({
    "Contract Date": [st_Contract_Date],
    "Living Area (sqm)": [st_Living_Area_sqm],
    "Rooms": [st_Rooms],
    "Floor": [st_Floor],
    "Total Floors": [st_Total_Floors],
    "Monthly Fee": [st_Monthly_Fee],
    "Annual Fee per sqm": [st_Annual_Fee_per_sqm],
    "Housing Type": [st_Housing_Type],
    "Elevator": [st_Elevator],
    "Balcony": [st_Balcony],
    "Heating Included": [st_Heating_Included],
    "Year Built": [st_Year_Built],
    "New Construction": [st_New_Construction],
    'long':[st_long],
    'lat':[st_lat]
    })


    with col2:
        st.write("")
        if st.button(
            "Predict Price (Ridge) vs (Random Forest) ",
            type="primary",
            use_container_width=True
        ):

            if not data_to_predict.isnull().values.any():

                engineer_features2 = engineer_features(data_to_predict)


                LR_pipeline = ridge["model"]

                prediction_LR = np.exp(LR_pipeline.predict(engineer_features2))

                prediction_ridge = int(prediction_LR[0])
                
                st.success(f"{space*18}📈 Ridge Regression: {prediction_ridge:,} SEK")

                

                RF_pipeline = rf["model"]

                prediction_RF = np.exp(RF_pipeline.predict(engineer_features2))

                prediction_rrf = int(prediction_RF[0])

                st.success(f"{space*18}🌲 Random Forest: {prediction_rrf:,} SEK")


                df_to_download = data_to_predict.copy()
                df_to_download["Ridge Regression"] = prediction_ridge
                df_to_download["Random Forest"] = prediction_rrf

                csv = df_to_download.to_csv(index=False)

                st.download_button(
                    label="Download predictions",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv'
                    )

            else:
                st.warning(f"{space*30}All attributes must be filled.")



elif page == "📊 Analyse models":
    st.title(f"{space*8}Ridge Regression vs Random Forest")

    col1, col2 = st.columns(2,border=True)
    
    with col1:

        col1_1, col2_1 = st.columns(2)
        
        with col1_1:
            st.metric(  
                "Mean Absolute Percentage Error (MAPE)",
                f"{ridge['metrics']['MAPE'].loc[0] * 100 :,.1f} %",
                help="Shows the average prediction error as a percentage of actual values. Lower is better."
            )
        with col2_1:
            st.metric("R² (Explained Variance)",
                      f"{ridge['metrics']['R²'].loc[0] * 100 :,.1f} %",
                      help="How much of the price changes the model can predict. Higher is better.")
            
        col3_1,col4_1 = st.columns(2)

        with col3_1 :
            st.metric("Average error",
                      f"{ridge['metrics']['MAE'].loc[0] / 1_000:,.0f}k kr".replace(","," "),
                      help="On average, how much the model's predictions are off from the actual values. Lower is better.")
        
        with col4_1:
            st.metric(
                "Root Mean Squared Error (RMSE)",
                f"{ridge['metrics']['RMSE'].loc[0] / 1_000_000:,.1f}m kr".replace(","," "),
                help="Shows how far predictions typically are from actual values, with bigger mistakes weighted more heavily. Lower is better.")
        
        
        fig2 = plot_scatter(ridge["y_test"],ridge["y_pred"])
        st.pyplot(fig2)


        fig = plot_feature_importance(ridge["feature_importance"].sort_values(),title="Linear Regression Feature Importance")
        st.pyplot(fig)
    
    with col2:

        col1_1_1, col2_1_2 = st.columns(2)
        
        with col1_1_1:
            st.metric(
                "Mean Absolute Percentage Error (MAPE)",
                f"{rf['metrics']['MAPE'].iloc[0] * 100 :,.1f} %",
                help="Shows the average prediction error as a percentage of actual values. Lower is better."
            )
        with col2_1_2:
            st.metric("R² (Explained Variance)",
                      f"{rf['metrics']['R²'].loc[0] * 100 :,.1f} %",
                      help="How much of the price changes the model can predict. Higher is better.")
        

        
        col3_1_3,col4_1_4 = st.columns(2)
        
        with col3_1_3 :
            st.metric("Average error",
                      f"{rf['metrics']['MAE'].loc[0] / 1_000:,.0f}k kr".replace(","," "),
                      help="On average, how much the model's predictions are off from the actual values. Lower is better.")
        with col4_1_4:
            st.metric(
                "Root Mean Squared Error (RMSE)",
                f"{rf['metrics']['RMSE'].loc[0] / 1_000:,.0f}k kr".replace(","," "),
                help="Shows how far predictions typically are from actual values, with bigger mistakes weighted more heavily. Lower is better.")
        
    

        fig2 = plot_scatter(rf["y_test"],rf["y_pred"])
        st.pyplot(fig2)


        fig = plot_feature_importance(rf["feature_importance"].sort_values(),title="Random Forest Feature Importance")
        st.pyplot(fig)
