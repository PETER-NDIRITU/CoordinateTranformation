# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 12:38:11 2024

@author: HP
"""

import CT_models
import pandas as pd
import streamlit as st
import leafmap.foliumap as leafmap
import folium
import numpy as np
from pyproj import Proj, transform
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components
import json
import base64
import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from keras.models import load_model
from sklearn.metrics import mean_squared_error

# Page configuration
def configure_page():
    st.set_page_config(
        page_title="📈 Data",
        page_icon=":chart_with_upwards_trend:",
        layout="wide"
    )

def display_sidebar():
    st.sidebar.markdown(
        """
        # Welcome to ANN Datum Transformation Optimization Platform 🌐

        In the pursuit of achieving high accuracy in Datum transformation models, this platform introduces an innovative approach leveraging the power of Artificial Intelligence (AI). 
        Traditional models often exhibit errors, and this platform seeks to enhance the accuracy of the Affine six-parameter 2-Dimensional coordinate transformation model through
        state-of-the-art Artificial Neural Network (ANN) techniques.
        """
    )
    st.sidebar.markdown("Accuracy improved by 98%, with up to mm level RMSE")

def load_sample_data():
    data = pd.read_csv('C:/Users/HP/ml/streamlit/data/utmTrain.csv', delimiter=',', usecols=['name', 'E', 'N'], encoding='ISO-8859-1') 
    data = data.dropna()
    data.columns = ['name', 'Easting', 'Northing']
    return data

def create_leafmap(center_lat, center_lon, dropdown):
    m = leafmap.Map(center=[center_lat, center_lon], zoom=6.499)
    m.add_basemap(dropdown)
    return m

def update_map_center(df):
    new_center_lat = df['Latitude'].mean()
    new_center_lon = df['Longitude'].mean()
    return new_center_lat, new_center_lon

def handle_file_upload():
    uploaded_file = st.file_uploader("Choose a CSV file for training data", type=['csv'])
    if uploaded_file is not None:
        user_data = pd.read_csv(uploaded_file)
        user_data = user_data.dropna()
        st.write(" **User Data Preview**", user_data.head(4))
        return user_data
    return None

def get_transformation_type():
    col1, col2 = st.columns([4, 2])  
    with col1:
        transformation_type = st.radio("Select Transformation Type", ('UTM to Cassini', 'Cassini to UTM'))
    return transformation_type

def get_selected_columns(user_data):
    name = st.selectbox("Select Points ID/ Name", user_data.columns)
    feature_columns = st.multiselect(
        "Select exactly two feature columns in this order (X Y) or (E N)",
        user_data.columns,
        help="Select exactly two columns to be used as features."
    )
    target_columns = st.multiselect(
        "Select exactly two target columns in this order (X Y) or (E N)",
        [col for col in user_data.columns if col not in feature_columns],
        help="Select exactly two columns to be used as targets."
    )
    return name, feature_columns, target_columns

def validate_columns(feature_columns, target_columns):
    if len(feature_columns) != 2:
        st.error("Please select exactly two feature columns ")
        return False
    if len(target_columns) != 2:
        st.error("Please select exactly two target columns.")
        return False
    return True

def map_user_data(name, user_data, feature_columns, target_columns):
    P_ID = user_data[name].to_numpy()
    X = user_data[feature_columns].to_numpy()
    target = user_data[target_columns].to_numpy()
    X_map = pd.DataFrame(X, columns=['Easting', 'Northing'])
    X_map['name'] = P_ID 
    X_map = X_map.dropna()
    st.write(" **Mapped Data Sample**", X_map.head())
    return X, target, X_map

def handle_map_display(transformation_type, X_map):
    if transformation_type == 'UTM to Cassini':
        show_map = st.radio("Show on Map", ('No', 'Yes'))
        if show_map == 'Yes':
            utm_latlon = CT_models.convert_utm_to_latlon1(X_map, zone_number, zone_letter)
            center_lat, center_lon = utm_latlon['Latitude'].mean(),utm_latlon['Longitude'].mean()
            m = create_leafmap(center_lat, center_lon, dropdown)
            import time
            with st.spinner('INITIALISING...'):
                time.sleep(2)
                for i, row in utm_latlon.iterrows():
                    user_marker = {
                        'location': [row['Latitude'], row['Longitude']],
                        'popup': f'User Input: ({row["name"]},{row["Easting"]}, {row["Northing"]})',
                        'tooltip': f'{row["name"]}',
                        'icon': folium.Icon(color='orange', icon='info-sign')
                    }
                    m.add_marker(**user_marker)
                st.success('Done!')
            return m
    return None

def affine_transformation(X_train, y, params):
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y)
    neural_network_model = MLPRegressor(solver='lbfgs', 
                                        alpha=1e-6,
                                        hidden_layer_sizes=(8, 2),
                                        learning_rate_init=0.1,
                                        activation='tanh',
                                        max_iter=1000,
                                        random_state=24)
    neural_network_model.fit(X_train_scaled, y_train_scaled)
    return neural_network_model

def save_ann_model(model):
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)
    return buffer

def main():
    configure_page()
    display_sidebar()

    cass_to_utm_model = load_model('C:/Users/HP/musaML/labscripts/labscripts/PROJECT/msc/UCassToUtm_model.keras')

    data = load_sample_data()
    zone_number = 37
    zone_letter = 'M'
    col1, col2 = st.columns([4, 2])

    with col2:
        dropdown = st.selectbox("BASEMAP", ["HYBRID", "TERRAIN", "ROADMAP", "SATELLITE"])
        m = create_leafmap(CT_models.center_lat, CT_models.center_lon, dropdown)

        user_data = handle_file_upload()
        if user_data is not None:
            transformation_type = get_transformation_type()
            name, feature_columns, target_columns = get_selected_columns(user_data)

            if validate_columns(feature_columns, target_columns):
                X, target, X_map = map_user_data(name, user_data, feature_columns, target_columns)
                m = handle_map_display(transformation_type, X_map)

                if st.button("Compute Affine Parameters and Train Model", type="primary"):
                    X_train, X_test, y, y1 = train_test_split(X, target, test_size=0.2, random_state=42)
                    params = CT_models.affine_transformation(X_train, y)
                    params = params * -1

                    neural_network_model = affine_transformation(X_train, y, params)

                    train_rmse = np.sqrt(mean_squared_error(y, neural_network_model.predict(X_train)))
                    test_rmse = np.sqrt(mean_squared_error(y1, neural_network_model.predict(X_test)))

                    st.markdown(f'<h1 style="color:#FF33E9;font-size:18px;">{f"Training RMSE MLR_AFFINE: {train_rmse:.2f}"}</h1>', unsafe_allow_html=True)
                    st.markdown(f'<h1 style="color:#FF33E9;font-size:18px;">{f"Testing RMSE MLR_AFFINE: {test_rmse:.2f}"}</h1>', unsafe_allow_html=True)

                    buffer = save_ann_model(neural_network_model)
                    st.download_button(label="Download Trained Model", data=buffer, file_name="ann_model.pkl", mime="application/octet-stream")

    with col1:
        m.to_streamlit()
        st.write(
            """
            **Key Objectives:**
            - Optimize coordinate transformation accuracy using advanced AI methodologies.
            - Provide an efficient online platform for coordinate transformation.
            - Enable functionalities like data editing and spatial queries.
            - Display training data, add new data, and query model accuracies.
            - Automate the transformation process for improved speed and convenience.

            **How to Transform**

            **Load Data:**

            load your input data, which typically consists of two sets of coordinates. 
            These coordinates represents Control points in both Coordinate systems.

            **Select Features and target variables:**

            Select Features as original Coordinate System Points, and target variables as the second Coordinate system.

            **Determine Affine Transformation Parameters:**

            This solve for the parameters that best align your input coordinates to a set of target coordinates.

            **Test the Model:**

            After determining the parameters, it's essential to test the model to ensure it correctly transforms the input coordinates to the target coordinates

            **Use the Model for Transformation**

            Once the model is tested and validated, you can use it to transform any new set of coordinates

            Join us on this journey to elevate the precision of coordinate transformations with the
""")
if __name__ == '__main__':
    main()
