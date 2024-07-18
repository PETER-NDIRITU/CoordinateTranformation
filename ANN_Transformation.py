# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:55:49 2024

@author: HP
"""
import CT_models
import pandas as pd
import streamlit as st
import leafmap.foliumap as leafmap
import numpy as np
from pyproj import Proj, transform
from pathlib import Path
import pickle
import folium

st.set_page_config(
    page_title="📈 Data",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)
menu_options = {
    "📈 Data",
   
}

# Navigation selection
selection = st.sidebar.selectbox("Select Page:", menu_options)

# Navigation logic
if selection == "📈 Data":
    # Redirect to the Data Page
    st.experimental_set_query_params(page="data")
    
# Add a markdown section to the sidebar
st.sidebar.markdown(
    """
    # Welcome to ANN Datum Transformation Optimization Platform 🌐

    In the pursuit of achieving high accuracy in Datum transformation models, this platform introduces an innovative approach leveraging the power of Artificial Intelligence (AI). Traditional models often exhibit errors, and this platform seeks to enhance the accuracy of the Affine six-parameter 2-Dimensional coordinate transformation model through state-of-the-art Artificial Neural Network (ANN) techniques.


   """
)
st.sidebar.markdown(
    """
   
    """
    )
st.sidebar.markdown(
    """
    Accuracy improved by 98%, with atmost mm RMSE
    """
    )

data = pd.read_csv('C:/Users/HP/ml/streamlit/data/utmTrain.csv',delimiter=',', 
                   usecols=['name', 'E', 'N'], encoding='ISO-8859-1') 
data = data.dropna()
data.columns= ['name', 'Easting','Northing']



col1, col2 =st.columns([4,2])

with col2:
    

    dropdown = st.selectbox("BASEMAP",["HYBRID","TERRAIN", "ROADMAP", "SATELLITE"])
    
    m = leafmap.Map(center=[CT_models.center_lat, CT_models.center_lon], zoom=10)
    m.add_basemap(dropdown)
    for i, row in CT_models.df_latlon.iterrows():
    # Setup the content of the popup
       
        popup_content = f'Name: {row["name"]} <br> Easting (X): {row["Easting"]} <br> Northing (Y): {row["Northing"]}'
        # Add each point to the map
        m.add_marker(location=[row['Latitude'], row['Longitude']], popup=popup_content, tooltip=row['name'])

    
    
    # Main function to handle form submission
    def main():
      #  st.title('Coordinate Transformation Web App')
        col1, col2 =st.columns([4,2])  
        with col1:
        # Allow the user to choose the transformation type
            transformation_type = st.radio("Select Transformation Type", ('UTM to Cassini', 'Cassini to UTM'))
        with col2:
           # Allow the user to choose the transformation type
            show_map= st.radio("Show on Map", ('No','Yes'))   
       
           # U
        # Use st.form_submit_button for form submission
        with st.form(key='my_form'):
            # User input for x and y
            x_input = st.number_input("Enter the value for x:", value=0.0, step=0.1)
            y_input = st.number_input("Enter the value for y:", value=0.0, step=0.1)
            input_data = np.array([[x_input, y_input]])
    
            # creating a button for Prediction
            submit_button = st.form_submit_button(label='Transform Coordinates')
       #reset_button = st.button(label='Reset Form')
    
        transformed_coords = ''
   
        # Convert user inputs to latitude and longitude
        if transformation_type == 'UTM to Cassini':
            transformed_coords = CT_models.utm_to_cass_transform(input_data)
            if show_map == 'Yes':
                utm_latlon = CT_models.convert_utm_to_latlon(pd.DataFrame({'Easting': [x_input], 'Northing': [y_input]}), 37, 'M')
                user_marker = {'location': [utm_latlon['Latitude'].iloc[0], utm_latlon['Longitude'].iloc[0]],
                           'popup': f'User Input: ({x_input}, {y_input})',
                           'tooltip': 'User Input',
                           'icon': folium.Icon(color='orange', icon='info-sign')} 
                m.add_marker(**user_marker)
              
            elif show_map =='No':
              u=1
                
        elif transformation_type == 'Cassini to UTM':
            transformed_coords = CT_models.cass_to_utm_transform(input_data)
            if show_map == 'Yes':
                utm_latlon = CT_models.convert_utm_to_latlon(pd.DataFrame({'Easting': [transformed_coords[0, 0]], 'Northing': [transformed_coords[0, 1]]}), 37, 'M')
                user_marker = {'location': [utm_latlon['Latitude'].iloc[0], utm_latlon['Longitude'].iloc[0]],
                           'popup': f'User Input: ({x_input}, {y_input})',
                           'tooltip': 'User Input',
                           'icon': folium.Icon(color='orange', icon='info-sign')} 
                m.add_marker(**user_marker)
               
            elif show_map =='No':
                o=1
        if submit_button:
            st.text("Transformed Coordinates:")
            st.write(transformed_coords[0,0],transformed_coords[0,1])
             
    
            # Display the result
           # st.text("Transformed Coordinates:")
           # st.write(transformed_coords)
    
    if __name__ == '__main__':
        main()

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

        Join us on this journey to elevate the precision of coordinate transformations and explore the seamless integration of AI in the realms of surveying and mapping. 🚀
        """)