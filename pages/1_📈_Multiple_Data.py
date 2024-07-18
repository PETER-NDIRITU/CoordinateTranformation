# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 17:16:23 2023

@author: HP
"""
from ANN_Transformation import m as map1
import CT_models
import json
import streamlit as st
import pandas as pd
import numpy as np
import folium
import base64
import io

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

zone_number = 37
zone_letter ='M'
#df =app1.data
#st.dataframe(df, use_container_width=True)
col1, col2 =st.columns([4,2])

with col2:
    # Main function to handle form submission
    def main():
    
        # Add an option for users to choose the transformation direction
        #transformation_direction = st.selectbox("Choose Transformation Direction:", ["Cassini to UTM", "UTM to Cassini"])
    
        col1, col2, col3 =st.columns([3,2,2])  
        with col1:
        # Allow the user to choose the transformation type
            transformation_type = st.radio("Select Transformation Type", ('UTM to Cassini', 'Cassini to UTM'))
        with col2:
           # Allow the user to choose the transformation type
            show_map= st.radio("Show on Map", ('No','Yes'))   
        with col3:
           # Allow the user to choose the transformation type
            show_data= st.radio("Show data", ('No','Yes'))          
       
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        # Use st.file_uploader for file uploading
        with st.form(key='my_form'):
            # User input for x and y
               # creating a button for Prediction

            if uploaded_file is not None:
                # Load the uploaded CSV file into a Pandas DataFrame
                input_data = pd.read_csv(uploaded_file, header=None, delimiter=',', encoding='ISO-8859-1')
                input_data2 = input_data.copy()
                input_data1 = input_data.copy()
                input_data.columns = ['E', 'N']
                input_data1.columns= ['Easting','Northing']
                input_data2.columns= ['E (X)','N (Y)']
                E = input_data['E'].values
                N = input_data['N'].values
                
                if show_data =='Yes':
                    
                    st.write("Loaded DataFrame:")
                    loaded_dataframe = st.write(input_data2)
                elif show_data =='No':
                    st.empty()

                # Button to generate matrix
                submit_button = st.form_submit_button(label='Submit Data')
        
                if submit_button:
                    show_data ='No'
                    transformed_coords = ''
                    
                    # Perform the transformation based on user's choice
                    if transformation_type == "Cassini to UTM":
                        
                        transformed_coords = CT_models.cass_to_utm_transform1(np.column_stack((E,N)))
                        if show_map == 'Yes':
                            utm_latlon = CT_models.convert_utm_to_latlon(pd.DataFrame({'Easting': transformed_coords[:, 0], 'Northing': transformed_coords[:, 1]}), zone_number, zone_letter)
                    
                            # Loop through each row in the dataframe
                            for i, row in utm_latlon.iterrows():
                                user_marker = {
                                    'location': [row['Latitude'], row['Longitude']],
                                    'popup': f'User Input: ({row["Easting"]}, {row["Northing"]})',
                                    'tooltip': 'User Input',
                                    'icon': folium.Icon(color='orange', icon='info-sign')
                                }
                                map1.add_marker(**user_marker)
                            print(input_data[0])       
                        elif show_map =='No':
                            a=3
                            #map1.get_root().clear_layers()
                            #map1.clear_markers()
                        # Code for Cassini to UTM transformation
                        # ...
            
                    elif transformation_type == "UTM to Cassini":

                        transformed_coords = CT_models.utm_to_cass_transform1(np.column_stack((E,N)))
                        if show_map == 'Yes':
                            utm_latlon = CT_models.convert_utm_to_latlon(input_data1, zone_number, zone_letter)
                            
                            # Loop through each row in the dataframe
                            for i, row in utm_latlon.iterrows():
                                user_marker = {
                                    'location': [row['Latitude'], row['Longitude']],
                                    'popup': f'User Input: ({row["Easting"]}, {row["Northing"]})',
                                    'tooltip': 'User Input',
                                    'icon': folium.Icon(color='orange', icon='info-sign')
                                }
                                map1.add_marker(**user_marker)
                            
                                             
                        elif show_map =='No':
                           # map1.get_root().clear_layers()
                          a=2
                            #map1.clear_markers()
                        # Code for UTM to Cassini transformation
                        # ...
                   #uploaded_file = None
                    #st.empty()
       
                    st.text("Transformed Coordinates:")
                    loaded_dataframe = st.write(transformed_coords)
    
        #else:
         #   st.warning("Invalid transformation direction selected.")

    if __name__ == '__main__':
        main()
with col1:
    
    map1.to_streamlit()
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