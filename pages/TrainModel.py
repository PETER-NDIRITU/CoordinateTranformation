import CT_models
import strings
import pandas as pd
import streamlit as st
import leafmap.foliumap as leafmap
import folium
import numpy as np
from pyproj import Proj, transform
from pathlib import Path
import pickle
import folium
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components
import json
import numpy as np
import base64
import io, time
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import tensorflow as tf
from keras.utils import get_custom_objects
from keras import regularizers
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import mean_squared_error

# Page configuration
st.set_page_config(
    page_title="📈 Data",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

st.sidebar.markdown(strings.MD)
st.sidebar.markdown("Accuracy improved by 98%, with upto mm level RMSE")

#DEFINING VARIABLES
X, target, X_map, X_test = None, None, None, None
errorsT = pd.DataFrame()
errors = pd.DataFrame()
params = None
zone_number, zone_letter =None, None

col1, col2 = st.columns([3, 2])

with col2:
    dropdown = st.selectbox("BASEMAP",["HYBRID","TERRAIN", "ROADMAP", "SATELLITE"])
    m = leafmap.Map(center=[CT_models.center_lat, CT_models.center_lon], zoom=6.499)
    m.add_basemap(dropdown)
# Main function to handle form submission
    def main(): 
        st.write("## Train Your Own Model")
        uploaded_file = st.file_uploader("Choose a CSV file for training data", type=['csv'])   
        
        if uploaded_file is not None:
            user_data = pd.read_csv(uploaded_file)
            user_data = user_data.dropna()
            # Initialize session state if it doesn't already exist
            if 'show_text' not in st.session_state:
                st.session_state.show_text = False

            # Function to toggle the visibility of the text
            def toggle_text():
                st.session_state.show_text = not st.session_state.show_text

            # Display the button to toggle the text
            st.button(" **User Data Preview**", on_click=toggle_text)

            # Conditionally display the text based on session state
            if st.session_state.show_text:
                st.write( user_data.head(4))
            st.markdown("""
                    <style>
                    .streamlit-container {
                        border: 2px solid #045;
                        padding: 10px;
                    }
                    </style>
                    """, unsafe_allow_html=True)
    
            col1, col2, col3,col4 = st.columns([1.9,1.5,1.5,1.5])  
            with col1:
                transformation_type = st.radio("**Transformation Type**", ('UTM to Cassini', 'Cassini to UTM'))
            with col2:
                zone_vals = [36,37]
                default_ix = zone_vals.index(37)
                zone_number= st.selectbox("**UTM Zone**",
                                zone_vals, index = default_ix )
            with col3:
                letter_vals = ['M','N', 'O']
                default_id = letter_vals.index('M')
                zone_letter = st.selectbox("**Zone letter**",
                                letter_vals, index = default_id)
            with col4:
                show_map= st.radio("Show on Map", ('No','Yes'))  
                            
            name= st.selectbox("Select Points ID/ Name",
                                user_data.columns)
                        
            # Multi-select for feature columns
            feature_columns = st.multiselect(
                "Select exactly two feature columns in this order (X Y) or (E N)",
                user_data.columns,
                help="Select exactly two columns to be used as features."
            )
        
            # Ensure exactly two feature columns are selected
            if len(feature_columns) != 2:
                st.error("Please select exactly two feature columns ")
            else:
                # Multi-select for target columns
                target_columns = st.multiselect(
                    "Select exactly two target columns in this order (X Y) or (E N)",
                    [col for col in user_data.columns if col not in feature_columns],
                    help="Select exactly two columns to be used as targets."
                )
        
                # Ensure exactly two target columns are selected
                if len(target_columns) != 2:
                    st.error("Please select exactly two target columns.")
                else:    
                    P_ID = user_data[name].to_numpy()
                    X = user_data[feature_columns].to_numpy()
                    #X['name'] = P_ID 
                    target = user_data[target_columns].to_numpy()
                    
                   # X_name = X.copy()
                  #  X_name['name'] = Xname['P_ID']
                    X_map = pd.DataFrame(X, columns=['Easting','Northing'])
                    X_map['name'] = P_ID 
                            
                    X_map = X_map.dropna()
                    
                    all_data = pd.DataFrame(X, columns=['X', 'Y'])
                    all_targets = pd.DataFrame(target,columns= ['E','N'])

                    def toggle_text():
                        st.session_state.show_text = not st.session_state.show_text

                    # Display the button to toggle the text
                    st.button(" **Mapped Data Sample** ", on_click=toggle_text)

                    # Conditionally display the text based on session state
                    if st.session_state.show_text:
                        st.write(X_map.head())
                  
                 #   E = X_map['E'].values
                  #  N = X_map['N'].values
                  # Allow the user to choose the transformation type
                if transformation_type == 'UTM to Cassini':
                    
                    if show_map == 'Yes':
                        utm_latlon = CT_models.convert_utm_to_latlon1(X_map, zone_number, zone_letter)
   
                        with st.spinner('INITIALISING...'):
                            time.sleep(2)
                            CT_models.add_markers_to_map(utm_latlon, m)
                            st.success('Done!')
                     
                    elif show_map =='No':
                       # map1.get_root().clear_layers()
                      a=2
                    if 'show_text2' not in st.session_state:
                        st.session_state.show_text2 = False
    
                # Function to toggle the visibility of the text
                    def toggle_text():
                        st.session_state.show_text2 = not st.session_state.show_text2
    
                # Display the button to toggle the text
                        
                    st.button(" **Compute Affine Parameters and Train Model** ", on_click=toggle_text)
    
                # Conditionally display the text based on session state
                    if st.session_state.show_text2:
                      
      
                   # if st.button("Compute Affine Parameters and Train Model", type="primary"):
                        X_train, X_test, y, y1 = CT_models.split_data(X, target)
                        
                        # create a matrix
                        df_train, df_target, df_Xtest, E, N = CT_models.create_dataframes(X_train, y, X_test)
                        inputs = CT_models.initialize_matrix(df_train)
                        df_alternating = CT_models.create_alternating_dataframe(df_target)
    
                        params = CT_models.affine_transformation(inputs, df_alternating)
                        params = params * -1
                        params_df = pd.DataFrame(params)
    
          # Apply affine transformation to the user input data train data
                        X_transformedT, Y_transformedT = CT_models.apply_affine_transformation1(params,
                                                         df_train['X'].to_numpy(), df_train['Y'].to_numpy() )
                     
                        transformed_dataT, y_train = CT_models.transform_and_compute_errors(X_transformedT,
                                                                                            Y_transformedT, y)
        
                        # Apply affine transformation to the user input data test data
                        transformed_data, Y_transformed = CT_models.apply_affine_transformation2(params, 
                                                        df_Xtest['X'].to_numpy(), df_Xtest['Y'].to_numpy() )
                       
    
                        transformed_data, y_test = CT_models.transform_and_compute_errors(transformed_data,
                                                                                            Y_transformed, y1)
                      #  st.write("Affine Transformation Errors:", errors)  # Calculate and display RMSE
                        train_rmse_Affine = np.sqrt(mean_squared_error(y, transformed_dataT ))
                        test_rmse_Affine = np.sqrt(mean_squared_error(y1, transformed_data))
             
                       # Train the neural network
                        
                        train_predictions, test_predictions, train_rmse, test_rmse, transf_coord,neural_network_model, scaler_X,scaler_y = CT_models.train_and_evaluate_neural_network(X_train, y_train,
                                                                                                    X_test, y_test, transformed_data)
                     
                        col1,col2 = st.columns(2)
                        with col1:
                            st.markdown(f'<h1 style="color:#FF33E9;font-size:18px;">{f"Training RMSE MLR_AFFINE: {train_rmse:.2f}"}</h1>', unsafe_allow_html=True)
                        with col2:
                            st.markdown(f'<h1 style="color:#FF33E9;font-size:18px;">{f"Testing RMSE MLR_AFFINE: {test_rmse:.2f}"}</h1>', unsafe_allow_html=True)
    
                        # Convert the DataFrame to CSV
                        csv = transf_coord.to_csv(index=False)
                                                
                        st.write("Download files")
                        # Convert the DataFrame to CSV
                        csv1, csv_file_name  = params_df.to_csv(index=False), "params.csv"
    
                     # Provide a name for the CSV file
                     
                        csv_file_name = "transf_coord.csv"
                        col1, col2,col3 = st.columns([2,1.5,3])  
                        with col1:
                            st.download_button( 
                                        label="Affine Parameters",
                                        data=csv1,
                                        file_name=csv_file_name,
                                        mime="text/csv",
                            )
                        with col2:
                            st.download_button( 
                                    label="Test Data",                                                
                                    data=csv,
                                    file_name=csv_file_name,
                                    mime="text/csv",
                                                )   
                        with col3:
                        # Function to save the ANN model
                            def save_ann_model(model):
                                buffer = io.BytesIO()
                                pickle.dump(model, buffer)
                                buffer.seek(0)
                                return buffer
                        
                                # Save and download the model
                            buffer = save_ann_model(neural_network_model)
                            st.download_button(label="Trained ANN-FFINE Model", 
                                               data=buffer, file_name="ann_model.pkl", 
                                               mime="application/octet-stream")
    
    
    # load new data
    
                                    
                    if 'show_text2' not in st.session_state:
                        st.session_state.show_text2 = True
        
                    # Function to toggle the visibility of the text
                    def toggle_text():
                        st.session_state.show_text2 = not st.session_state.show_text2
        
                    # Display the button to toggle the text
                    st.button(" **Transform Unknown Data** ", on_click=toggle_text)
        
                    # Conditionally display the text based on session state
                    if st.session_state.show_text2:
                    
                        #
                        uploaded_file1 = st.file_uploader("Upload data to transform", type=['csv'])   
                        
                        if uploaded_file1 is not None:
                            user_data1 = pd.read_csv(uploaded_file1)
                            user_data1 = user_data1.dropna()
                            # Initialize session state if it doesn't already exist
                            if 'show_text1' not in st.session_state:
                                st.session_state.show_text1 = True
        
                            # Function to toggle the visibility of the text
                            def toggle_text():
                                st.session_state.show_text1 = not st.session_state.show_text1
        
                            # Display the button to toggle the text
                            st.button(" **User Data Preview** ", on_click=toggle_text)
        
                            # Conditionally display the text based on session state
                            if st.session_state.show_text1:
                                st.write( user_data1.head(4))    
                                                   
                            # Multi-select for feature columns
                            name1= st.selectbox("Select Points IDS",
                                                user_data1.columns)
                            feature_columns1 = st.multiselect(
                                "Select exactly two feature columns in this order (E N)",
                                user_data1.columns
                            )
                        
                            # Ensure exactly two feature columns are selected
                            if len(feature_columns1) != 2:
                                st.error("Please select exactly two feature columns ")
                            else:
                           
                                    P_ID1 = user_data1[name1].to_numpy()
                                    X1 = user_data1[feature_columns1].to_numpy()
                                   
                                    data_X = pd.DataFrame(X1, columns=['X', 'Y'])
                                    X_T, Y_T = CT_models.apply_affine_transformation1(params,
                                                                     data_X['X'].to_numpy(), data_X['Y'].to_numpy() )
                                    
                                    data_X['name'] = P_ID1
                                    data_Scaled = scaler_X.transform(X1)
                                    # Make predictions
                                    preds_scaled = neural_network_model.predict(data_Scaled)
                                    preds = scaler_y.inverse_transform(preds_scaled)
                                    ene_11 = pd.DataFrame(X_T)
                                    ene_12 = pd.DataFrame(Y_T)
                                    
                                    # Join the DataFrames
                                    m11 = ene_11.join(ene_12, how='right', lsuffix='_left', rsuffix='_right')
                                    
                                    # Create a DataFrame from the joined data
                                    dataT = pd.DataFrame(m11)
                                    transf_cd = dataT + preds
                                   
                                    # Join the DataFrames
                                    m12 =data_X.join(transf_cd, how='right', lsuffix='_left', rsuffix='_right')
                                    
                                    # Create a DataFrame from the joined data
                                    transformed_dataFinal = pd.DataFrame(m12)
                                               # Save the trained model to a file
                                    transformed_dataFinal = transformed_dataFinal.rename(columns={"0_left": "Easting", "0_right": "Northing"})
                                    st.write("transformed data", transformed_dataFinal.head(3))
                                    print("ANN model saved successfully.", transformed_dataFinal)
                                    
                                    findata = transformed_dataFinal.to_csv(index=False)
                                    st.download_button( 
                                          label="Transformed Data",                                                
                                          data=findata,
                                          file_name=csv_file_name,
                                          mime="text/csv",
                                                      )  
                            
        
    
    



                                
                elif transformation_type == 'Cassini to UTM':
                    if 'show_text3' not in st.session_state:
                        st.session_state.show_text3 = False
        
                    # Function to toggle the visibility of the text
                    def toggle_text():
                        st.session_state.show_text3 = not st.session_state.show_text3
        
                    # Display the button to toggle the text
                    st.button(" **Compute Affine Parameters & Train Model** ", on_click=toggle_text)
        
                    # Conditionally display the text based on session state
                    if st.session_state.show_text3:
                        
                                               
                        
                          X_train, X_test, y, y1 = CT_models.split_data(X, target)
                          # for all data
                          
                          # create a matrix
                          df_train, df_target, df_Xtest, E, N = CT_models.create_dataframes(X_train, y, X_test)
                          # for all data
                          
                          inputs = CT_models.initialize_matrix(df_train)
                          # for all data
                     
                          df_alternating = CT_models.create_alternating_dataframe(df_target)
                          # for all data
                         
                          params = CT_models.affine_transformation(inputs, df_alternating)
                          # for all data
                          params = params * -1
                          params_df = pd.DataFrame(params)
                          
                          # Convert the DataFrame to CSV
                          csv1, csv_file_name  = params_df.to_csv(index=False), "params.csv"
      
                    
            # Apply affine transformation to the user input data train data
                          X_transformedT, Y_transformedT = CT_models.apply_affine_transformation1(params,
                                                           df_train['X'].to_numpy(), df_train['Y'].to_numpy() )
                       # for all data
                          allX_transformedT, allY_transformedT = CT_models.apply_affine_transformation1(params,
                                                           all_data['X'].to_numpy(), all_data['Y'].to_numpy() )
                          
                          transformed_dataT, y_train = CT_models.transform_and_compute_errors(X_transformedT,
                                                                                         Y_transformedT, y)
          
                          # for all data
                          all_transformed_dataT, all_y_train = CT_models.transform_and_compute_errors(allX_transformedT,
                                                                                         allY_transformedT, all_targets)
                          
                          # Apply affine transformation to the user input data test data
                          transformed_data, Y_transformed = CT_models.apply_affine_transformation2(params, 
                                                          df_Xtest['X'].to_numpy(), df_Xtest['Y'].to_numpy() )
                         
      
                          
                          transformed_data, y_test = CT_models.transform_and_compute_errors(transformed_data,
                                                                                              Y_transformed, y1)
                        #  st.write("Affine Transformation Errors:", errors)  # Calculate and display RMSE
                          train_rmse_Affine = np.sqrt(mean_squared_error(y, transformed_dataT ))
                          test_rmse_Affine = np.sqrt(mean_squared_error(y1, transformed_data))
               
                         # Train the neural network
                                             
                          
                          train_predictions, test_predictions, train_rmse, test_rmse, transf_coord,neural_network_model,all_data_final, scaler_X,scaler_y= CT_models.train_and_evaluate_neural_network1(X_train, y_train,
                           # for all data
                                                                                                                                               X_test, y_test, transformed_data, all_data, all_transformed_dataT)
                       
                          
                          transf_coord.columns=['E','N']
                          #st.write("**Sample Transformed Coordinate**",transf_coord.head()) 
                          if show_map == 'Yes':
                                                      
                              X_map1 = all_data_final.rename(columns={"0_left": "Easting", "0_right": "Northing"})
                            
                              
                              
                              
                              X_map1['name'] = P_ID 
                              X_map1 = X_map1.dropna()
                              
                              print("ALL DATA.", all_data_final)
                              print("ANN model saved successfully.", all_data_final.columns)
                              utm_latlon = CT_models.convert_utm_to_latlon1(X_map1, zone_number, zone_letter)
                     
                              with st.spinner('INITIALISING...'):
                                  time.sleep(2)
                                  CT_models.add_markers_to_map(utm_latlon, m)
                                  st.success('Done!')
                           
                          elif show_map =='No':
                             # map1.get_root().clear_layers()
                            a=2
                       
                          # Convert the DataFrame to CSV
                          df_X_test = pd.DataFrame(X_test, columns=['X','Y'])
                          csv = pd.concat([df_X_test,transf_coord], axis=1)
                          csv = csv.to_csv(index=False)
                          col1,col2 = st.columns(2)
                          with col1:
                                st.markdown(f'<h1 style="color:#FF33E9;font-size:18px;">{f"Training RMSE MLR_AFFINE: {train_rmse:.2f}"}</h1>', unsafe_allow_html=True)
                          with col2:
                                st.markdown(f'<h1 style="color:#FF33E9;font-size:18px;">{f"Testing RMSE MLR_AFFINE: {test_rmse:.2f}"}</h1>', unsafe_allow_html=True)
####
                            # Convert the DataFrame to CSV
                          csv = transf_coord.to_csv(index=False)
                        
                          
                          st.write("Download files")
                      # Provide a name for the CSV file
                          csv_file_name = "transf_coord.csv"
                          col1, col2,col3 = st.columns([2.5, 3,2])
                          with col1:
                              st.download_button( 
                                          label="Parameters",
                                          data=csv1,
                                          file_name=csv_file_name,
                                          mime="text/csv",
                          )
                          with col2:
                              st.download_button( 
                                      label="Test Data",                                                
                                      data=csv,
                                      file_name=csv_file_name,
                                      mime="text/csv",
                                                  )   
                          with col3:
                          # Function to save the ANN model
                              def save_ann_model(model):
                                  buffer = io.BytesIO()
                                  pickle.dump(model, buffer)
                                  buffer.seek(0)
                                  return buffer
                          
                                  # Save and download the model
                              buffer = save_ann_model(neural_network_model)
                              st.download_button(label=" Trained MLR-AFFINE Model", 
                                                 data=buffer, file_name="ann_model.pkl", 
                                                 mime="application/octet-stream")
                 
                          
                                      
                          if 'show_text4' not in st.session_state:
                              st.session_state.show_text4 = True
              
                          # Function to toggle the visibility of the text
                          def toggle_text():
                              st.session_state.show_text4 = not st.session_state.show_text4
              
                          # Display the button to toggle the text
                          st.button(" **Transform your Data** ", on_click=toggle_text)
              
                          # Conditionally display the text based on session state
                          if st.session_state.show_text4:
                          
                              #
                              uploaded_file1 = st.file_uploader("Upload data to transform", type=['csv'])   
                              
                              if uploaded_file1 is not None:
                                  user_data1 = pd.read_csv(uploaded_file1)
                                  user_data1 = user_data1.dropna()
                                  # Initialize session state if it doesn't already exist
                                  if 'show_text1' not in st.session_state:
                                      st.session_state.show_text1 = True
              
                                  # Function to toggle the visibility of the text
                                  def toggle_text():
                                      st.session_state.show_text1 = not st.session_state.show_text1
              
                                  # Display the button to toggle the text
                                  st.button(" **User Data Preview** ", on_click=toggle_text)
              
                                  # Conditionally display the text based on session state
                                  if st.session_state.show_text1:
                                      st.write( user_data1.head(4))    
                                                         
                                  # Multi-select for feature columns
                                  name1= st.selectbox("Select Points IDS",
                                                      user_data1.columns)
                                  feature_columns1 = st.multiselect(
                                      "Select exactly two feature columns in this order (E N)",
                                      user_data1.columns
                                  )
                              
                                  # Ensure exactly two feature columns are selected
                                  if len(feature_columns1) != 2:
                                      st.error("Please select exactly two feature columns ")
                                  else:
                                 
                                          P_ID1 = user_data1[name1].to_numpy()
                                          X1 = user_data1[feature_columns1].to_numpy()
                                         
                                          data_X = pd.DataFrame(X1, columns=['X', 'Y'])
                                          X_T, Y_T = CT_models.apply_affine_transformation1(params,
                                                                           data_X['X'].to_numpy(), data_X['Y'].to_numpy() )
                                          
                                          data_X['name'] = P_ID1
                                          data_Scaled = scaler_X.transform(X1)
                                          # Make predictions
                                          preds_scaled = neural_network_model.predict(data_Scaled)
                                          preds = scaler_y.inverse_transform(preds_scaled)
                                          ene_11 = pd.DataFrame(X_T)
                                          ene_12 = pd.DataFrame(Y_T)
                                          
                                          # Join the DataFrames
                                          m11 = ene_11.join(ene_12, how='right', lsuffix='_left', rsuffix='_right')
                                          
                                          # Create a DataFrame from the joined data
                                          dataT = pd.DataFrame(m11)
                                          transf_cd = dataT + preds
                                         
                                          # Join the DataFrames
                                          m12 =data_X.join(transf_cd, how='right', lsuffix='_left', rsuffix='_right')
                                          
                                          # Create a DataFrame from the joined data
                                          transformed_dataFinal = pd.DataFrame(m12)
                                                     # Save the trained model to a file
                                          transformed_dataFinal = transformed_dataFinal.rename(columns={"0_left": "Easting", "0_right": "Northing"})
                                          st.write("transformed data", transformed_dataFinal .head(3))
                                          print("ANN model saved successfully.", transformed_dataFinal)
                                          
                                          findata = transformed_dataFinal.to_csv(index=False)
                                          st.download_button( 
                                                label="Transformed Data",                                                
                                                data=findata,
                                                file_name=csv_file_name,
                                                mime="text/csv",
                                                            )  
            
                         
                        
                          
    if __name__ == '__main__':
        main()

with col1:
    m.to_streamlit()
    st.write(strings.aims)
    st.write(strings.footnote,strings.footnote1, strings.footnote2,
             strings.footnote3,strings.footnote4)

   
