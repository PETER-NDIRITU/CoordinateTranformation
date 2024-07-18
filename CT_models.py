# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:35:36 2024

@author: HP -moses
"""
import pandas as pd
import numpy as np
from pyproj import Proj, transform
from pathlib import Path

from sklearn.metrics import mean_squared_error
from keras.models import load_model
import folium

from sklearn.model_selection import train_test_split
import pandas as pd
#loaded_model = load_model()
# loading the saved model
cass_to_utm_model = load_model('data/UCassToUtm_model.keras')
#load training data
data = pd.read_csv('data/utmTrain.csv',delimiter=',', 
                   usecols=['name', 'E', 'N'], encoding='ISO-8859-1') 
data = data.dropna()
data.columns= ['name', 'Easting','Northing']


# Load UTM to Cassini model
utm_to_cass_model = load_model('data/UtmToCass_model.keras')

# Load parameters and scaler for UTM to Cassini transformation
params_cass = pd.read_csv('data/paramsCass.csv', header=None).to_numpy()
scaler_utm_to_cass_T = MinMaxScaler()
scaler_utm_to_cass_L = MinMaxScaler()
error1 =pd.read_csv('data/aff_errorCass.csv') 
scaler_utm_to_cass_T.fit(pd.read_csv('data/utmTrain.csv', delimiter=',', usecols=['E', 'N']))
scaler_utm_to_cass_L.fit(error1)
# Load Cassini to UTM model


# Load parameters and scaler for Cassini to UTM transformation
params_utm = pd.read_csv('data/paramsUC.csv', header=None).to_numpy()
scaler_cass_to_utm_T = MinMaxScaler()
error2 = pd.read_csv('data/aff_errorUtm.csv')
scaler_cass_to_utm_L = MinMaxScaler()
scaler_cass_to_utm_T.fit(pd.read_csv('data/cassTrain.csv'))

scaler_cass_to_utm_L.fit(error2)


# Function to perform affine transformation
def affine_transformation(inputs, targets):
    A = inputs.to_numpy()
    At = A.T
    ATA = np.dot(At, A)
    ATL = np.dot(At, targets.to_numpy())
    
    m_inv = np.linalg.inv(ATA) * -1
    x1 = np.dot(m_inv, ATL)
    
    return x1



# Function to apply affine transformation
def apply_affine_transformation(params, data):
    a, b, c, d, e, f = params
    X = data[:, 0]
    Y = data[:, 1]
    
    X_1 = a * X + b * Y + c
    Y_1 = d * X + e * Y + f
    
    return np.column_stack((X_1, Y_1))

def apply_affine_transformation1(params, x, y):
    a, b, c, d, e, f = params
   
    X_1 = a * x + b * y + c
    Y_1 = d * x + e * y + f
    
    return X_1, Y_1

# Function to perform UTM to Cassini transformation
def utm_to_cass_transform(input_data):
    
    input_data_normalized = scaler_utm_to_cass_T.transform(input_data)
    aff_coords = apply_affine_transformation(params_cass, input_data)
    prediction = utm_to_cass_model.predict(input_data_normalized)
    nn_preds = scaler_utm_to_cass_L.inverse_transform(prediction)
    cassini_coords = nn_preds + aff_coords
    return cassini_coords

def utm_to_cass_transform1(input_data):
    
    input_data_normalized = scaler_utm_to_cass_T.transform(input_data)
    aff_coords = apply_affine_transformation(params_cass, input_data)
    prediction = utm_to_cass_model.predict(input_data_normalized)
    nn_preds = scaler_utm_to_cass_L.inverse_transform(prediction)
    cassini_coords = nn_preds + aff_coords
    
    # Create a DataFrame to store transformed coordinates
    transformed_df = pd.DataFrame({
        'Original_Easting': input_data[:, 0],
        'Original_Northing': input_data[:, 1],
        'Transformed_X': cassini_coords[:, 0],
        'Transformed_Y': cassini_coords[:, 1],
    })
    
    # You can write the DataFrame to a CSV file if needed
    transformed_df.to_csv('transformed_coordinates.csv', index=False)
    return transformed_df

# Function to perform Cassini to UTM transformation
def cass_to_utm_transform(input_data):
    input_data_normalized = scaler_cass_to_utm_T.transform(input_data)
    aff_coords = apply_affine_transformation(params_utm, input_data)
    prediction = cass_to_utm_model.predict(input_data_normalized)
    nn_preds = scaler_cass_to_utm_L.inverse_transform(prediction)
    utm_coords = nn_preds + aff_coords
    return utm_coords

def cass_to_utm_transform1(input_data):
    input_data_normalized = scaler_cass_to_utm_T.transform(input_data)
    aff_coords = apply_affine_transformation(params_utm, input_data)
    prediction = cass_to_utm_model.predict(input_data_normalized)
    nn_preds = scaler_cass_to_utm_L.inverse_transform(prediction)
    utm_coords = nn_preds + aff_coords
    # Create a DataFrame to store transformed coordinates
    transformed_df = pd.DataFrame({
        'Original_X': input_data[:, 0],
        'Original_Y': input_data[:, 1],
        'Transformed_Easting': utm_coords[:, 0],
        'Transformed_Northing': utm_coords[:, 1],
    })
    # You can write the DataFrame to a CSV file if needed
    transformed_df.to_csv('transformed_coordinates.csv', index=False)
    return transformed_df
    # for map 

def convert_utm_to_latlon(df, zone_number, zone_letter):
    utm_proj = Proj(proj='utm', zone=zone_number, ellps='WGS84', south=(zone_letter < 'N'))
    lonlat_proj = Proj(proj='latlong', datum='WGS84')
    
    lon, lat = transform(utm_proj, lonlat_proj, df['Easting'].values, df['Northing'].values)
    
    return pd.DataFrame({'Latitude': lat, 'Longitude': lon,'Easting': df['Easting'],'Northing': df['Northing'] })

df_latlon = convert_utm_to_latlon(data, 37, 'M')
extracted_col = data["name"]
center_lat = df_latlon['Latitude'].mean()
center_lon = df_latlon['Longitude'].mean()

df_latlon= df_latlon.join(extracted_col)


def convert_utm_to_latlon1(df, zone_number, zone_letter):
    utm_proj = Proj(proj='utm', zone=zone_number, ellps='WGS84', south=(zone_letter < 'N'))
    lonlat_proj = Proj(proj='latlong', datum='WGS84')
    
    lon, lat = transform(utm_proj, lonlat_proj, df['Easting'].values, df['Northing'].values)
    
    return pd.DataFrame({'Latitude': lat, 'Longitude': lon,'Easting': df['Easting'],'Northing': df['Northing'], 'name': df['name'] })


# for Training
# Define function to update map center based on user input

def update_map_center(df):
    new_center_lat = df['Latitude'].mean()
    new_center_lon = df['Longitude'].mean()
    return new_center_lat, new_center_lon


    """
    Add markers to a folium map based on a DataFrame and display the map in a Streamlit app.
    
    Parameters:
    utm_latlon (pd.DataFrame): DataFrame containing 'Latitude', 'Longitude', 'name', 'Easting', and 'Northing' columns.
    """
def add_markers_to_map(latlon,map):
    center_lat, center_lon = latlon['Latitude'].mean(),latlon['Longitude'].mean()

    # Add markers to the map
    for i, row in latlon.iterrows():
        user_marker = {
            'location': [row['Latitude'], row['Longitude']],
            'popup': f'User Input: ({row["name"]},{row["Easting"]}, {row["Northing"]})',
            'tooltip': f'{row["name"]}',
            'icon': folium.Icon(color='orange', icon='info-sign')
        }
        folium.Marker(**user_marker).add_to(map)



def split_data(X, target, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    
    Parameters:
    X (pd.DataFrame): Features.
    target (pd.Series): Target variable.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Controls the shuffling applied to the data before the split.
    
    Returns:
    X_train (pd.DataFrame): Training features.
    X_test (pd.DataFrame): Testing features.
    y_train (pd.Series): Training target variable.
    y_test (pd.Series): Testing target variable.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def create_dataframes(X_train, y_train, X_test):
    """
    Create dataframes for training and testing data.
    
    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target variable.
    X_test (pd.DataFrame): Testing features.
    
    Returns:
    df_train (pd.DataFrame): DataFrame of training features.
    df_target (pd.DataFrame): DataFrame of training target variable.
    df_Xtest (pd.DataFrame): DataFrame of testing features.
    E (np.ndarray): Array of 'X' values from the testing features.
    N (np.ndarray): Array of 'Y' values from the testing features.
    """
    df_train = pd.DataFrame(X_train, columns=['X', 'Y'])
    df_target = pd.DataFrame(y_train)
    df_Xtest = pd.DataFrame(X_test, columns=['X', 'Y'])
    
    E = df_Xtest['X'].values
    N = df_Xtest['Y'].values
  
    return df_train, df_target, df_Xtest, E, N


def initialize_matrix(df_train):
    """
    Create a matrix format with IDs as the first row and column.
    
    Parameters:
    df_train (pd.DataFrame): DataFrame of training features.
    
    Returns:
    matrix (pd.DataFrame): Initialized matrix.
    """
    matrix_size = len(df_train) * 2
    matrix = pd.DataFrame(index=range(matrix_size), columns=range(6), data=0)
    
    for i in range(len(df_train)):
        matrix.iloc[i * 2, :3] = [df_train.loc[i, 'X'], df_train.loc[i, 'Y'], 1]
        matrix.iloc[i * 2 + 1, 3:] = [df_train.loc[i, 'X'], df_train.loc[i, 'Y'], 1]
   
    return matrix

def create_alternating_dataframe(df_target):
    """
    Create a new DataFrame with the alternating values in a single column.
    
    Parameters:
    df_target (pd.DataFrame): DataFrame of training target variable.
    
    Returns:
    df_alternating (pd.DataFrame): DataFrame with alternating values in a single column.
    """
    alternating_values = df_target.values.flatten()
    df_alternating = pd.DataFrame(alternating_values)

    return df_alternating

def apply_affine_transformation2(params, x, y):
    a, b, c, d, e, f = params
   
    X_1 = a * x + b * y + c
    Y_1 = d * x + e * y + f
    
    return X_1, Y_1

def transform_and_compute_errors(X_transformedT, Y_transformedT, y):
    """
    Create DataFrames from transformed data, join them, and compute errors.
    
    Parameters:
    X_transformedT (np.ndarray or pd.DataFrame): Transformed feature data.
    Y_transformedT (np.ndarray or pd.DataFrame): Transformed target data.
    y (pd.Series): Original target data.
    
    Returns:
    transformed_dataT (pd.DataFrame): Joined DataFrame of transformed data.
    errorsT (pd.DataFrame): DataFrame of errors between original target and transformed data.
    """
    # Create DataFrames from transformed data
    ene_11 = pd.DataFrame(X_transformedT)
    ene_12 = pd.DataFrame(Y_transformedT)
    
    # Join the DataFrames
    m11 = ene_11.join(ene_12, how='right', lsuffix='_left', rsuffix='_right')
    
    # Create a DataFrame from the joined data
    transformed_dataT = pd.DataFrame(m11)
    
    # Compute errors
    errorsT = y - transformed_dataT
    
    return transformed_dataT, errorsT
                    
#ANN MODELLING
def train_and_evaluate_neural_network1(X_train, y_train, X_test, y_test, transformed_data, all_data, all_transformed_dataT):
    """
    Normalize the data, train a neural network, make predictions, and calculate RMSE.
    
    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target variable.
    X_test (pd.DataFrame): Testing features.
    y_test (pd.Series): Testing target variable.
    transformed_data (pd.DataFrame): Transformed data to be added to predictions.
    
    Returns:
    train_predictions (np.ndarray): Inverse transformed predictions for training data.
    test_predictions (np.ndarray): Inverse transformed predictions for testing data.
    train_rmse (float): Root Mean Squared Error for training data.
    test_rmse (float): Root Mean Squared Error for testing data.
    transf_coord (pd.DataFrame): Test predictions combined with transformed data.
    """
    # Normalize the data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    all_data_scaled = scaler_X.transform(all_data)
    
    # Train the neural network
    neural_network_model = MLPRegressor(solver='lbfgs', 
                                        alpha=1e-6,
                                        hidden_layer_sizes=(8, 2),
                                        learning_rate_init=0.1,
                                        activation='tanh',
                                        max_iter=1000,
                                        random_state=24)
    neural_network_model.fit(X_train_scaled, y_train_scaled)
    
    # Make predictions
    train_predictions_scaled = neural_network_model.predict(X_train_scaled)
    test_predictions_scaled = neural_network_model.predict(X_test_scaled)
    all_data_predScaled =  neural_network_model.predict(all_data_scaled)
    # Inverse transform the predictions
    train_predictions = scaler_y.inverse_transform(train_predictions_scaled)
    test_predictions = scaler_y.inverse_transform(test_predictions_scaled)
    transf_coord = test_predictions + transformed_data
    all_data_final = scaler_y.inverse_transform(all_data_predScaled)
    all_data_final = all_transformed_dataT + all_data_final
    # Calculate and display RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    

    return train_predictions, test_predictions, train_rmse, test_rmse, transf_coord,neural_network_model, all_data_final, scaler_X,scaler_y

def train_and_evaluate_neural_network(X_train, y_train, X_test, y_test, transformed_data):
    """
    Normalize the data, train a neural network, make predictions, and calculate RMSE.
    
    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target variable.
    X_test (pd.DataFrame): Testing features.
    y_test (pd.Series): Testing target variable.
    transformed_data (pd.DataFrame): Transformed data to be added to predictions.
    
    Returns:
    train_predictions (np.ndarray): Inverse transformed predictions for training data.
    test_predictions (np.ndarray): Inverse transformed predictions for testing data.
    train_rmse (float): Root Mean Squared Error for training data.
    test_rmse (float): Root Mean Squared Error for testing data.
    transf_coord (pd.DataFrame): Test predictions combined with transformed data.
    """
    # Normalize the data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)

    
    # Train the neural network
    neural_network_model = MLPRegressor(solver='lbfgs', 
                                        alpha=1e-6,
                                        hidden_layer_sizes=(8, 2),
                                        learning_rate_init=0.1,
                                        activation='tanh',
                                        max_iter=1000,
                                        random_state=24)
    neural_network_model.fit(X_train_scaled, y_train_scaled)
    
    # Make predictions
    train_predictions_scaled = neural_network_model.predict(X_train_scaled)
    test_predictions_scaled = neural_network_model.predict(X_test_scaled)
   
    # Inverse transform the predictions
    train_predictions = scaler_y.inverse_transform(train_predictions_scaled)
    test_predictions = scaler_y.inverse_transform(test_predictions_scaled)
    transf_coord = test_predictions + transformed_data

    # Calculate and display RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    
    return train_predictions, test_predictions, train_rmse, test_rmse, transf_coord,neural_network_model,  scaler_X,scaler_y



