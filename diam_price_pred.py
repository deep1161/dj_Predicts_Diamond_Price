import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib as jb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


st.header("Diamond Price Prediction using ML")
st.markdown("#### This classic dataset contains the prices and other attributes of almost 54,000 diamonds. It's a great dataset for beginners learning to work with data analysis and visualization.")
st.markdown("\n")

# creating function which takes user input
def input_diam(data):
	st.sidebar.markdown("## User Input Features")

	carat = st.sidebar.slider("Carat Weight",0.2,5.01,0.23)
	
	cut = st.sidebar.selectbox("Cut Quality",data["cut"].unique().tolist())
	cut_val = dict_cut[cut]

	color = st.sidebar.selectbox("Color Quality",sorted(data["color"].unique().tolist()))
	color_val = dict_color[color]

	clarity = st.sidebar.selectbox("Clarity Level",["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"])
	clarity_val = dict_clarity[clarity]


	table = st.sidebar.slider("Table Width",43.0,95.0,58.0)

	x = st.sidebar.slider("Length (in mm)",0.0,10.8,4.01)
	
	y = st.sidebar.slider("Width (in mm)",0.0,58.9,13.4)

	z = st.sidebar.slider("Depth (in mm)",0.0,31.8,12.3)

	val = (2*z)/(x+y)

	depth = val

	return [carat, cut_val, color_val, clarity_val, depth, table, x, y, z]


# reading csv file
data = pd.read_csv("diamonds.csv")

# creating dictionaries for conversion later
dict_cut = {'Ideal':5, 'Premium':4, 'Good':3, 'Very Good':2, 'Fair':1}
dict_color = {'D':7, 'E':6, 'F':5, 'G':4, 'H':3, 'I':2, 'J':1}
dict_clarity = {'I1':1,'SI2':2,'SI1':3,'VS2':4,'VS1':5,'VVS2':6,'VVS1':7,'IF':8}

# Calling input function
df = input_diam(data)

# Data Preprocessing
data["cut"].replace(['Ideal', 'Premium', 'Good', 'Very Good', 'Fair'],[5,4,3,2,1], inplace=True)
data["color"].replace(['D', 'E', 'F', 'G', 'H', 'I', 'J'], [7,6,5,4,3,2,1], inplace=True)
data["clarity"].replace(['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'], [1,2,3,4,5,6,7,8], inplace=True)

# Data splitting and Feature Scaling
X = data.drop("price", axis=1)
y = data["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Loading saved ML model .model file
rf = jb.load('diamond_model.model')

# Transforming to Scaling level
df = scaler.transform([df])

# Predicting Price of diamond
res = rf.predict(df)

# Displaying Model Score
st.markdown("### Model Score : ")
mod_score = rf.score(X_train, y_train)
st.write(np.round(mod_score*100,4),"%")

# Displaying Predicted Price
st.markdown("### Predicted Price : ")
st.write("$",res[0])
