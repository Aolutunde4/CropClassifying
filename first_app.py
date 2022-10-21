import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

crops = pd.read_csv(r"Crop_recommendation.csv")

X = crops[['N','P','K','temperature','humidity','ph','rainfall']]
y = crops['label']

st.write("""
# Introduction

###### Welcome to my first streamlit application!
###### We help recommend what to farm to beginning farmers.
###### We have data on 22 different crops!

""")
st.write(pd.DataFrame({"Crop":crops['label'].unique()}))

st.sidebar.header("Farmer Input Parameters")

def user_input_features():
    N = st.sidebar.slider('Nitrogen', 0, 215, 13)
    P = st.sidebar.slider('Phosphorus', 0, 215, 7)
    K = st.sidebar.slider('Potassium', 0, 215, 43)
    temperature = st.sidebar.slider('Temperature', 0,50,18)
    humidity = st.sidebar.slider('Humidity',10,100,91)
    ph= st.sidebar.slider('Ph level',0,14,7)
    rainfall = st.sidebar.slider('Rainfall',0,300,110)
    data = {'Nitrogen': N,
            'Phosphorus':P,
            'Potassium':K,
            'Temperature (C)':temperature,
            'Humidity':humidity,
            'Ph level':ph,
            'Rainfall':rainfall}
    features = pd.DataFrame(data, index=[0])
    
    return features

df = user_input_features()

st.subheader("User input parameters")

st.write(df)

st.write("""

#### Change parameters by clicking the top left arrow and enjoy!

""")

model = LogisticRegression(C=0.01, penalty = 'l2', solver='newton-cg')
model.fit(X,y)

prediction = model.predict(df)
prediction_probability = model.predict_proba(df)

st.write("""

##### Prediction:

""")
st.write(prediction)

# st.subheader("Prediction Probability for each crop")
# st.write(sorted(prediction_probability[:5], reverse=True))
