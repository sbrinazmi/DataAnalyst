import streamlit as st
import pandas as pd
import numpy as np 
from prediction import predict 

st.title('Classifying Iris Flowers')
st.markdown('Toy model to play to classify iris flowers into \ (setosa, versicolor, virginica) based on their sepal/petal \   and length/width.')

st.header("Plant Features")
col1, col2 = st.columns(2)

with col1:
  st.text("Sepal characteristics")
  sepal_1 = st.slider('Sepal length (cm)', 1.0, 8.0, 0.5)
  sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)

with col2:
  st.text("Pepal characteristics")
  petal_1 = st.slider('Petal length (cm)', 1.0, 7.0, 0.5)
  petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)  

st.text('')
if st.button("predict Type of Iris"):
  result = predict(
    np.array([[sepal_1, sepal_w, petal_1, petal_w]]))
      st.text(result[0])

st.text('')
st.text('')
st.markdown('')

