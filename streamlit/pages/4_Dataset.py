import streamlit as st
import pandas as pd

data_path = "./Data/dummy_iot_dataset_last_en.csv"

st.set_page_config(page_title="Veri Seti")

@st.cache_data
def load_data(data_path : str):
    df = pd.read_csv(data_path)
    return df

data = load_data(data_path=data_path)


st.subheader("Projede Kullanılan Veriseti")
st.write(data)