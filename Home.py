import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Investor Dashboard",
    page_icon="✨",
)

st.write("# Investor Dashboard")
image = Image.open('./images/home8.jpg')
st.image(image)
# st.sidebar.success("Select a investment option above")
image = Image.open('./images/home6.jpg')
st.sidebar.subheader("Select a investment option above 👆 ")
st.sidebar.image(image)
st.sidebar.caption('Presented by Jeff, Thomas and Ray :hotsprings:')