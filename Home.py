# home page creation

# imports
import streamlit as st
from PIL import Image
# setup title page tab
st.set_page_config(
    page_title="Investor Dashboard",
    page_icon="✨",
)
# setup the primary area space
st.write("# ✨Investor Dashboard")
st.markdown('### Enhanced investment insights & tools to better serve you.')
st.markdown('<div style="text-align: justify;"> Get investment information straight from our dashboard in a modern way to research the market. Whether its stocks to crypto we have you covered. Our dashboard allows for simple, direct access to investment information with less website jumping, and more flexibility to see what you want to see.</div>', unsafe_allow_html=True)
st.markdown(' ')
st.markdown(' ')
image = Image.open('./images/home8.jpg')
st.image(image)
# setup the sidebar information
image = Image.open('./images/home6.jpg')
st.sidebar.subheader("Let's get started select a page 👆")
st.sidebar.image(image)
st.sidebar.caption('Presented by Jeff, Thomas and Ray :hotsprings:')
