import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Investor Dashboard",
    page_icon="✨",
)

st.write("# ✨Investor Dashboard")
st.markdown('### Enhanced investment insights & tools to better serve you.')
st.markdown('<div style="text-align: justify;"> Get investment information straight from our dashboard in a modern way to research the market. Whether its stocks to crypto we have you covered. Our dashboard allows for simple, direct access to investment information with less website jumping, and more flexibility to see what you want to see.</div>', unsafe_allow_html=True)
st.markdown(' ')
st.markdown(' ')
image = Image.open('./images/home8.jpg')
st.image(image)
# st.sidebar.success("Select a investment option above")
image = Image.open('./images/home6.jpg')
st.sidebar.subheader("Select a investment option above 👆 ")
st.sidebar.image(image)
st.sidebar.caption('Presented by Jeff, Thomas and Ray :hotsprings:')


# Enhanced investment insights & tools to better serve you
# Get investment information straight from the Hub; a modern way to review your investment accounts.