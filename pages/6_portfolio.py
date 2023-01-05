##########################
# portfolio page setup #
##########################

#pip install yfinance
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import webbrowser
from PIL import Image

##################
# Set up sidebar #
##################
# set sidebar title 
st.sidebar.title('My Portfolio :heavy_dollar_sign:')
    
# from PIL import Image 
image = Image.open('./images/portfolio.jpg')
st.sidebar.image(image)
# load stock symbols list
option = st.sidebar.selectbox('Select Portfolio', ('Schwab','Fidelity','Robinhood','E*TRADE','TD Ameritrade','SoFi','MERRILL','J.P.Morgan'))


# add creator information
st.sidebar.caption('Presented by Jeff, Thomas and Ray :hotsprings:')

##################
# Portfolio data #
##################
# setup of the main body window
st.title('Investment Portfolio')


# add a progress bar
progress_bar = st.progress(0)
st.subheader("Future Enhancement - BETA Development")

