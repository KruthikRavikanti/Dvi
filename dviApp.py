from altair.utils.schemapi import textwrap
from markdownlit import mdlit
import pandas as pd
import streamlit as st
import numpy as np
from streamlit_card import card
import yfinance as yf
import graphviz
import altair as alt

# Set the background color and opacity for the container
container_style = """
    background-color: rgba(55, 65, 82, 0.7);
    padding: 100px;
    border-radius: 10px;
    margin-top: 20px;
    margin-bottom: 20px;
"""

st.markdown("<h1 style='text-align: center;'>Volatility Indicator</h1>", unsafe_allow_html=True)
st.write(""" ### Economic Volitility Examination""")
# Create a translucent container
x = card(title="", text = "When we talk about stock volitility, we typically need fundamental data like company earning reports, interest rates, and technical analysis trends",
         styles = {
            "card": {
                "width": "650px",
                "height": "200px",
                "background-color": "rgba(55, 65, 82, 1)",
                "padding": "20px",
                "margin-top": "20px",
                #"margin-bottom": "20px",
            },
            
         }

         )


# Download historical stock data for Tesla
ticker = "TSLA"
start_date = "2021-09-29"
end_date = "2022-09-29"

stock_data = yf.download(ticker, start=start_date, end=end_date)



# Calculate daily returns
stock_data["Daily_Return"] = stock_data["Close"].pct_change()

# Calculate historical volatility (standard deviation)
historical_volatility = stock_data["Daily_Return"].std()

# Streamlit app
st.markdown("<h1 style='text-align: center;'>Tesla Stock Volatility Analysis</h1>", unsafe_allow_html=True)


# Display historical stock data
st.subheader("Historical Stock Data")
st.write(stock_data)



stock_data = yf.download(ticker, start=start_date, end=end_date)

# Calculate daily returns
stock_data["Daily_Return"] = ((stock_data["Close"] / stock_data["Open"]) - 1)
notable = []
days = []
for day in stock_data["Daily_Return"]:
    if day > 0.1:
        notable.append(day)
        days.append("Date")
    elif day < -0.1:
        notable.append(day)
        days.append("Date")




# Line chart for stock prices
st.subheader("Tesla Stock Prices Over Time")
line_chart = alt.Chart(stock_data.reset_index()).mark_line().encode(
    x="Date:T",
    y="Daily_Return",
    tooltip=["Date", "Daily_Return"]
).properties(width=800, height=400)
st.altair_chart(line_chart, use_container_width=True)

st.write("2021-11-09 00:00:00: \"Telsa fire in Stanford took 42 minutes to extinguish\" ")
st.write("2022-01-27 00:00:00: \"Tesla drops more than 11% as investors digest new vehicle delays\"")
st.write("2022-02-23 00:00:00: \"Tesla model Y wins EV award\"")
st.write("2022-04-26 00:00:00: \"Elon Musk says people might download their personalities onto a human robot constructed by Tesla\"")


st.markdown("<h1 style='text-align: center;'>Our Approach</h1>", unsafe_allow_html=True)

x = card(title="", 
         text = "How can we predict the potential social impact on stock volitility? Qualitative tabular data poses a challenge concerning data processing resources",
         styles = {
            "card": {
                "width": "650px",
                "height": "200px",
                "background-color": "rgba(55, 65, 82, 1)",
                "padding": "50px",
                "margin-top": "10px",
                "margin-bottom": "10px",
            },
            
         }

         )
st.write("")


# Streamlit app
st.title('First Model')
model1 = card(title="", 
         text = "",
         styles = {
            "card": {
                "width": "650px",
                "height": "200px",
                "margin-top": "10px",
                "margin-bottom": "10px",
            },
         },
         image="https://i.postimg.cc/Bn8q0Ddy/XBoost.png",
         on_click=lambda: st.write("The model generating embeddings represent the data in the prompt. Each embedding captures an immense amount of training data that is then used to project desired data")

         )
st.title('Second Model')
mod2 = card(title="", 
         text = "",
         styles = {
            "card": {
                "width": "700px",
                "height": "400px",
                "margin-top": "20px",
                "margin-bottom": "20px",
            }
         },
         image="https://miro.medium.com/v2/resize:fit:976/1*oc1gaCFvgWXq_gHQFM63UQ.png",
         on_click=lambda: st.write("A neural network learns to map input data to output by adjusting the strengths of connections (weights) between nodes during a training process. This enables the network to recognize patterns and make predictions on new data.")

         )
