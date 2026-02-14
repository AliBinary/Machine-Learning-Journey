'''
Warning: to view this Streamlit app on a browser, run it with the following
command:
streamlit run "F:\Projects\Machine Learning (maktabkhooneh)\18__ Create WebApp for ML\app.py"
'''

import time
from PIL import Image
import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt


st.text("Text : This is Ali Ghanbari")

st.write("Write : This is Ali Ghanbari")

st.markdown("# This is Heading")

st.title("This is Title")

st.latex(r'''
...     a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
...     \sum_{k=0}^{n-1} ar^k =
...     a \left(\frac{1-r^{n}}{1-r}\right)
...     ''')

# -------------------------------------------------
df = pd.read_csv(
    r"F:\Projects\Machine Learning (maktabkhooneh)\18__ Create WebApp for ML\day.csv")
st.dataframe(df)
# st.table(df)

# -------------------------------------------------
photo = Image.open(
    r"F:\Projects\Machine Learning (maktabkhooneh)\18__ Create WebApp for ML\best_ai_companies_2025.jpg")
st.image(photo)

# -------------------------------------------------
st.error("Error Message")

st.warning("Warning message")

st.info('Info message')

st.success("Success messages")

# -------------------------------------------------
if st.button("Click here"):
    st.toast("🎉 You clicked the button!")
    st.balloons()

st.checkbox("Check1")
st.checkbox("Check2")
st.checkbox("Check3")
st.checkbox("Check4")

st.radio("RADIO", [1, 2, 3, 4, 5, 6])

st.slider("Slide me", 10, 100)
st.select_slider('Slide to select', options=[
                 'Ali', 'Ghanbari', 'Zahra', 'Ghalenave'])

st.selectbox('Select', [1, 2, 3])
st.multiselect('Multiple selection', [21, 85, 53])

st.file_uploader("Upload:.......")

st.text_input('Enter some text')
st.text_area('Text area')

st.color_picker("COLOR")

# -------------------------------------------------
my_bar = st.progress(0)

for percent_complete in range(100):
    time.sleep(0.05)
    my_bar.progress(percent_complete + 1)

with st.spinner("Loading..."):
    time.sleep(1)
with st.spinner("Processing..."):
    time.sleep(2)
with st.spinner("Generating output..."):
    time.sleep(1)

# -------------------------------------------------
# Title
st.title("📈 Math Plot Example")

# Data
x = np.linspace(0, 10, 500)
y = np.sin(x)  # sine function

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y, label='sin(x)')
ax.set_title("Sine Function")
ax.legend()

# Display in Streamlit
st.pyplot(fig)
