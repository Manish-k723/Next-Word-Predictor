import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="NextWordPredictor",
    page_icon="🚀"
)

with open('tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

model = load_model('model.h5')
# with open('model.pickle', 'rb') as file:
#     loaded_model = pickle.load(file)
with open("essay.txt", "r") as file:
    content = file.read()

with st.form("my_form"):
    st.title("Trained Content")
    st.write("Kept training data for fast performance as i am not using any special resource like GPU")
    st.markdown(content, unsafe_allow_html=True)
    text = st.text_input("Try something")
    how_many_words = st.slider("Required words", min_value = 1, max_value = 20, value = 3)
    submitted = st.form_submit_button("Predict")

while not submitted:
    pass
st.divider()
st.markdown("Your input text:\n"+ text)
st.divider()
for i in range(how_many_words):
  # tokenize
  token_text = tokenizer.texts_to_sequences([text])[0]
  # padding
  padded_text = pad_sequences([token_text], maxlen = 22, padding = 'pre')
  # predict
  pos = np.argmax(model.predict(padded_text))

  for word, index in tokenizer.word_index.items():
    if index==pos:
      # print(word)
      text = text+' '+word
st.markdown("Suggested output: \n"+ text)
st.markdown("Thanks you for using")
st.markdown("Developed by [Manish](https://www.linkedin.com/in/manish-kumar-244a55202/)", unsafe_allow_html=True)