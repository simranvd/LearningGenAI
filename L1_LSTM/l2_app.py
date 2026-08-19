import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('LSTM.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(line,model,tokenizer,max_seq_len):
  token_list = tokenizer.texts_to_sequences([line])[0]
  if(len(token_list)>=max_seq_len):
    token_list = token_list[-(max_seq_len-1):]
  token_list = pad_sequences([token_list],max_seq_len-1, padding='pre')
  predicted = model.predict(token_list, verbose=0)
  predicted_index = np.argmax(predicted,axis=1)
  for word,index in tokenizer.word_index.items():
    if index==predicted_index:
      return word
  return None

st.title("Shakespeare Word Prediction App")

line = st.text_input("Enter a line of text:", "Be or not to be")
if st.button("Predict Next Word"):
    max_seq_len = len(tokenizer.word_index) + 1
    next_word = predict_next_word(line, model, tokenizer, max_seq_len)
    st.write(f"Next word: {next_word}")