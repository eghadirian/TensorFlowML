import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my car',
    'You love my dog!',
    'Do you think my dog is amazing?'
]
tokenizer = Tokenizer(num_words=100, oov_token='<oov>') # 100 most common words, out of vocabulary
tokenizer.fit_on_texts(sentences) # upper==>lower, takes care of !
word_index = tokenizer.word_index # makes a dictionary
print(word_index)
sequence = tokenizer.texts_to_sequences(sentences)
print(sequence)
padded = pad_sequences(sequence, padding='post', maxlen=5, truncating='post')
# makes sentences the same size by putting zero where there is no word
print(padded)
test_data = [
    'I really love my dog',
    'My dog loves my manatee'
]
test_seq = tokenizer.texts_to_sequences(test_data)
test_padded = pad_sequences(test_seq, maxlen=10)
print(test_padded)