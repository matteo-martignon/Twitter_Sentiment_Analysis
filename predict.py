import pandas as pd
import tensorflow as tf
import utils as u
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from warnings import filterwarnings
filterwarnings("ignore")

columns = ['polarity', 'id', 'date', 'query', 'user', 'text']
print('load data...')
df = pd.read_csv('data/training.1600000.processed.noemoticon.csv', encoding='latin-1', engine='python', header=None)
df.columns = columns
df = df[['polarity', 'text']]
print('data loaded.')
print('cleaning text...')
df['clean_text'] = u.clean(df['text'])
print('text cleaned.')

X = df['clean_text'].values
y = df['polarity'].values

print('start Tokenizer...')
tokenizer = Tokenizer(num_words=50_000, lower=True, split=' ')

tokenizer.fit_on_texts(X)

X_seq = tokenizer.texts_to_sequences(X)
print('end Tokenizer.')

len_ = []
for i in df.clean_text:
    len_.append(len(i.split(' ')))
max_len = max(len_)

X_seq_matrix = sequence.pad_sequences(X_seq, maxlen=max_len)

y_st = y/4

print('load model...')

model = tf.keras.models.load_model("lstm_1")
print('model loaded.')

print('evaluating...')
results = model.evaluate(X_seq_matrix, y_st)
print(results)
