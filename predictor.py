from keras import backend as K
from keras_text_summarization.library.seq2seq import Seq2SeqSummarizer
import pandas as pd
import numpy as np
import pickle
import json

documents = pickle.load(open('./doc.pkl', 'rb'))
text = [s[0].upper() + s[1:] for s in documents]
for i in range(len(text)):
    if text[i][-1] not in ['.', '!', '?']:
        text[i] += '.'
text = (' '.join(text)).replace(' .', '.')
model_dir_path = './seq2seq_models'
config = np.load(Seq2SeqSummarizer.get_config_file_path(model_dir_path=model_dir_path)).item()
summarizer = Seq2SeqSummarizer(config)
summarizer.load_weights(weight_file_path=Seq2SeqSummarizer.get_weight_file_path(model_dir_path=model_dir_path))
headline = summarizer.summarize(text)
final_sentence = [{"key": 0, "sentence": headline}]
with open('./static/json/seq2seq.json', 'w') as f:
    json.dump({ "seq2seq": final_sentence }, f)
