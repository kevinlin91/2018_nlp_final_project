from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
import gc
from nocache import nocache
from gensim import corpora, models
from gensim.summarization.summarizer import summarize
from keras_text_summarization.library.seq2seq import Seq2SeqSummarizer
import pandas as pd
import numpy as np
import pickle
import json
import cv2
import os
import glob
import math

def rm_files():
    base_dir = ['scene_detection', 'static/images', 'scene_detection_pickle']
    for base in base_dir:
        path = os.path.join('.',base,'*')
        files = glob.glob(path)
        for f in files:
            os.remove(f)

def scene_detection_parse(name):
    file_list = [name.split('.')[0]]
    for file_name in file_list:
        scene_path = './scene_detection/' + file_name + '.csv'
        scene_status = True
        parse_data = list()
        with open(scene_path,'r') as f:
            for i in range(3):
                f.readline()
            data = f.readlines()
            if len(data) < 2:
                scene_status=False
            if scene_status:
                parse_data = [ int(x.strip().split(',')[1]) for x in data if x!='\n']
            else:
                parse_data = [-1]
        pickle.dump(parse_data, open('./scene_detection_pickle/%s.pickle' % (name.split('.')[0]),'wb'))

def scene_detection(name):    
    shell = 'scenedetect --input ./user_video/%s -d content -t 15 --csv-output ./scene_detection/%s.csv' % (name, name.split('.')[0])
    print (shell)
    os.system(shell)
    scene_detection_parse(name)

def get_video_frame(name):
    
    frame_number = pickle.load(open('./scene_detection_pickle/%s.pickle' % name.split('.')[0], 'rb'))
    cap = cv2.VideoCapture('./user_video/%s' % name)
    success = True
    count = 0
    index = 0
    while success:
        success, image = cap.read()
        if(success == False):
            continue
        if count in frame_number:
            cv2.imwrite('./static/images/%s.jpg' % index,image)
            index +=1
        count +=1

def save_frame_txt():

    path = os.path.abspath(os.path.join(os.getcwd(), 'static/images', '*'))

    files = glob.glob(path)

    result = ','.join(files)

    with open('./im2txt/input_image.txt', 'w') as f:

        f.write(result)
        
app = Flask(__name__)

   
@app.route('/')
@nocache
def root():
   gc.collect()
   return render_template('index.html')

@app.route('/index')
@nocache
def index():
   return render_template('index.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
@nocache
def uploader():
   if request.method == 'POST':
      f = request.files['file']
      style = request.form['im2txt_type']
      name = f.filename
      f.save('./user_video/' + secure_filename(f.filename))
      with open('./file_name.txt','w') as f:
         f.write(name)
      with open('./style.txt', 'w') as f:
         f.write(style)
      return render_template('uploading_finish.html')

@app.route('/keyframe_extraction')
@nocache
def keyframe_extraction():
   return render_template('keyframe_extraction.html')

@app.route('/ajax/keyframe_extraction')
@nocache
def ajax_keyframe_extraction():
   rm_files()
   with open('./file_name.txt') as f:
      file_name = f.readlines()
   #keyframe extract
   scene_detection(file_name[0])
   get_video_frame(file_name[0])
   #im2txt
   with open('./style.txt') as f:
      style = f.readlines()[0]
   save_frame_txt()
   if style == 'desc':
      os.system("python ./im2txt/run_inference_desc.py ")
   elif style == 'story':
      os.system("python ./im2txt/run_inference_story.py ")
   #pickle2json
   im2txt_result = pickle.load(open('./im2txt_output/im2txt_result.pkl', 'rb'))
   im2txt_result = sorted(im2txt_result, key=lambda tup:int(tup[0].split('.')[0]))
   keyframe_list = list()
   count = 0
   for item in im2txt_result:
      keyframe_list.append( { "key": count+1, "image": item[0], "sentence":item[1]} )
      count +=1
   with open('./static/json/keyframe.json', 'w') as f:
      json.dump({ "keyframe": keyframe_list }, f)
   sentences = [x[1] for x in im2txt_result]
   pickle.dump(sentences, open('./doc.pkl', 'wb'))
   return redirect(url_for('description_result_redirect'))


@app.route('/description_result_redirect')
@nocache
def description_result_redirect():
   gc.collect()
   return render_template('description_result_redirect.html')

@app.route('/description_result')
@nocache
def description_result():
   return render_template('description_result.html')

@app.route('/summarization', methods = ['GET', 'POST'])
@nocache
def summarization():
   if request.method == 'POST':
      method = request.form['method']
      documents = pickle.load(open('./doc.pkl', 'rb'))
      length = len(documents)
      if method == 'lsa':
         stoplist = set('for a of the and to in'.split())
         texts = [[word for word in document.lower().split()] for document in documents]
         dictionary = corpora.Dictionary(texts)
         corpus = [dictionary.doc2bow(text) for text in texts]
         tfidf = models.TfidfModel(corpus)
         corpus_tfidf = tfidf[corpus]
         num = int(length * 0.2)
         if num < 1:
             num = 1
         lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num)
         corpus_lsi = lsi[corpus_tfidf]
         final_sentence = list()
         total = list()
         for i in range(num):
            proba_all = list()
            try:
               for item in corpus_lsi:
                   proba_all.append(item[i][1])
            except:
               pass
            topic_sentence = documents[proba_all.index(max(proba_all))]
            total.append(topic_sentence)
         total = set(total)
         for i, s in enumerate(total):
            final_sentence.append({"key": i, "sentence": s})
         with open('./static/json/lsa.json', 'w') as f:
            json.dump({ "lsa": final_sentence }, f)
         return redirect(url_for('lsa_result_redirect'))
      elif method == 'textrank':
         text = [s[0].upper() + s[1:] for s in documents]
         for i in range(len(text)):
            if text[i][-1] not in ['.', '!', '?']:
               text[i] += '.'
         text = (' '.join(text)).replace(' .', '.')
         final_sentence = set(summarize(text, ratio=0.2, split=True))
         final_sentence = [{"key": i, "sentence": s} for i, s in enumerate(final_sentence)]
         with open('./static/json/textrank.json', 'w') as f:
            json.dump({ "TextRank": final_sentence }, f)
         return redirect(url_for('textrank_result_redirect'))
      if method == 'seq2seq':
         os.system("python ./predictor.py ")
         return redirect(url_for('seq2seq_result_redirect'))

@app.route('/lsa_result_redirect')
@nocache
def lsa_result_redirect():
   gc.collect()
   return render_template('lsa_result_redirect.html')

@app.route('/lsa_result')
@nocache
def lsa_result():
   return render_template('lsa_result.html')

@app.route('/textrank_result_redirect')
@nocache
def textrank_result_redirect():
   gc.collect()
   return render_template('textrank_result_redirect.html')

@app.route('/textrank_result')
@nocache
def textrank_result():
   return render_template('textrank_result.html')

@app.route('/seq2seq_result_redirect')
@nocache
def seq2seq_result_redirect():
   gc.collect()
   return render_template('seq2seq_result_redirect.html')

@app.route('/seq2seq_result')
@nocache
def seq2seq_result():
   return render_template('seq2seq_result.html')


@app.after_request
@nocache
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"

    r.headers["Pragma"] = "no-cache"

    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'

    return r
		
if __name__ == '__main__':
   app.run(debug = True, host='0.0.0.0')
