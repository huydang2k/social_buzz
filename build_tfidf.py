from os import TMP_MAX
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import json
import pickle
import numpy as np
 
import os
import re

from os import TMP_MAX
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import json
import pickle
import numpy as np


stop_words = [x for x in open('stop_words.txt','r').read().split('\n')]

for filename in os.listdir('store'):
    os.mkdir('store2/'+filename)
    
    print(filename)

    f = open('class_content/'+filename+'/content.json','r',encoding='utf-8')

    #load segmented corpus 
    corpus = json.load(f)
     
    print(len(corpus))
    
    prob = 3.0/len(corpus)
    tfvectoror = TfidfVectorizer(lowercase=True,stop_words=stop_words,min_df=prob,max_df = 0.9)
    #tfvectoror = pickle.load(open('classifier/vectorizer.pkl','rb'))
    tfidfs = tfvectoror.fit_transform(corpus)
    print(len(tfvectoror.vocabulary_))
    
    
    #json.dump(vocab_dic,open('vocab/vocab_'+filename+ '.json','w',encoding='utf-8'),ensure_ascii=False)
    open('store2/'+filename+'/tfidf.pkl','w')
    pickle.dump(tfidfs,open('store2'+ filename +'/tfidfs.pkl','wb'))
    
    