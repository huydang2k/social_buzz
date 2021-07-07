import pickle
import json
import json
import pickle
import re
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import json
import pickle
import numpy as np
from vncorenlp import VnCoreNLP


ce = pickle.load(open('classifier/classifier.pkl','rb'))
label_map = json.load(open('label_map.json','r',encoding='utf-8'))
annotator = VnCoreNLP("./vncore/VnCoreNLP-master/VnCoreNLP-master/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx512m') 
tfvectoror = pickle.load(open('classifier/vectorizer.pkl','rb'))

def get_cate_name(i:int):
    return list(label_map.keys())[list(label_map.values()).index(i)]

app = FastAPI()
class MyReq(BaseModel):
    name:str
    cluter_no:int
@app.post("/show_clus")
def show_clus(myr: MyReq):
    clustor = pickle.load(open('store/'+myr.name+'/clusteror.pkl','rb'))
    f = open('store/'+myr.name+'/content.json','r',encoding='utf-8')
    print(clustor.num_clusters)
    corpus = json.load(f)
    res = []
    for i in range(len(clustor.labels_)):
        if (clustor.labels_[i] == myr.cluter_no):
            res.append(corpus[i])
    return res
@app.post('/classify/')
def pr(doc :str):
    print(doc)
    doc_seg = ''
    word_segmented_text = annotator.tokenize(doc)
    for sen in word_segmented_text:
        doc_seg += ' ' + ' '.join(sen)
    print(doc_seg)
    tfidf = tfvectoror.transform([doc_seg]).toarray()
    print(tfidf)
    label = ce.predict_proba(tfidf)
    print(ce.classes_)
    answer_id = np.argmax(np.array(label[0]))
    print(label)
    res = {}
    for i in range(15):
        res[ce.classes_[i]] = label[0][i]
    res['answer'] =ce.classes_[answer_id]
    return res