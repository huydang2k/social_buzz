from os import TMP_MAX
from sklearn import metrics
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
from scipy.sparse import find

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import random
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from itertools import product
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
from collections import Counter
# #gen data and label
def gen_data_and_label(bunch:int):
    corpus = []
    label = []
    stop_words = [x for x in open('stop_words.txt','r').read().split('\n')]
    for catename in os.listdir('store'):
        print(catename)
        tmp_cor = json.load(open('class_content/'+catename+'/content.json','r',encoding='utf-8'))
        # random.shuffle(tmp_cor)
        # l = len(tmp_cor[0:bunch])
        # if (l < bunch):
        #     add_docs = json.load(open('store/'+catename+'/content.json','r',encoding='utf-8'))[0:bunch-l]
        #     tmp_cor.extend(add_docs)
        corpus.extend(tmp_cor)
        label.extend([catename]*len(tmp_cor))
        # corpus.extend(tmp_cor)
        # label.extend([catename]*len(tmp_cor))

    print('Done collect')
    #shuffle
    c = list(zip(corpus,label))
    random.shuffle(c)
    corpus,label = zip(*c)

    #vectorize
    prob = 3.0/len(corpus)
    tfvectoror = TfidfVectorizer(lowercase=True,stop_words=stop_words,min_df=prob,max_df = 0.98)
    tfidfs = tfvectoror.fit_transform(corpus)
    print(tfidfs.shape[0])
    print(len(tfvectoror.vocabulary_))

    cor_dict = {}
    for i in range(len(corpus)):
        #cate = list(mapp.keys())[list(mapp.values()).index(label[i])]
        cor_dict[str(i)] = {"content": corpus[i],"cate":label[i]}

    #save
    pickle.dump(tfvectoror,open('classifier/vectorizer.pkl','wb'))
    json.dump(cor_dict,open('classifier/global_corpus_30k.json','w',encoding='utf-8'),ensure_ascii=False)    
    pickle.dump(tfidfs,open('classifier/global_tfidfs_30k.pkl','wb'))
    pickle.dump(label,open('classifier/global_label_30k.pkl','wb'))



def build_classifier():
    X = pickle.load(open('classifier/global_tfidfs_30k.pkl','rb')).toarray()
    y = np.array(pickle.load(open('classifier/global_label_30k.pkl','rb')))

    print(X.shape)
    print(y.shape)

    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #gnb = GaussianNB()

    #knn = KNeighborsClassifier(n_neighbors=17,metric = 'cosine',weights='distance')

    model = ensemble.ExtraTreesClassifier()
    #clf = RandomForestClassifier(n_estimators = 1000, max_depth=10, random_state=0)
    y_pred1 = model.fit(X_train, y_train).predict(X_train)
    print('train')

    print(classification_report(y_train,y_pred1))
    print('test')
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    print(model.n_estimators)
    #protocol = 4 for big dataa
    pickle.dump(model,open('classifier/classifier.pkl','wb'),protocol=4)
def test(bunch:int):

    vectoror = pickle.load(open('classifier/vectorizer.pkl','rb'))
    # corpus = []
    # label = []
    # for filename in os.listdir('data'):
    #     cate = filename.replace('data_','').replace('.json','')
    #     tmpd = json.load(open('data/' + filename,'r',encoding='utf-8'))[:bunch]
    #     tmp = []
    #     for x in tmpd:
    #         if ('message' in x.keys()):
    #             tmp.append(x['message'])
    #     corpus.extend(tmp)
    #     label.extend([cate]*len(tmp))
    corpus = []
    label = []
    for catename in os.listdir('store'):
        print(catename)
        
        tmp_cor = json.load(open('store/'+catename+'/content.json','r',encoding='utf-8'))
        random.shuffle(tmp_cor)
        # l = len(tmp_cor[0:bunch])
        # print(l)
        # if (l < bunch):
        #     add_docs = json.load(open('store/'+catename+'/content.json','r',encoding='utf-8'))[0:bunch-l]
        #     tmp_cor.extend(add_docs)
        #     print('add')
        corpus.extend(tmp_cor[0:bunch])
        label.extend([catename]*len(tmp_cor[0:bunch]))

    print('done collect')
    #shuffle
    c = list(zip(corpus,label))
    random.shuffle(c)
    corpus,label = zip(*c)

    tfidfs = vectoror.transform(corpus).toarray()
    print(tfidfs.shape[0])
    print(len(vectoror.vocabulary_))
    

    knn = pickle.load(open('classifier/classifier.pkl','rb'))
    
    y_pred = knn.predict(tfidfs)
   
    print(classification_report(label,y_pred))
def test_class(cate:str,bunch:int):
    tmp_cor = json.load(open('store/'+cate+'/content.json','r',encoding='utf-8'))
    
    tmp_cor = tmp_cor[:bunch]
    # label = [cate]*len(tmp_cor[0:bunch])
    vectoror = pickle.load(open('classifier/vectorizer.pkl','rb'))

    model = pickle.load(open('classifier/classifier.pkl','rb'))
    tfidfs = vectoror.transform(tmp_cor).toarray()
    y_pred = model.predict(tfidfs)
    #infer
    res = []
    for i in range(len(y_pred)):
        res.append({"content":tmp_cor[i],"Predicted_label":y_pred[i]})
    json.dump(res,open('output.json','w',encoding='utf-8'),ensure_ascii=False) 

    return Counter(y_pred)


# gen_data_and_label(bunch=2000)
# build_classifier()
test(bunch=1000)
# dic = {}
# # test_class("Pháp luật",1000)
# for cate in os.listdir('store'):
   
#     print(cate)
#     dic[cate] = test_class(cate,1000)
# json.dump(dic,open('test.json','w',encoding='utf-8'),ensure_ascii=False)