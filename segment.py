from os import TMP_MAX
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import json
import pickle
import numpy as np
 
import os


from os import TMP_MAX
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import json
import pickle
import numpy as np
from vncorenlp import VnCoreNLP

import re
#re encode
# name = "Xã hội"
# ls = json.load(open('data/data_'+name+'.txt','rb'))
# json.dump(ls,open('data/data_'+name+'.txt','w',encoding='utf-8'), ensure_ascii= False)


# print(ls[0]['message'])
# print(len(ls))
# for j in range(len(ls)):
#     if ('message' not in ls[j].keys()):
#         a += 1
#         print(j)
#         print(ls[j])
#     else:   docs[str(j)] = ls[j]['message']
# print(a)
# print(len(ls))



err = []

#stop_words = [x for x in open('stop_words.txt','r').read().split('\n')]
for filename in os.listdir('data'):
    filename = filename.replace('data_','').replace('.json','')
    
    if (filename != "Chính trị"): continue
    print(filename)


    f = open('data/data_'+filename+'.json','r',encoding='utf-8')

    #load pure corpus (not segmented)
    corpus = []
    for dic in json.load(f):
        if ('message' in dic.keys()):
            #remove css text
            doc = re.sub('<.*?>', '', dic['message'])
            if (doc != " " and doc != ""):
                corpus.append(doc)


    #segment corpus
    seg_corpus = []
    annotator = VnCoreNLP('./vncore/VnCoreNLP-master/VnCoreNLP-master/VnCoreNLP-1.1.1.jar', annotators="wseg", max_heap_size='-Xmx512m') 
    #for too logn document
    chunk_size = 15000  #word limit 
    # word_segmented_text = annotator.tokenize(corpus[3114][:90000]) 
    i  = -1
    for doc in corpus:
        i += 1
        if (i % 1000 == 0): break
        doc_seg = ""
        #split doc into chunks of word
        doc = doc.split(' ')
        doc_chunks =  [' '.join(doc[i:i+chunk_size]) for i in range(0,len(doc),chunk_size) ]
     
        for chunk in doc_chunks:
            try:

                word_segmented_text = annotator.tokenize(chunk)
                for sen in word_segmented_text:
                    doc_seg += ' ' + ' '.join(sen)

            except Exception:
                print('loi')
                print(i)
                err.append(chunk)
            
        doc_seg = doc_seg.replace('\xa0','')
        seg_corpus.append(doc_seg)
    break
    # json.dump(seg_corpus,open('store/'+filename+'/content.json','w',encoding='utf-8'),ensure_ascii=False)
    # json.dump(err,open('f.json','w',encoding='utf-8'),ensure_ascii=False)
    # tfvector = TfidfVectorizer(lowercase=True,stop_words=stop_words)
    # tfidfs = tfvector.fit_transform(seg_corpus).toarray().tolist()
    # pickle.dump(tfidfs,open('store/'+filename +'/tfidfs.pkl','wb'))
    

# for filename in os.listdir('store'):
    
 
#     print(filename)
#     f = open('data/data_'+filename+'/content.json','r',encoding='utf-8')

#     #load segmented corpus 
#     corpus = json.load(f)
#     i = 0
#     new_cor = []
#     #remove null text
#     print(len(corpus))
#     for doc in corpus:
    
      
#         doc = re.sub(r'\s+', ' ', doc) 
#         if (doc != " " and doc == ""):
#             new_cor.append(doc)
#         else:
#             i += 1
#     print('remove')
#     print(i)
#     print(len(corpus))

#     json.dump(new_cor,open('store/'+filename+'/content.json','w',encoding='utf-8'),ensure_ascii=False)
    
    