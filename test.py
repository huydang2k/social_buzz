import pickle
from sklearn.cluster import AgglomerativeClustering
import os
import json
import traceback  
import re
from vncorenlp import VnCoreNLP
import numpy as np
from sklearn.metrics import classification_report
# for catename in os.listdir('store'):
 
    
#     # if (catename != "Chính trị"): continue
    
#     print(catename)
#     #tfidfs = pickle.load(open('store/'+catename+'/tfidfs.pkl','rb'))
#     #cor = json.load(open('store/'+catename+'/content.json','r',encoding='utf-8'))
#     try:
#         clusteror = pickle.load(open('store/'+catename+'/clusteror.pkl','rb'))
#         print(clusteror.num_clus)
#         # tfidfs = tfidfs.toarray()    
#         # print(len(tfidfs))
#         # v = 0.9
#         # hyer_clusteror = AgglomerativeClustering(n_clusters = None,compute_full_tree = True, affinity = 'cosine', linkage = 'complete',distance_threshold = v)
    
#         # hyer_clusteror.fit(tfidfs)
#         # print(hyer_clusteror.n_clusters_)

#         # pickle.dump(hyer_clusteror,open('store/'+catename+'/clusteror.pkl','wb'))
#     except Exception:

#         print('loi')
#         traceback.print_exc()

def remove_source():
    
    for filename in os.listdir('store'):
        filename = "Chính trị"
        f = open('store/'+filename+'/content.json','r',encoding='utf-8')
        corpus = json.load(f)
    
        for doc in corpus:
            doc = corpus[3]
            doc = re.sub('[https:].*?[\/\/]', '', doc)
            doc = re.sub('(http:).*?(\.html)', '', doc)
            doc = re.sub('(https:).*?(\.html)', '', doc)
            doc = re.sub('{.*?}', '', doc) 
            doc = re.sub('{.', '', doc) 
            doc = re.sub('}.', '', doc) 
            # doc = re.sub('Nguồn.*[https:].*[\.html]', '', doc)
            print(doc)

            break
        break

ce = pickle.load(open('classifier/classifier.pkl','rb'))
label_map = json.load(open('label_map.json','r',encoding='utf-8'))
annotator = VnCoreNLP("./vncore/VnCoreNLP-master/VnCoreNLP-master/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx512m') 
tfvectoror = pickle.load(open('classifier/vectorizer.pkl','rb'))

doc = 'Lâu nay giáo dục có nhiệm vụ quảng bá sự ưu việt bằng cách tạo ra con số tốt nghiệp trung học phổ thông trên 95%, bằng những tấm huy chương mang về qua các kỳ thi quốc tế với học sinh trung học phổ thông, bằng tỷ lệ mù chữ chỉ vài ba phần trăm,… Nhưng đó không thể và không phải là nhiệm vụ đích thực của giáo dục. Nhiệm vụ đích thực của giáo dục là đào tạo ra những thế hệ người Việt “Có văn hóa” và “Có kỹ năng”.'
def pr(doc :str):
    print(doc)
    doc_seg = ''
    word_segmented_text = annotator.tokenize(doc)
    print(word_segmented_text)
    for sen in word_segmented_text:
        doc_seg += ' ' + ' '.join(sen)

    if (doc == doc_seg): print(True)
    tfidf = tfvectoror.transform([doc_seg]).toarray()

    label = ce.predict_proba(tfidf)
    print(ce.classes_)
    answer_id = np.argmax(np.array(label[0]))
    print(label)
    res = {}
    for i in range(15):
        res[ce.classes_[i]] = label[0][i]
    res['answer'] =ce.classes_[answer_id]
    return res

def test(doc:str):

    vectoror = pickle.load(open('classifier/vectorizer.pkl','rb'))

    corpus = [doc]
    label = ["Giáo dục"]
    
    print('done collect')
    #shuffle
    
   

    tfidfs = vectoror.transform(corpus).toarray()
    
    

    knn = pickle.load(open('classifier/classifier.pkl','rb'))
    
    y_pred = knn.predict_proba(tfidfs)
    print(knn.classes_)
    answer_id = np.argmax(np.array(y_pred[0]))
    print(label)

    print(knn.classes_[answer_id])
    print(y_pred)
    

test(doc)
print(pr(doc))