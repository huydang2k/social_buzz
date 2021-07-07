from vncorenlp import VnCoreNLP
import json
import re
import os
import pickle
import random
# for filename in os.listdir('store'):
    
    
#     filename = "Pháp luật"
#     f = open('store/'+filename+'/content.json','r')
#     tfidfs = pickle.load(open('store/'+filename+'/tfidfs.pkl','rb')).toarray()
    
#     #load pure corpus
#     corpus = json.load(f)
#     print('before')
#     print(len(corpus))
#     r = []
#     for i in range(len(tfidfs)):
#         check = False
#         for j in (tfidfs[i]):
#             if j!= 0: 
#                 check = True
#                 break
#         if check == False:
#             print(i)
#             print(corpus[i] + "a")
            
#             if (corpus[i].isspace()):
#                 print('space')
#             r.append(corpus[i])
#     r.append("a")
#     json.dump(r,open('err.json','w',encoding='utf-8'),ensure_ascii=False)  
#     break
    # new_res = []
    # for doc in corpus:
    #     if (doc != " " and doc != ""):
    #         new_res.append(doc)
    # print('after')
    # print(len(new_res))
    # json.dump(new_res,open('store/'+filename+'/content.json','w',encoding='utf-8'),ensure_ascii=False)
def remove_source():
    
    for catename in os.listdir('store'):
        catename = "Văn hóa"
        f = open('class_content/'+catename+'/content.json','r',encoding='utf-8')
        corpus = json.load(f)
        for doc in corpus:
            doc = re.sub('(function).*}',' ',doc)
            # doc = re.sub('Nguồn.*(https:].*[\/\/]*', '', doc)
            # doc = re.sub('Nguồn.*[https:].*[\.html]', '', doc)
            # doc = re.sub('Nguồn.*[https:].*[\.html]', '', doc)
            print(doc)

            break
        break

def gen_clean_class_data():
    for catename in os.listdir('store'):
        #catenme = "Chính trị","Đời sống", bla bl
        
        # if (catename != "Chính trị" and catename != "Quân sự" and catename != "Đối ngoại" and catename != "Giải trí"
        # and catename != "Kinh tế" and catename!= "Xã hội"): continue
        print(catename)
        cor = json.load(open('store/'+catename+'/content.json','r',encoding='utf-8'))
        new_cor = []
        
        for doc in cor:
            doc = re.sub('(http:\/\/).*?(\.htm)',' ',doc)
            doc = re.sub('(http:\/\/).*?(\.html)',' ',doc)
            doc = re.sub('(https:\/\/).*?(\/\/)',' ',doc)
            doc = re.sub('(https:\/\/).*?(\.htm)',' ',doc)
            doc = re.sub('(https:\/\/).*?(\.html)',' ',doc)
            doc = re.sub('(https:\/\/).*?(\.vn)',' ',doc)
            doc = re.sub('(https:\/\/).*?(\.net)',' ',doc)
            
            
            doc = re.sub('\n',' ',doc)
            doc = re.sub('\t',' ',doc)
            doc = re.sub('\r',' ',doc)

            
            new_cor.append(doc)
        #dump
        json.dump(new_cor,open('store/'+catename+'/content.json','w',encoding='utf-8'),ensure_ascii=False)
        #json.dump(new_cor,open('class_content/'+catename+'/content.json','w',encoding='utf-8'),ensure_ascii=False)
#gen_clean_class_data()

    # #load pure corpus (not segmented)
    # corpus = []
    # for dic in json.load(f):
    #     if ('message' in dic.keys()):
    #         #remove css text
    #         doc = re.sub('<.*?>', '', dic['message'])
    #         if (doc != " " and doc != ""):
    #             corpus.append(doc)


    # #segment corpus
    # seg_corpus = []
    # annotator = VnCoreNLP("/home/huydang/Save/doc_analysis/vncore/VnCoreNLP-master/VnCoreNLP-master/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx512m') 
    # #for too logn document
    # chunk_size = 15000  #word limit 
    # # word_segmented_text = annotator.tokenize(corpus[3114][:90000]) 
    # i  = -1
    # for doc in corpus:
    #     i += 1
    #     if (i % 1000 == 0): print(i)
    #     doc_seg = ""
    #     #split doc into chunks of word
    #     doc = doc.split(' ')
    #     doc_chunks =  [' '.join(doc[i:i+chunk_size]) for i in range(0,len(doc),chunk_size) ]
     
    #     for chunk in doc_chunks:
    #         try:

    #             word_segmented_text = annotator.tokenize(chunk)
    #             for sen in word_segmented_text:
    #                 doc_seg += ' ' + ' '.join(sen)

    #         except Exception:
    #             print('loi')
    #             print(i)
    #             err.append(chunk)
                
    #     doc_seg = doc_seg.replace('\xa0','')
    #     seg_corpus.append(doc_seg)
    # json.dump(seg_corpus,open('store/'+filename+'/content.json','w',encoding='utf-8'),ensure_ascii=False)
    # json.dump(err,open('f.json','w',encoding='utf-8'),ensure_ascii=False)
    
    # tfvector = TfidfVectorizer(lowercase=True,stop_words=stop_words)
    # tfidfs = tfvector.fit_transform(seg_corpus).toarray().tolist()
    # pickle.dump(tfidfs,open('store/'+filename +'/tfidfs.pkl','wb'))
def random_infer(number:int,cate:str):
    corpur1 = json.load(open('store/'+cate+'/content.json','r',encoding='utf-8'))
    corpur2 = json.load(open('class_content/'+cate+'/content.json','r',encoding='utf-8'))

    random.shuffle(corpur1)
    random.shuffle(corpur2)

    json.dump(corpur1[:number],open('infer1.json','w',encoding='utf-8'),ensure_ascii=False)
    json.dump(corpur2[:number],open('infer2.json','w',encoding='utf-8'),ensure_ascii=False)


random_infer(1000,'Chính trị')