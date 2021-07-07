import pickle
import json
import os
import re
import traceback
dic = json.load(open('infor.json','r',encoding='utf-8'))

#gen number of doc
# num_doc_dic = {}
# for filename in os.listdir('store'):
#     print(filename)
#     corpus = json.load(open('store/'+filename+'/content.json','r',encoding='utf-8'))
    
#     num_doc_dic[filename] = len(corpus)

# dic["number"] = num_doc_dic



#reencode and delete escape sequence
# for filename in os.listdir('data'):
#     print(filename)
    
#     corpus = json.load(open('data/'+filename,'rb'))

#     for ind,doc_dic in enumerate(corpus):
#         for k,v in doc_dic.items():
#             if (k == 'categories'): continue
#             corpus[ind][k] = v.replace('\n',' ').replace('\r',' ').replace('\t',' ')
#             corpus[ind][k] = re.sub(r'\s+', ' ', corpus[ind][k])
#     json.dump(corpus,open('data/'+filename,'w',encoding='utf-8'), ensure_ascii= False)
#re encode
# name = "Xã hội"
# ls = json.load(open('data/data_'+name+'.txt','rb'))
# json.dump(ls,open('data/data_'+name+'.txt','w',encoding='utf-8'), ensure_ascii= False)


#vocab size for each cate
# vocab = {}

# for filename in os.listdir('vocab'):
#     a = json.load(open('vocab/' + filename,'r',encoding='utf-8'))
#     name = filename.replace('vocab_','').replace('.json','')
#     vocab[name] = a['size']
    
# dic ['vocab'] = vocab 


#number of clus per cate
# num_clus = {}
# for catename in os.listdir('store'):
#     try:
#         print(catename)
#         clustor = pickle.load(open('store/' + catename+'/clusteror.pkl','rb'))
#         num_clus[catename] = clustor.n_clusters_
#     except Exception:
#         traceback.print_exc()
# dic ['num_of_clus'] = num_clus 

#gen number of doc
num_doc_dic = {}
for filename in os.listdir('store'):
    print(filename)
    corpus = json.load(open('class_content/'+filename+'/content.json','r',encoding='utf-8'))
    
    num_doc_dic[filename] = len(corpus)

dic["number_of_docs_for_classification"] = num_doc_dic

    

json.dump(dic,open('infor.json','w',encoding='utf-8'),ensure_ascii=False)