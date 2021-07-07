import pickle
from sklearn.cluster import AgglomerativeClustering
import os
import json
import traceback  
for catename in os.listdir('store'):
 
    
    if (catename == "Chính trị"): continue
    
    print(catename)
    tfidfs = pickle.load(open('store/'+catename+'/tfidfs.pkl','rb'))
    #cor = json.load(open('store/'+catename+'/content.json','r',encoding='utf-8'))
    try:
        tfidfs = tfidfs.toarray()    
        print(len(tfidfs))
        v = 0.95
        hyer_clusteror = AgglomerativeClustering(n_clusters = None,compute_full_tree = True, affinity = 'cosine', linkage = 'complete',distance_threshold = v)
    
        hyer_clusteror.fit(tfidfs)
        print(hyer_clusteror.n_clusters_)

        pickle.dump(hyer_clusteror,open('store/'+catename+'/clusteror.pkl','wb'))
    except Exception:
        print()
        print('loi')
        traceback.print_exc()


    
    
    



