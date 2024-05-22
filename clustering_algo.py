from sentence_transformers import SentenceTransformer, util
import torch as T
import pandas as pd
import pickle
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_csv", "../cv_seg/kinetics_test.csv", "csv corresponding to the dataset with label column containing all the unique classnames")
flags.DEFINE_string("dataset_csv_column", "label", "name of the column containing all the unique classnames")
flags.DEFINE_string("surrogate_pickle", "../imagenet.pkl", "surrogate dataset mapper from class name to class id pickle file location")
flags.DEFINE_string("output", "class_mapper.pkl", "store class mapper pickle file output file")
    

class Cluster:
    
    def __init__(self, k = 10 ):
        self.encoder = SentenceTransformer('all-mpnet-base-v2')
        self.k = k
        
    def sim(self, vecA, vecB ):
        # ToDo: Check cosine sim between given arrays of Nx1 and Mx1 and return NxM
        self.sim_table = util.cos_sim(vecA, vecB) 

    def cluster_update_find_nearest_class(self, vecB):
        # ToDo: Finds the nearest classes which lies in the same
        new_cluster = {}
        for clusthead, clustmem in self.clusters.items():
            newclusthead = self.sim_table[clustmem,:].sum(axis=0).argmax().item()
            new_cluster[newclusthead] = clustmem
        
        self.clusters = new_cluster
        # print(T.tensor(list(self.clusters.keys())), type(self.clusters.keys()))
        self.cluster_heads = T.tensor(list(self.clusters.keys()))

    def cluster_update_evaluate_classes(self, vecA):
        
        clustidx = self.sim_table[:,self.cluster_heads].argmax(axis=-1)
        self.clusters = {}
        for clust in self.cluster_heads:
            self.clusters[clust.item()] = []

        self.loss = 0.0

        for i, (clust, vec) in enumerate(zip(clustidx, vecA)):
            self.clusters[self.cluster_heads[clust.item()].item()].append(i)
            self.loss += self.sim_table[i, self.cluster_heads[clust].item()]

        return self.loss



    def clustering(self, classA, classB, iters=10):
        # ToDo: nodified kmeans for fixed cluster heads with dist as 1-cosine ( bad idea as it brings the space totally down to a circle of dist 1)
        # ClassA names ClassB names A mapped to -> B of topk

        vecA = self.encoder.encode(classA, convert_to_tensor=True)
        vecB = self.encoder.encode(classB, convert_to_tensor=True)
        self.sim(vecA, vecB)
        print(self.sim_table)
        ridx = T.randperm(len(classB))[:self.k]

        self.cluster_heads = ridx

        for it in range(iters):
            loss = self.cluster_update_evaluate_classes(vecA)
            print(it+1, loss.item())
            self.cluster_update_find_nearest_class(vecB)
 
        return self.clusters
             

def main(argv):
    
    df = pd.read_csv(FLAGS.dataset_csv)
    A = list(set(df[FLAGS.dataset_csv_column]))
    with open(FLAGS.surrogate_pickle, 'rb') as f:
        db = pickle.load(f)
    B = [i for _, i in db.items()]

    clt = Cluster(k=20)
    clst = clt.clustering(A,B)

    clst_map = {}
    for k, v in clst.items():
        for vv in v:
            clst_map[A[vv]] = B[k]

    with open(FLAGS.output, 'wb') as f:
        pickle.dump(clst_map, f)

if __name__ == "__main__":
    app.run(main)