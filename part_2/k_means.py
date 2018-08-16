
# coding: utf-8

# In[5]:



import json
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import string
import collections
from pprint import pprint
import numpy as np

import sys
import re, string
import copy
import random

def bagOfWords(string):
        words = str(string).lower().strip().split(' ')
        for word in words:
            word = word.rstrip().lstrip()
            if not re.match(r'^https?:\/\/.*[\r\n]*', word)             and not re.match('^@.*', word)             and not re.match('\s', word)             and word not in cachedStopWords             and word != 'rt'             and word != '':
                yield regex.sub('', word)



def jaccard_distance(query, document):
        intersection = set(query).intersection(set(document))
        union = set(query).union(set(document))
        return 1 - len(intersection)/len(union)



def initialize_matrix(df):    
        for tid1 in df['id']:
            for tid2 in df['id']:
                jacc_df[tid1][tid2] = jaccard_distance(set(bagOfWords(df[df['id']==tid1]['text'])), 
                                               set(bagOfWords(df[df['id']==tid2]['text'])))

        return jacc_df


def calcNewClusters(tweet_id, seed_id, jacc_df):
        cluster_df = pd.DataFrame(index = tweet_id, columns = seed_id)
        cluster_df.fillna(0, inplace=True)
        for t_id in tweet_id:
            min_dist = float("inf")
            nearest_seed = t_id
            for s_id in seed_id:
                d = jacc_df[s_id][t_id]
                if(d < min_dist):
                    min_dist = d
                    nearest_seed = s_id

            cluster_df.loc[t_id, nearest_seed] = 1

        #cluster_df.fillna(0)
        return cluster_df


def find_clusters(new_cluster_df, _id):
        return new_cluster_df.index[new_cluster_df[_id]==1].tolist()
    

def calcNewCentroids(cluster_df_new, seed_id, jacc_df):
        seed_id = cluster_df_new.columns.tolist()
        new_means = []
        for s_id in seed_id:
            new_centroid = s_id
            cluster_mem = []
            count = 0
            cluster_mem = find_clusters(cluster_df_new, s_id)

            avg_dist = 0
            min_dist = float("inf")
            for t_id1 in cluster_mem:
                dist = 0
                count = 0
                for t_id2 in cluster_mem:
                    dist = dist + jacc_df[t_id1][t_id2]

                avg_dist = dist/len(cluster_mem)

                if(avg_dist < min_dist):
                    min_dist = avg_dist
                    new_centroid = t_id1
                    #print(min_dist)
            new_means.append(new_centroid)        
        return new_means


# In[1098]:


def converge(tweet_id, seed_id):
        for i in range(0, 1000):
            new_cluster = pd.DataFrame()
            new_means = []
            new_cluster = calcNewClusters(tweet_id, seed_id, jacc_df)
            new_means = calcNewCentroids(new_cluster, seed_id, jacc_df)
            if(seed_id != new_means):
                seed_id = new_means
            else:
                break
        return new_means, new_cluster

def calculate_SSE(cluster_mem, mean):
        SSE = 0.0
        for mem in cluster_mem:
            SSE = SSE + jacc_df[mem][mean] * jacc_df[mem][mean]
        return SSE/len(cluster_mem)

def find_kmeans(K, tweet_id, seed_id, output_file):
        print('Clustering tweets... Please wait...')
        k_means, k_clusters = converge(tweet_id, seed_id)
        f=open(output_file, "w+")
        i = 1
        SSE = 0.0
        for mean_id in k_means:
            cluster_mem = []
            cluster_mem = find_clusters(k_clusters, mean_id)
            SSE = SSE + calculate_SSE(cluster_mem, mean_id)
            text = str(i) + "  " + ",".join([str(x) for x in cluster_mem]) 
            i = i + 1
            f.write(text)
            f.write('\n')

        print('SSE = ', SSE/K)

        print('K-means clustering done successfully... Open file to view results')
        return

if __name__ == "__main__":
        K = int(sys.argv[1])
        seed_file = sys.argv[2]
        tweet_file = sys.argv[3]
        output_file = sys.argv[4]
        f = open(seed_file)
        seed_id_from_file = [int(line.rstrip(',\n')) for line in f.readlines()]
        f.close()

        regex = re.compile('[%s]' % re.escape(string.punctuation))
        cachedStopWords = stopwords.words('english')

        tweets = {}
        tweet_text = []
        tweet_id = []
        with open(tweet_file, 'r') as f:
            for line in f:
                tweet = json.loads(line)
                tweet_id.append(tweet['id'])
                tweet_text.append(tweet['text'])

        d = {'id': tweet_id, 'text': tweet_text}
        df = pd.DataFrame(data=d)

        df.set_index('id')

        if(K > len(seed_id_from_file)):
            print('K cannot be greater than', len(seed_id_from_file) , 'Hence taking default K=25')
            K = 25

        seed_id = random.sample(seed_id_from_file, K)
        jacc_df = pd.DataFrame(index = tweet_id, columns = tweet_id)

        jacc_df.fillna(1)

        jacc_df = initialize_matrix(df) 

        find_kmeans(K, tweet_id, seed_id, output_file)

