import pandas as pd 
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

with open('models/pipeline.pickle', 'rb') as f:
    loaded_pipe = pickle.load(f)

def predict_pipeline(data):
    return predict(loaded_pipe,data)

def preprocess(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def predict(model, data):
    preprocessed_df = preprocess(data)
    predictions = model.fit_predict(preprocessed_df)
    predictions_list = predictions.tolist()
    return {'cluster':predictions_list}

if __name__=="__main__":
    df = pd.read_csv('https://storage.googleapis.com/the_public_bucket/wine-clustering.csv')
    new_data = df
    predictions = predict_pipeline(new_data)    
    print('la data pertenece a los siguientes cluster:',predictions)