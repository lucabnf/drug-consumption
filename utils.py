# Standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import mat73
import utils

# outlier removal libraries
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
# standardization libraries
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# sampling libraries
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import KMeansSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN
# model libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
# utilities libraries
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

def remove_outliers_DBSCAN(df,eps,min_samples):
    outlier_detection = DBSCAN(eps = eps, min_samples = min_samples)
    clusters = outlier_detection.fit_predict(df.values.reshape(-1,1))
    data = pd.DataFrame()
    data['cluster'] = clusters
    return data['cluster']

def evaluate_model(model, model_params, X_train, y_train, scaler=None, outlier_model=None, outlier_model_params={}, sampling_model=None, sampling_model_params={}):
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
    results = []
    
    X_train_numpy = X_train.to_numpy()
    
    # OUTLIER REMOVAL
    if outlier_model != None and outlier_model != "DBSCAN":
        iso = outlier_model(**outlier_model_params)
        yhat = iso.fit_predict(X_train_numpy)

        # select all rows that are not outliers
        mask = yhat != -1
        X_train_cleaned, y_train_cleaned = X_train_numpy[mask, :], y_train[mask]
    elif outlier_model == "DBSCAN":
        
        clusters = np.zeros(X_train_numpy.shape[0])
        for feature in X_train.columns:
            clust=remove_outliers_DBSCAN((X_train[feature]),0.3,5)
            clust[clust != -1] = 0
            clusters += clust


        mask = clusters >= 0
        X_train_cleaned, y_train_cleaned = X_train_numpy[mask, :], y_train[mask]
    else:
        X_train_cleaned, y_train_cleaned = X_train_numpy, y_train
        
    # STANDARDIZATION
    if scaler != None: 
        scaler = scaler()
        scaler.fit(X_train_cleaned)
        X_train_scaled = scaler.transform(X_train_cleaned)
    else: 
        X_train_scaled = X_train_cleaned
    
    for train_fold_index, val_fold_index in cv.split(X_train_scaled, y_train_cleaned):
        X_train_fold, y_train_fold = X_train_scaled[train_fold_index], y_train_cleaned[train_fold_index]
        X_val_fold, y_val_fold = X_train_scaled[val_fold_index], y_train_cleaned[val_fold_index]
        
        # SAMPLING
        if sampling_model != None:
            X_train_fold_sampled, y_train_fold_sampled = sampling_model(**sampling_model_params).fit_resample(X_train_fold, y_train_fold)
        else:
            X_train_fold_sampled, y_train_fold_sampled = X_train_fold, y_train_fold
          
        # TRAINING
        model_trained = model(**model_params).fit(X_train_fold_sampled, y_train_fold_sampled)
        result = metrics.f1_score(y_val_fold, model_trained.predict(X_val_fold), average='macro')
        
        results.append(result)
        
    return np.array(results)