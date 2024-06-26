import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier, IsolationForest

#import umap
from sklearn.manifold import TSNE

def dense_umap(df):
    dens_map = TSNE().fit_transform(df)
    return dens_map

def preprocess_df(df): 
    df = df.select_dtypes(["float32", "float64", "int32", "int64"])
    scaler = StandardScaler()
    impute = SimpleImputer()
    
    out = impute.fit_transform(df)
    out, outliers = drop_outliers(out)
    out = scaler.fit_transform(out)
    return out, outliers

def drop_outliers(df):
    od = IsolationForest(contamination=0.01)
    f_outliers = od.fit_predict(df)
    drop_o = np.nonzero(np.where(f_outliers==-1, 1, 0))[0]
    out = np.delete(df, drop_o, axis=0)
    #return outliers
    return out, drop_o

def cluster_df(df, n=5):
    clust = AgglomerativeClustering(n_clusters=n)
    y = clust.fit_predict(df)
    return y

def feature_importance(df, labels):
    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(df, labels)
    feat_import = rf.feature_importances_
    return np.argsort(feat_import)[::-1]

def extract_features(df, ret_labels=False, labels=None):
    """Extract important features from the dataframe. Features are 'important' if they are able to predict the labels. If no labels are provided, 
    the function will cluster the data and use the clusters as labels. If the labels are provided, the function will encode the labels and use them to predict the features.
    The function will return the column names of the important features and the labels if ret_labels is True. If ret_labels is False, the function will return the column names of the important features."""
    #handle the labels
    if labels is None:
        labels = cluster_df(pre_df)
        labels_feat = labels.copy()
    elif isinstance(labels, np.ndarray):
        if labels.dtype != np.int32:
            #encode the labels
            labels_feat = LabelEncoder().fit_transform(labels)
    pre_df, outliers = preprocess_df(df)
    if outliers is not None:
        labels_feat = np.delete(labels_feat, outliers)
    idx_feat = feature_importance(pre_df, labels_feat)
    col = df.columns.values

    if outliers is not None:
        assert labels.shape[0] == df.shape[0]
        #labels = np.insert(labels, outliers, np.full(outliers.shape[0], -1))

    if ret_labels:
        return col[idx_feat], labels
    else:
        return col[idx_feat]