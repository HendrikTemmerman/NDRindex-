from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

# Define the number of clusters
n_clusters = 3


def calculated_ARI(data,true_labels):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    predicted_labels = kmeans.labels_
    ARIScore = adjusted_rand_score(predicted_labels, true_labels)
    return ARIScore
