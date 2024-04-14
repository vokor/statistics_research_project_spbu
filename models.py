from sklearn.cluster import KMeans

from info import df

models = []

class SimpleClusterModel:
    def __init__(self, features):
        self.features = features
        self.model = None

    def fit(self, n_clusters):
        data = df[self.features]
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.model.fit(data)

    def predict(self, data):
        return self.model.predict(data)
