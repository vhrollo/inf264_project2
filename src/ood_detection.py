import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from preprocessing import hog_features

class OODDetection:
    def __init__(self, best_svc, best_pca, X_train_pca, y_train, scaler, seed=42, n_components=17):
        self.best_svc = best_svc
        self.best_pca = best_pca
        self.seed = seed
        self.n_components = n_components
        self.scaler = scaler

        # Fit GaussianMixture model on the training PCA data
        self.gmm = GaussianMixture(n_components=self.n_components, 
                                   covariance_type='full', 
                                   random_state=self.seed, 
                                   init_params='kmeans')
        self.gmm.fit(X_train_pca)
        print(f"GaussianMixture model with {self.n_components} components fitted on training data.")

        self.class_means = []
        self.inv_cov_matrices = []

        for class_label in np.unique(y_train):
            X_class = X_train_pca[y_train == class_label]
            mean = np.mean(X_class, axis=0)
            cov = np.cov(X_class, rowvar=False)
            
            self.class_means.append(mean)
            self.inv_cov_matrices.append(np.linalg.inv(cov))        


    def fit(self, X):
        X_hog = hog_features(X)
        X_scaled = self.scaler.transform(X_hog)
        X_pca = self.best_pca.transform(X_scaled)
        return X_pca


    def predict_guassian(self, X_transformed):
        return self.best_svc.predict_proba(X_transformed)


    def detect_ood_gaussian(self, X_transformed, percentile=5.5):
        log_likelihood = self.gmm.score_samples(X_transformed)

        threshold = np.percentile(log_likelihood, percentile)
        ood_indices = log_likelihood < threshold

        print(f"Number of OOD samples: {ood_indices.sum()}")
        return ood_indices, ood_indices.sum()


    def mahalanobis_distance_classwise(self, x):
        
        distances = []
        for mean, inv_cov in zip(self.class_means, self.inv_cov_matrices):
            diff = x - mean
            dist = np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))
            distances.append(dist)
        return np.min(distances)


    def detect_ood_mahalanobis(self, X_transformed, percentile=93):
        distances = np.array([self.mahalanobis_distance_classwise(x) for x in X_transformed])

        threshold = np.percentile(distances, percentile)
        ood_mask = distances > threshold

        print(f"Number of OOD samples (Mahalanobis): {ood_mask.sum()}")
        return ood_mask, ood_mask.sum()


