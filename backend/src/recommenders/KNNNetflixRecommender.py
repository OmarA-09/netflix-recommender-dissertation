from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from src.recommenders.BaseNetflixRecommender import BaseNetflixRecommender
from src.integration_service import NetflixRecommenderIntegration

class KNNNetflixRecommender(BaseNetflixRecommender):
    def __init__(self, profile_path="data/user_profile_knn.pkl", n_neighbours=10):
        super().__init__(profile_path)
        self.n_neighbours = n_neighbours
        self.knn_model = None
        
    def _additional_preprocessing(self):
        self.logger.info(f"Fitting KNN model with {self.n_neighbours} neighbours...")
        self.knn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbours + 1, self.tfidf_matrix.shape[0]),
            algorithm='auto',
            metric='euclidean'
        )
        self.knn_model.fit(self.tfidf_matrix)
    
    def _get_scored_items(self):
        user_profile_matrix = self.user_profile.reshape(1, -1)
        
        distances, indices = self.knn_model.kneighbors(
            user_profile_matrix, 
            n_neighbors=min(50, self.tfidf_matrix.shape[0])
        )
        
        max_distance = distances.max()
        if max_distance > 0:
            similarities = 1 - (distances[0] / max_distance)
        else:
            similarities = np.ones_like(distances[0])
        
        scored_items = list(zip(indices[0], similarities))
        
        return scored_items
    
    def get_similar_titles(self, title, n=10):
        matches = self.df[self.df['title'] == title]
        if matches.empty:
            self.logger.warning(f"Title not found: {title}")
            return pd.DataFrame()
        
        title_idx = matches.index[0]
        title_features = self.tfidf_matrix[title_idx].reshape(1, -1)
        
        # Find nearest neighbours to the title
        distances, indices = self.knn_model.kneighbors(title_features, n_neighbors=n+1)
        
        # Skip the first result its the title itself
        similar_indices = indices[0][1:]
        similarity_scores = 1 - (distances[0][1:] / distances[0][1:].max())
        
        similar_titles = self.df.iloc[similar_indices][
            ['title', 'type', 'rating', 'release_year', 'description']
        ].copy()
        
        similar_titles['similarity_score'] = similarity_scores
        similar_titles['rank'] = range(1, len(similar_titles) + 1)
        
        return similar_titles