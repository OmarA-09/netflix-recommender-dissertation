from sklearn.metrics.pairwise import cosine_similarity
from src.recommenders.BaseNetflixRecommender import BaseNetflixRecommender
from src.integration_service import NetflixRecommenderIntegration

class CosineNetflixRecommender(BaseNetflixRecommender):
    
    def __init__(self, profile_path="data/user_profile_cosine.pkl"):
        super().__init__(profile_path)
    
    def _get_scored_items(self):
        user_profile_matrix = self.user_profile.reshape(1, -1)
        cosine_similarities = cosine_similarity(user_profile_matrix, self.tfidf_matrix)[0]
        
        # creating list of (idx, score) tuples
        scored_items = list(enumerate(cosine_similarities))
        
        # Sort by highest similairty
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        return scored_items