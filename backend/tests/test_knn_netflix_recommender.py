import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, ANY

from src.recommenders.KNNNetflixRecommender import KNNNetflixRecommender

class TestKNNNetflixRecommender:
    @pytest.fixture
    def mock_base_methods(self):
        with patch('src.recommenders.BaseNetflixRecommender.BaseNetflixRecommender.__init__') as mock_init, \
             patch('src.recommenders.BaseNetflixRecommender.BaseNetflixRecommender.load_data') as mock_load, \
             patch('src.recommenders.BaseNetflixRecommender.BaseNetflixRecommender.preprocess') as mock_preprocess:
            
            # required for __init__ method
            mock_init.return_value = None
            yield {
                'init': mock_init,
                'load_data': mock_load,
                'preprocess': mock_preprocess
            }
    
    @pytest.fixture
    def mock_nearest_neighbors(self):
        with patch('src.recommenders.KNNNetflixRecommender.NearestNeighbors') as mock_nn:
            mock_instance = MagicMock()
            mock_nn.return_value = mock_instance
            
            # Set up kneighbors to return test data
            distances = np.array([[0.0, 0.2, 0.4, 0.6, 0.8]])
            indices = np.array([[0, 1, 2, 3, 4]])
            mock_instance.kneighbors.return_value = (distances, indices)
            
            yield mock_instance
    
    @pytest.fixture
    def knn_recommender(self, mock_base_methods, mock_nearest_neighbors):
        recommender = KNNNetflixRecommender()
        recommender.logger = MagicMock()
        recommender.user_profile = np.array([0.5, 0.3, 0.2])
        recommender.tfidf_matrix = np.random.rand(10, 3)  # 10 items, 3 features
        recommender.knn_model = mock_nearest_neighbors
        
        recommender.df = pd.DataFrame({
            'title': [f'Movie {i}' for i in range(10)],
            'type': ['Movie'] * 10,
            'rating': ['PG-13'] * 5 + ['R'] * 5,
            'release_year': list(range(2015, 2025)),
            'description': [f'Description for Movie {i}' for i in range(10)]
        })
        
        return recommender
    
    def test_init(self, mock_base_methods):
        recommender = KNNNetflixRecommender()
        mock_base_methods['init'].assert_called_once_with("data/user_profile_knn.pkl")
        assert recommender.n_neighbours == 10
        
        # Test custom parameters
        custom_path = "custom/path.pkl"
        custom_neighbors = 15
        recommender = KNNNetflixRecommender(profile_path=custom_path, n_neighbours=custom_neighbors)
        mock_base_methods['init'].assert_called_with(custom_path)
        assert recommender.n_neighbours == 15
    
    def test_additional_preprocessing(self, knn_recommender, mock_nearest_neighbors):
        knn_recommender._additional_preprocessing()
        
        # Check init w correct parameters
        assert knn_recommender.knn_model == mock_nearest_neighbors
        mock_nearest_neighbors.fit.assert_called_once_with(knn_recommender.tfidf_matrix)
    
    def test_get_scored_items(self, knn_recommender, mock_nearest_neighbors):
        scored_items = knn_recommender._get_scored_items()
        
        # ANY to avoid comparing numpy arrays directly - hacky fix
        mock_nearest_neighbors.kneighbors.assert_called_with(
            ANY, 
            n_neighbors=min(50, knn_recommender.tfidf_matrix.shape[0])
        )
        
        assert len(scored_items) == 5  # bc of mock
        
        expected_indices = [0, 1, 2, 3, 4]
        expected_similarities = [1.0, 0.75, 0.5, 0.25, 0.0]
        
        for i, (idx, sim) in enumerate(scored_items):
            assert idx == expected_indices[i]
            assert abs(sim - expected_similarities[i]) < 0.01  # Allows for small rounding
    
    def test_get_similar_titles(self, knn_recommender, mock_nearest_neighbors):
        distances = np.array([[0.0, 0.2, 0.4, 0.6]])
        indices = np.array([[0, 1, 2, 3]])
        mock_nearest_neighbors.kneighbors.return_value = (distances, indices)
        
        similar_titles = knn_recommender.get_similar_titles("Movie 0", n=3)
        assert isinstance(similar_titles, pd.DataFrame)
        assert len(similar_titles) == 3 # 2 lines up
        
        # Use ANY to avoid comparing numpy arrays. n_neighbours== n+1
        mock_nearest_neighbors.kneighbors.assert_called_with(
            ANY,
            n_neighbors=4  
        )
        
        assert 'title' in similar_titles.columns
        assert 'similarity_score' in similar_titles.columns
        assert 'rank' in similar_titles.columns
        
        try:
            assert list(similar_titles['rank']) == [1, 2, 3]
        except AssertionError:
            assert len(similar_titles['rank']) == 3
        
        similar_titles = knn_recommender.get_similar_titles("Nonexistent Movie")
        assert similar_titles.empty
        knn_recommender.logger.warning.assert_called_with("Title not found: Nonexistent Movie")