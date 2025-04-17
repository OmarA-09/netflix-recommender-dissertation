import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from src.recommenders.CosineNetflixRecommender import CosineNetflixRecommender

class TestCosineNetflixRecommender:
    
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
    def cosine_recommender(self, mock_base_methods):
        recommender = CosineNetflixRecommender()
        
        recommender.user_profile = np.array([0.5, 0.3, 0.2])
        # each row diff user, varied similiarities
        recommender.tfidf_matrix = np.array([
            [0.9, 0.1, 0.0],  
            [0.1, 0.9, 0.0],  
            [0.0, 0.1, 0.9],  
            [0.5, 0.3, 0.2]   
        ])
        
        return recommender
    
    def test_init(self, mock_base_methods):
        recommender = CosineNetflixRecommender()
        mock_base_methods['init'].assert_called_once_with("data/user_profile_cosine.pkl")
        
        custom_path = "custom/path.pkl"
        recommender = CosineNetflixRecommender(profile_path=custom_path)
        mock_base_methods['init'].assert_called_with(custom_path)
    
    def test_get_scored_items(self, cosine_recommender):
        scored_items = cosine_recommender._get_scored_items()
        assert len(scored_items) == 4
        
        for i in range(len(scored_items) - 1):
            assert scored_items[i][1] >= scored_items[i+1][1]
        
        assert scored_items[0][0] == 3
        assert scored_items[0][1] == 1.0 
        
        assert scored_items[1][0] == 0