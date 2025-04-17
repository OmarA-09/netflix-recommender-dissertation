import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import os

from src.recommenders.BaseNetflixRecommender import BaseNetflixRecommender

class TestBaseNetflixRecommender:
    
    @pytest.fixture
    def mock_integration_service(self):
        mock_service = MagicMock()
        mock_service.preprocess_dataset.side_effect = lambda df: df
        # Mock load_user_profile to return None
        mock_service.load_user_profile.return_value = None
        # Mock load_models to return None
        mock_service.load_models.return_value = None
        return mock_service
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D'],
            'description': ['Action movie', 'Comedy film', 'Drama about life', 'Documentary'],
            'director': ['Director 1', 'Director 2', 'Director 3', 'Director 4'],
            'cast': ['Actor 1', 'Actor 2', 'Actor 3', 'Actor 4'],
            'listed_in': ['Action', 'Comedy', 'Drama', 'Documentary'],
            'rating': ['TV-MA', 'PG-13', 'R', 'TV-PG'],
            'type': ['Movie', 'Movie', 'Movie', 'TV Show'],
            'release_year': [2020, 2019, 2021, 2018],
            'combined_features': [
                'Action movie Director 1 Actor 1 Action',
                'Comedy film Director 2 Actor 2 Comedy',
                'Drama about life Director 3 Actor 3 Drama',
                'Documentary Director 4 Actor 4 Documentary'
            ]
        })
    
    @pytest.fixture
    def base_recommender(self, mock_integration_service):
        with patch('src.recommenders.BaseNetflixRecommender.NetflixRecommenderIntegration', 
                  return_value=mock_integration_service):
            recommender = BaseNetflixRecommender()
            # Override _get_scored_items - not implemented in the base class
            recommender._get_scored_items = MagicMock(return_value=[
                (0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6)
            ])
            return recommender
    
    def test_init(self, mock_integration_service):
        with patch('src.recommenders.BaseNetflixRecommender.NetflixRecommenderIntegration', 
                  return_value=mock_integration_service):
            recommender = BaseNetflixRecommender()
            assert recommender.df is None
            assert recommender.tfidf_matrix is None
            assert recommender.user_liked_titles == set()
            assert recommender.user_disliked_titles == set()
            assert recommender.user_profile is None
            assert recommender.profile_path == "user_profile.pkl"
    
    def test_load_data(self, base_recommender, sample_df):
        # Mock pd.read_csv to return our the DataFrame
        with patch('pandas.read_csv', return_value=sample_df):
            recommender = base_recommender.load_data("dummy_path.csv")
            
            # Check lload_data returned self
            assert recommender == base_recommender
            assert recommender.df is not None
            assert len(recommender.df) == 4
            
            # Check integration service was used
            recommender.integration_service.preprocess_dataset.assert_called_once()
    
    def test_like_dislike_title(self, base_recommender, sample_df):
        base_recommender.df = sample_df
        base_recommender.tfidf_matrix = np.random.rand(4, 10)  # Random matrix for testing
        base_recommender.user_profile = np.zeros(10)
        
        # liking
        assert base_recommender.like_title("Movie A") == True
        assert "Movie A" in base_recommender.user_liked_titles
        
        assert base_recommender.like_title("Non-existent Movie") == False

        # disliking   
        assert base_recommender.dislike_title("Movie B") == True
        assert "Movie B" in base_recommender.user_disliked_titles
    
        assert base_recommender.dislike_title("Movie A") == True
        assert "Movie A" not in base_recommender.user_liked_titles
        assert "Movie A" in base_recommender.user_disliked_titles
        
        # chanhging title from disliked to liked
        assert base_recommender.like_title("Movie B") == True
        assert "Movie B" not in base_recommender.user_disliked_titles
        assert "Movie B" in base_recommender.user_liked_titles
    
    def test_search_titles(self, base_recommender, sample_df):
        base_recommender.df = sample_df
        
        results = base_recommender.search_titles("Movie")
        assert len(results) == 4
        
        results = base_recommender.search_titles("Movie", top_n=2)
        assert len(results) == 2
        
        results = base_recommender.search_titles("XYZ")
        assert len(results) == 0
    
    def test_get_recommendations(self, base_recommender, sample_df):
        base_recommender.df = sample_df
        base_recommender.tfidf_matrix = np.random.rand(4, 10)
        base_recommender.user_profile = np.random.rand(10)
        base_recommender.user_liked_titles = {"Movie A"}
        
        # Mock filter_recommendations to return some indices
        base_recommender.integration_service.filter_recommendations.return_value = [
            (1, 0.8), (2, 0.7), (3, 0.6)
        ]
        
        recommendations = base_recommender.get_recommendations(top_n=3)
        assert len(recommendations) == 3
        
        # Check recommendations contain expected columns
        assert 'title' in recommendations.columns
        assert 'similarity_score' in recommendations.columns
        assert 'rank' in recommendations.columns
        
        base_recommender.get_recommendations(rating_filter='PG-13')
        
        # Checkk integration service used
        base_recommender.integration_service.filter_recommendations.assert_called_with(
            base_recommender.df, base_recommender._get_scored_items(), 
            {"Movie A"}, 'PG-13', 10
        )
    
    def test_create_weighted_features(self, base_recommender, sample_df):
        weighted_features = base_recommender._create_weighted_features(sample_df)
        assert len(weighted_features) == 4
        
        assert 'TV-MA TV-MA TV-MA' in weighted_features[0]
        assert 'PG-13 PG-13 PG-13' in weighted_features[1]
    
    def test_reset_preferences(self, base_recommender):
        base_recommender.user_liked_titles = {"Movie A", "Movie B"}
        base_recommender.user_disliked_titles = {"Movie C"}
        base_recommender.tfidf_matrix = np.random.rand(4, 10)
        base_recommender.user_profile = np.ones(10)
        # reset
        base_recommender.reset_preferences()
        
        assert len(base_recommender.user_liked_titles) == 0
        assert len(base_recommender.user_disliked_titles) == 0
        assert np.all(base_recommender.user_profile == 0)

        # Checkk integration service used
        base_recommender.integration_service.reset_preferences.assert_called_once()
    
    def test_update_user_profile(self, base_recommender, sample_df):
        base_recommender.df = sample_df
        base_recommender.tfidf_matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0]
        ])
        base_recommender.user_profile = np.zeros(3)
        
        base_recommender.user_liked_titles = {"Movie A", "Movie D"}
        base_recommender.user_disliked_titles = {"Movie B"}
        
        # Update profile
        base_recommender.update_user_profile()
        assert not np.all(base_recommender.user_profile == 0)
        
        # Checkk integration service used
        base_recommender.integration_service.save_user_profile.assert_called_once()
        
    def test_preprocess(self, base_recommender, sample_df):
        base_recommender.df = sample_df
        
        # Create a spy save_models
        save_models_spy = MagicMock()
        base_recommender.integration_service.save_models = save_models_spy
        
        # Check no existing models loaded
        base_recommender.integration_service.load_models.return_value = None
        
        base_recommender.preprocess()
        
        # Check that save_models was called
        assert save_models_spy.called, "save_models was not called, suggesting models weren't created"
        assert hasattr(base_recommender, 'tfidf_matrix'), "tfidf_matrix wasn't created"
        assert base_recommender.user_profile is not None, "user_profile wasn't initialized"
            
    def test_get_user_preferences(self, base_recommender, sample_df):
        base_recommender.df = sample_df
        base_recommender.user_liked_titles = {"Movie A", "Movie B"}
        base_recommender.user_disliked_titles = {"Movie C"}
        
        # Mock analyse_user_preferences
        base_recommender.integration_service.analyse_user_preferences.return_value = {
            'favorite_genres': {'Action': 2, 'Comedy': 1},
            'genre_diversity': 2
        }
        
        # Get then check  user preferences
        liked_df, disliked_df = base_recommender.get_user_preferences()
        
        assert len(liked_df) == 2
        assert len(disliked_df) == 1
        assert liked_df.iloc[0]['title'] in ['Movie A', 'Movie B']
        assert disliked_df.iloc[0]['title'] == 'Movie C'
        
        # Checkk integration service used
        base_recommender.integration_service.analyse_user_preferences.assert_called_once()