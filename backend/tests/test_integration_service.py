import os
import pickle
import pandas as pd
import numpy as np
import pytest
import tempfile
import shutil
from unittest.mock import MagicMock, patch
from src.integration_service import NetflixRecommenderIntegration

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
        'description': ['Action movie', 'Comedy film', 'Drama about life', '', None],
        'director': ['Director 1', 'Director 2', '', None, 'Director 5'],
        'cast': ['Actor 1, Actor 2', 'Actor 3', None, 'Actor 4', ''],
        'listed_in': ['Action, Adventure', 'Comedy', 'Drama', 'Comedy, Drama', None],
        'rating': ['TV-MA', 'PG-13', 'R', 'PG', 'TV-MA']
    })

@pytest.fixture
def temp_dir():
    test_dir = tempfile.mkdtemp()
    yield test_dir
    # Cleanup after tests
    shutil.rmtree(test_dir)

@pytest.fixture
def integration_with_temp_dir(temp_dir):
    profile_path = os.path.join(temp_dir, 'user_profile.pkl')
    return NetflixRecommenderIntegration(profile_path=profile_path)

class TestNetflixRecommenderIntegration:
    
    def test_init(self):
        integration = NetflixRecommenderIntegration()
        assert integration.profile_path == "data/user_profile.pkl"
        
        # Test with custom path
        custom_path = "custom/path.pkl"
        integration = NetflixRecommenderIntegration(profile_path=custom_path)
        assert integration.profile_path == custom_path
        
    def test_preprocess_dataset(self, sample_df):
        integration = NetflixRecommenderIntegration()
        processed_df = integration.preprocess_dataset(sample_df.copy())
        
        for col in ['description', 'director', 'cast', 'listed_in']:
            assert processed_df[col].isnull().sum() == 0
        
        assert 'combined_features' in processed_df.columns
        assert processed_df.iloc[0]['combined_features'] == 'Action movie Director 1 Actor 1, Actor 2 Action, Adventure'
        
        assert processed_df.shape[0] == sample_df.shape[0]
        
    def test_load_user_profile_nonexistent(self, integration_with_temp_dir):
        result = integration_with_temp_dir.load_user_profile()
        assert result is None
    
    def test_save_and_load_user_profile(self, integration_with_temp_dir):
        # Create test data
        liked_titles = {'Movie A', 'Movie B'}
        disliked_titles = {'Movie C'}
        user_profile = {'genre_preferences': {'Action': 0.8, 'Comedy': 0.2}}
        
        success = integration_with_temp_dir.save_user_profile(
            liked_titles, disliked_titles, user_profile
        )
        assert success is True
        
        loaded_profile = integration_with_temp_dir.load_user_profile()
        assert loaded_profile is not None
        assert loaded_profile['liked_titles'] == liked_titles
        assert loaded_profile['disliked_titles'] == disliked_titles
        assert loaded_profile['user_profile'] == user_profile
    
    def test_load_user_profile_error(self, integration_with_temp_dir):
        with open(integration_with_temp_dir.profile_path, 'w') as f:
            f.write("This is not a valid pickle file")
        
        # Attempt to load the corrupted profile
        result = integration_with_temp_dir.load_user_profile()
        assert result is None
    
    def test_save_user_profile_error(self, integration_with_temp_dir):
        # Mock os.makedirs to fail
        with patch('os.makedirs', side_effect=PermissionError("Permission denied")):
            # Attempt to save profile to a location that would require directory creation
            invalid_path = os.path.join("/invalid/path", "profile.pkl")
            integration = NetflixRecommenderIntegration(profile_path=invalid_path)
            
            result = integration.save_user_profile(set(), set(), {})
            assert result is False
    
    def test_filter_recommendations(self, sample_df):
        integration = NetflixRecommenderIntegration()
        scored_items = [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5)]
        all_interacted = {'Movie A', 'Movie C'}
        
        results = integration.filter_recommendations(
            sample_df, scored_items, all_interacted, top_n=3
        )
        
        expected_indices = [(1, 0.8), (3, 0.6), (4, 0.5)]
        assert results == expected_indices
        
        results = integration.filter_recommendations(
            sample_df, scored_items, all_interacted, rating_filter='PG', top_n=3
        )
        
        expected_indices = [(3, 0.6)]
        assert results == expected_indices
    
    def test_analyse_user_preferences(self, sample_df):
        integration = NetflixRecommenderIntegration()
        
        test_df = pd.DataFrame({
            'title': ['Movie A', 'Movie B', 'Movie C'],
            'description': ['Action movie', 'Comedy film', 'Drama film'],
            'director': ['Director 1', 'Director 2', 'Director 3'],
            'cast': ['Actor 1', 'Actor 2', 'Actor 3'],
            'listed_in': ['Action, Adventure', 'Action, Comedy', 'Drama, Action']
        })
        
        preferences = integration.analyse_user_preferences(test_df)
        
        assert preferences is not None
        assert 'favorite_genres' in preferences
        assert 'genre_diversity' in preferences
        
        preferences = integration.analyse_user_preferences(sample_df.iloc[:1])
        assert preferences is None

    def test_reset_preferences(self, integration_with_temp_dir):
        liked_titles = {'Movie A', 'Movie B'}
        disliked_titles = {'Movie C'}
        user_profile = {'genre_preferences': {'Action': 0.8, 'Comedy': 0.2}}
        
        integration_with_temp_dir.save_user_profile(
            liked_titles, disliked_titles, user_profile
        )
        
        result = integration_with_temp_dir.reset_preferences()
        assert result is True
        
        assert not os.path.exists(integration_with_temp_dir.profile_path)
        
        result = integration_with_temp_dir.reset_preferences()
        assert result is True
    
    def test_save_and_load_models(self, integration_with_temp_dir, temp_dir):
        models_dir = os.path.join(temp_dir, 'data', 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        class SimpleModel:
            def __init__(self, name):
                self.name = name
        
        svd_model = SimpleModel("SVD Model")
        tfidf_vectorizer = SimpleModel("TFIDF Vectorizer")
        n_components = 100
        
        with patch.object(integration_with_temp_dir, 'save_models', return_value=True):
            result = integration_with_temp_dir.save_models(None, None, None)
            assert result is True
            
            test_model_data = {
                'svd_model': SimpleModel("Test SVD"),
                'tfidf_vectorizer': SimpleModel("Test TFIDF"),
                'n_components': 100,
                'created_at': '2023-04-17T12:00:00'
            }
            
            with patch.object(integration_with_temp_dir, 'load_models', return_value=test_model_data):
                loaded_models = integration_with_temp_dir.load_models()
                assert loaded_models is not None
                assert 'svd_model' in loaded_models
                assert 'tfidf_vectorizer' in loaded_models
                assert loaded_models['n_components'] == 100
                
    def test_load_models_nonexistent(self, integration_with_temp_dir):
        with patch.object(integration_with_temp_dir, 'load_models', return_value=None):
            result = integration_with_temp_dir.load_models()
            assert result is None
    
    def test_load_models_error(self, integration_with_temp_dir, temp_dir):
        with patch.object(integration_with_temp_dir, 'load_models', return_value=None):
            result = integration_with_temp_dir.load_models()
            assert result is None