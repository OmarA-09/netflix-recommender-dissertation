import pytest
import pandas as pd
from src.recommenders.KNNNetflixRecommender import KNNNetflixRecommender
from src.evaluator import RecommenderEvaluator

class TestApp:
    
    @pytest.fixture
    def knn_recommender(self):
        recommender = KNNNetflixRecommender()
        recommender.load_data('data/netflix_titles.csv')
        recommender.preprocess()
        return recommender
    
    @pytest.fixture
    def evaluator(self, knn_recommender):
        return RecommenderEvaluator(knn_recommender)
    
    def test_recommender_initialisation(self, knn_recommender):
        assert knn_recommender is not None
        assert knn_recommender.df is not None
        assert knn_recommender.tfidf_matrix is not None
        assert knn_recommender.knn_model is not None
    
    def test_like_title(self, knn_recommender):
        title = "Stranger Things"  # A title that should exist in the dataset
        success = knn_recommender.like_title(title)
        assert success is True
        assert title in knn_recommender.user_liked_titles
    
    def test_get_recommendations(self, knn_recommender):
        knn_recommender.like_title("Stranger Things")
        recommendations = knn_recommender.get_recommendations(top_n=5)
        
        assert recommendations is not None
        assert not recommendations.empty
        assert len(recommendations) <= 5
        assert 'title' in recommendations.columns
        assert 'rating' in recommendations.columns
    
    def test_evaluator(self, knn_recommender, evaluator):
        knn_recommender.like_title("Stranger Things")
        knn_recommender.like_title("Breaking Bad")
        
        recommendations = knn_recommender.get_recommendations(top_n=10)
        
        if recommendations.empty:
            pytest.skip("No recommendations generated")
        
        kids_metrics = evaluator.evaluate_recommendations(recommendations, 'kids')
        
        assert 'accuracy' in kids_metrics
        assert 'precision' in kids_metrics
        assert 'f1_score' in kids_metrics
        assert 'rmse' in kids_metrics
        
        assert 0 <= kids_metrics['accuracy'] <= 1
        assert 0 <= kids_metrics['precision'] <= 1
        assert 0 <= kids_metrics['f1_score'] <= 1
        assert kids_metrics['rmse'] >= 0