import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

class RecommenderEvaluator:
    
    def __init__(self, recommender=None):
        self.recommender = recommender
        self.age_rating_groups = {
            'kids': ['TV-Y', 'TV-Y7', 'TV-G', 'G'],
            'teens': ['TV-PG', 'PG', 'TV-14', 'PG-13'],
            'adults': ['TV-MA', 'R', 'NC-17']
        }
        # Rating values for numeric calculations
        self.rating_values = {
            'TV-Y': 1, 'TV-Y7': 2, 'TV-G': 3, 'G': 3, 
            'TV-PG': 4, 'PG': 4, 'TV-14': 5, 'PG-13': 5, 
            'TV-MA': 6, 'R': 6, 'NC-17': 7, 'NR': 5, 'UR': 5
        }
    
    def get_rating_group(self, rating):
        for group, ratings in self.age_rating_groups.items():
            if rating in ratings:
                return group
        return 'unknown'
    
    def evaluate_recommendations(self, recommendations_df, target_age_group):
        """
        Evaluate recommendations using ML metrics
        
        Parameters:
        - recommendations_df: DataFrame with recommendations (must include 'rating' column)
        - target_age_group: Target age group ('kids', 'teens', 'adults')
        
        Returns:
        - Dictionary with ML evaluation metrics
        """
        if recommendations_df.empty:
            return {"error": "No recommendations to evaluate"}
            
        actual_ratings = recommendations_df['rating'].tolist()
        actual_groups = [self.get_rating_group(r) for r in actual_ratings]
        
        predicted_groups = [target_age_group] * len(actual_groups)
        
        # Create binary classification for "appropriate" ratings
        target_ratings = self.age_rating_groups[target_age_group]
        is_appropriate_actual = [1 if r in target_ratings else 0 for r in actual_ratings]
        is_appropriate_target = [1] * len(actual_ratings)
        
        # For regression metrics, use rating values
        actual_values = [self.rating_values.get(r, 0) for r in actual_ratings]
        target_avg_value = np.mean([self.rating_values[r] for r in target_ratings])
        target_values = [target_avg_value] * len(actual_values)
        
        metrics = {
            "sample_size": len(recommendations_df),
            "accuracy": accuracy_score(is_appropriate_target, is_appropriate_actual),
            "precision": precision_score(is_appropriate_target, is_appropriate_actual, zero_division=0),
            "recall": recall_score(is_appropriate_target, is_appropriate_actual, zero_division=0),
            "f1_score": f1_score(is_appropriate_target, is_appropriate_actual, zero_division=0),
            "mse": mean_squared_error(target_values, actual_values),
            "mae": mean_absolute_error(target_values, actual_values),
            "rmse": np.sqrt(mean_squared_error(target_values, actual_values)),
            "rating_distribution": pd.Series(actual_ratings).value_counts().to_dict(),
            "group_distribution": pd.Series(actual_groups).value_counts().to_dict()
        }
        
        # Calculate percentage distribution
        for group in self.age_rating_groups:
            count = sum(1 for g in actual_groups if g == group)
            metrics[f"{group}_content_pct"] = count / len(actual_groups) * 100 if actual_groups else 0
            
        return metrics
    
    def compare_methods(self, test_titles, target_age_group, methods):
        """
        Compare different recommender methods using ML metrics
        
        Parameters:
        - test_titles: List of titles to use for creating test profile
        - target_age_group: Target age group for evaluation
        - methods: Dictionary of {method_name: recommender_instance}
        
        Returns:
        - DataFrame comparing different methods
        """
        results = {}
        
        for method_name, recommender in methods.items():
            print(f"Evaluating method: {method_name}")

            if hasattr(recommender, 'reset_preferences'):
                recommender.reset_preferences()
            
            for title in test_titles:
                if hasattr(recommender, 'like_title'):
                    try:
                        recommender.like_title(title)
                    except:
                        print(f"  Could not add title: {title}")
            
            if hasattr(recommender, 'get_recommendations'):
                recommendations = recommender.get_recommendations(top_n=20)
            else:
                raise ValueError(f"Method {method_name} has no get_recommendations function")
            
            if recommendations.empty:
                print(f"  No recommendations for {method_name}")
                continue
            
            results[method_name] = self.evaluate_recommendations(recommendations, target_age_group)
        
        comparison_data = []
        for method_name, metrics in results.items():
            row = {
                'Method': method_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'Kids Content %': metrics.get('kids_content_pct', 0),
                'Teens Content %': metrics.get('teens_content_pct', 0),
                'Adult Content %': metrics.get('adults_content_pct', 0),
                'Sample Size': metrics['sample_size']
            }
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)


if __name__ == "__main__":
    from src.recommenders.KNNNetflixRecommender import KNNNetflixRecommender
    
    recommender = KNNNetflixRecommender()
    recommender.load_data('data/netflix_titles.csv')
    recommender.preprocess()
    evaluator = RecommenderEvaluator()
    
    recommender.like_title("Stranger Things")
    recommender.like_title("Breaking Bad")
    
    recommendations = recommender.get_recommendations(top_n=20)
    
    kids_metrics = evaluator.evaluate_recommendations(recommendations, 'kids')
    teens_metrics = evaluator.evaluate_recommendations(recommendations, 'teens')
    adults_metrics = evaluator.evaluate_recommendations(recommendations, 'adults')
    
    print("\nKids target metrics:")
    print(f"Accuracy: {kids_metrics['accuracy']:.2f}")
    print(f"F1 Score: {kids_metrics['f1_score']:.2f}")
    print(f"RMSE: {kids_metrics['rmse']:.2f}")
    
    print("\nTeens target metrics:")
    print(f"Accuracy: {teens_metrics['accuracy']:.2f}")
    print(f"F1 Score: {teens_metrics['f1_score']:.2f}")
    print(f"RMSE: {teens_metrics['rmse']:.2f}")
    
    print("\nAdults target metrics:")
    print(f"Accuracy: {adults_metrics['accuracy']:.2f}")
    print(f"F1 Score: {adults_metrics['f1_score']:.2f}")
    print(f"RMSE: {adults_metrics['rmse']:.2f}")