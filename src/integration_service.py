import os
import pickle
import pandas as pd
import logging

class NetflixRecommenderIntegration:
    def __init__(self, profile_path="user_profile.pkl"):
        """
        Initialize integration service with configurable profile path
        """
        self.profile_path = profile_path
        logging.basicConfig(level=logging.INFO, 
                             format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def preprocess_dataset(self, df):
        """
        Comprehensive data preprocessing
        """
        self.logger.info("Starting dataset preprocessing")
        
        # Handle missing values
        preprocessing_cols = ['description', 'director', 'cast', 'listed_in']
        for col in preprocessing_cols:
            df[col] = df[col].fillna('')
        
        # Create combined features for content-based filtering
        df['combined_features'] = (
            df['description'] + ' ' + 
            df['director'] + ' ' + 
            df['cast'] + ' ' + 
            df['listed_in']
        )
        
        self.logger.info(f"Preprocessing complete. Dataset shape: {df.shape}")
        return df

    def load_user_profile(self):
        """
        Load user profile with enhanced error handling
        """
        try:
            if os.path.exists(self.profile_path) and os.path.getsize(self.profile_path) > 0:
                with open(self.profile_path, 'rb') as f:
                    profile_data = pickle.load(f)
                
                self.logger.info("User profile loaded successfully")
                return {
                    'liked_titles': profile_data.get('liked', set()),
                    'disliked_titles': profile_data.get('disliked', set()),
                    'user_profile': profile_data.get('profile', None)
                }
            else:
                self.logger.warning("No existing profile found")
                return None
        except Exception as e:
            self.logger.error(f"Profile loading error: {e}")
            return None

    def save_user_profile(self, liked_titles, disliked_titles, user_profile):
        """
        Save user profile with logging
        """
        profile_data = {
            'liked': liked_titles,
            'disliked': disliked_titles,
            'profile': user_profile
        }
        
        try:
            with open(self.profile_path, 'wb') as f:
                pickle.dump(profile_data, f)
            self.logger.info("User profile saved successfully")
            return True
        except Exception as e:
            self.logger.error(f"Profile saving error: {e}")
            return False

    def filter_recommendations(self, df, scored_items, all_interacted, rating_filter=None, top_n=10):
        """
        Advanced recommendation filtering
        """
        recommendation_indices = []
        
        for idx, score in scored_items:
            title = df.iloc[idx]['title']
            
            # Skip titles the user has already interacted with
            if title in all_interacted:
                continue
            
            # Apply rating filter if specified
            if rating_filter and df.iloc[idx]['rating'] != rating_filter:
                continue
                
            recommendation_indices.append((idx, score))
            if len(recommendation_indices) >= top_n:
                break
        
        return recommendation_indices

    def analyze_user_preferences(self, liked_df):
        """
        Comprehensive user preference analysis
        """
        if len(liked_df) >= 3:
            genres = []
            for _, row in liked_df.iterrows():
                if isinstance(row['listed_in'], str):
                    genres.extend([g.strip() for g in row['listed_in'].split(',')])
            
            genre_counts = pd.Series(genres).value_counts()
            favorite_genres = genre_counts[genre_counts > 1].head(5)
            
            return {
                'favorite_genres': favorite_genres.to_dict(),
                'genre_diversity': len(favorite_genres)
            }
        return None

    def reset_preferences(self):
        """
        Reset user profile
        """
        try:
            if os.path.exists(self.profile_path):
                os.remove(self.profile_path)
            self.logger.info("User profile reset successfully")
            return True
        except Exception as e:
            self.logger.error(f"Profile reset error: {e}")
            return False
    
    def save_models(self, svd_model, tfidf_vectorizer, n_components):
        """Save SVD and TF-IDF models for persistence"""
        models_dir = os.path.join(os.path.dirname(self.profile_path), 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        models_data = {
            'svd_model': svd_model,
            'tfidf_vectorizer': tfidf_vectorizer,
            'n_components': n_components,
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        models_path = os.path.join(models_dir, 'recommender_models.pkl')
        
        try:
            with open(models_path, 'wb') as f:
                pickle.dump(models_data, f)
            logging.info(f"Saved models to {models_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving models: {str(e)}")
            return False

    def load_models(self):
        """Load SVD and TF-IDF models if available"""
        models_dir = os.path.join(os.path.dirname(self.profile_path), 'models')
        models_path = os.path.join(models_dir, 'recommender_models.pkl')
        
        try:
            if os.path.exists(models_path):
                with open(models_path, 'rb') as f:
                    models_data = pickle.load(f)
                    logging.info(f"Loaded models with {models_data.get('n_components')} SVD components")
                    return models_data
            return None
        except Exception as e:
            logging.warning(f"Could not load models, will rebuild: {str(e)}")
            return None