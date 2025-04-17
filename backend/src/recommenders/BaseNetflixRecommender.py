import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
import os
from src.integration_service import NetflixRecommenderIntegration
import logging

class BaseNetflixRecommender:
    """Base class for Netflix recommender systems"""
    
    def __init__(self, profile_path="user_profile.pkl"):
        """Initialize the Netflix recommender system"""
        self.integration_service = NetflixRecommenderIntegration(profile_path)
        
        self.df = None
        self.tfidf_matrix = None
        self.user_liked_titles = set()
        self.user_disliked_titles = set()
        self.user_profile = None
        self.profile_path = profile_path
        self.models_path = profile_path.replace('.pkl', '_models.pkl')
        self.tfidf_vectorizer = None  
        self.svd_model = None         
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, filepath):
        """Load the Netflix dataset"""
        try:
            self.df = pd.read_csv(filepath)
            self.df = self.integration_service.preprocess_dataset(self.df)
            
            # Ensure 'combined_features' exists
            if 'combined_features' not in self.df.columns:
                self.logger.error("Column 'combined_features' missing from preprocessed data")
                raise ValueError("Required column 'combined_features' not found")
                
            self.logger.info(f"Loaded dataset with {self.df.shape[0]} titles")
            
            # Load existing user profile (removed the call to _load_models)
            profile_data = self.integration_service.load_user_profile()
            if profile_data:
                self.user_liked_titles = profile_data['liked_titles']
                self.user_disliked_titles = profile_data['disliked_titles']
                self.user_profile = profile_data['user_profile']
            
            return self
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def preprocess(self):
        """Create the TF-IDF matrix with SVD dimensionality reduction"""
        try:
            self.logger.info("Building TF-IDF matrix with age rating emphasis...")
            
            # First, try to load existing models
            models_data = self.integration_service.load_models()
            if models_data:
                self.svd_model = models_data.get('svd_model')
                self.tfidf_vectorizer = models_data.get('tfidf_vectorizer')
                self.n_components = models_data.get('n_components')
            
            # Create weighted features
            self.df['weighted_features'] = self._create_weighted_features(self.df)
            
            # Apply TF-IDF and SVD
            try:
                if self.tfidf_vectorizer is None:
                    # Build new models
                    self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
                    tfidf_matrix_raw = self.tfidf_vectorizer.fit_transform(self.df['weighted_features'])
                    
                    self.n_components = min(300, tfidf_matrix_raw.shape[1] - 1)
                    self.svd_model = TruncatedSVD(n_components=self.n_components, random_state=42)
                    self.tfidf_matrix = self.svd_model.fit_transform(tfidf_matrix_raw)
                    
                    # Save the new models
                    self.integration_service.save_models(
                        self.svd_model,
                        self.tfidf_vectorizer,
                        self.n_components
                    )
                else:
                    # Use existing models
                    tfidf_matrix_raw = self.tfidf_vectorizer.transform(self.df['weighted_features'])
                    self.tfidf_matrix = self.svd_model.transform(tfidf_matrix_raw)
            except Exception as e:
                self.logger.error(f"Error in TF-IDF/SVD processing: {str(e)}")
                # Fall back to simple TF-IDF without SVD if necessary
                self.logger.warning("Falling back to simple TF-IDF without SVD")
                self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['weighted_features']).toarray()
                self.svd_model = None
            
            # Initialize user profile if not already loaded
            if self.user_profile is None:
                self.user_profile = np.zeros(self.tfidf_matrix.shape[1])
            elif self.user_profile.shape[0] != self.tfidf_matrix.shape[1]:
                # Rebuild profile if dimensions don't match
                self.logger.warning("Rebuilding user profile due to dimension mismatch")
                self.user_profile = np.zeros(self.tfidf_matrix.shape[1])
                # If we have likes/dislikes, update the profile
                if self.user_liked_titles or self.user_disliked_titles:
                    self.update_user_profile()
            
            # Call model-specific preprocessing with error handling
            try:
                self._additional_preprocessing()
            except Exception as e:
                self.logger.error(f"Error in additional preprocessing: {str(e)}")
                self.logger.warning("Continuing with base preprocessing only")
            
            return self
        except Exception as e:
            self.logger.error(f"Critical error during preprocessing: {str(e)}")
            raise
    
    def _create_weighted_features(self, df):
        """Create weighted features with emphasis on age ratings"""
        # Define rating importance weights
        rating_emphasis = 3  # How much more we want to emphasize ratings
        
        # List of different rating types
        ratings = ['TV-Y', 'TV-Y7', 'TV-G', 'G', 'TV-PG', 'PG', 'TV-14', 'PG-13', 'TV-MA', 'R', 'NC-17']
        
        # Create weighted features
        weighted_features = []
        
        for _, row in df.iterrows():
            try:
                # Get the base features
                base_features = row['combined_features']
                
                # Check if this row's rating is in our list
                current_rating = row['rating'] if 'rating' in row and pd.notna(row['rating']) else ''
                
                # Add the rating multiple times to increase its weight in the feature vector
                rating_boosted = ' '.join([current_rating] * rating_emphasis) if current_rating in ratings else current_rating
                
                # Combine original features with emphasized rating
                weighted_feature = f"{base_features} {rating_boosted}".strip()
                weighted_features.append(weighted_feature)
            except Exception as e:
                self.logger.warning(f"Error creating weighted features for row: {str(e)}")
                weighted_features.append(row.get('combined_features', ''))
        
        return weighted_features

    def _additional_preprocessing(self):
        """Placeholder for model-specific preprocessing steps
        Derived classes should override this method to add their specific preprocessing steps.
        
        Note: After this method is called, self.tfidf_matrix contains the SVD-reduced feature matrix,
        not the original TF-IDF matrix. Make sure your derived class is aware of this.
        """
        pass
    
    # Rest of the class remains the same...
    
    def update_user_profile(self):
        """Update user profile based on liked and disliked titles"""
        if not self.user_liked_titles and not self.user_disliked_titles:
            self.logger.warning("No liked or disliked titles to build profile from")
            return
        
        # Get indices of liked and disliked titles
        liked_indices = []
        for title in self.user_liked_titles:
            matches = self.df[self.df['title'] == title]
            if not matches.empty:
                liked_indices.append(matches.index[0])
        
        disliked_indices = []
        for title in self.user_disliked_titles:
            matches = self.df[self.df['title'] == title]
            if not matches.empty:
                disliked_indices.append(matches.index[0])
        
        # Update user profile based on liked and disliked content in the SVD space
        if liked_indices:
            liked_profile = np.mean(self.tfidf_matrix[liked_indices], axis=0)
        else:
            liked_profile = np.zeros(self.tfidf_matrix.shape[1])
            
        if disliked_indices:
            disliked_profile = np.mean(self.tfidf_matrix[disliked_indices], axis=0)
            # Subtract disliked profile with lower weight (0.5)
            self.user_profile = liked_profile - 0.5 * disliked_profile
        else:
            self.user_profile = liked_profile
            
        # Normalize profile
        profile_norm = np.linalg.norm(self.user_profile)
        if profile_norm > 0:
            self.user_profile = self.user_profile / profile_norm
            
        # Save profile using integration service
        self.integration_service.save_user_profile(
            self.user_liked_titles, 
            self.user_disliked_titles, 
            self.user_profile
        )
    
    def like_title(self, title):
        """User likes a title"""
        matches = self.df[self.df['title'] == title]
        if matches.empty:
            self.logger.warning(f"Title not found: {title}")
            return False
        
        title_info = matches.iloc[0]
        self.logger.info(f"You liked: {title_info['title']} ({title_info['type']}, {title_info['rating']})")
        
        # Add to liked set and remove from disliked if present
        self.user_liked_titles.add(title)
        self.user_disliked_titles.discard(title)
        
        # Update user profile
        self.update_user_profile()
        return True
    
    def dislike_title(self, title):
        """User dislikes a title"""
        matches = self.df[self.df['title'] == title]
        if matches.empty:
            self.logger.warning(f"Title not found: {title}")
            return False
        
        title_info = matches.iloc[0]
        self.logger.info(f"You disliked: {title_info['title']} ({title_info['type']}, {title_info['rating']})")
        
        # Add to disliked set and remove from liked if present
        self.user_disliked_titles.add(title)
        self.user_liked_titles.discard(title)
        
        # Update user profile
        self.update_user_profile()
        return True
    
    def search_titles(self, query, top_n=5):
        """Search for titles in the dataset"""
        # Simple case-insensitive search in title
        matches = self.df[self.df['title'].str.lower().str.contains(query.lower())]
        
        if matches.empty:
            print(f"No titles found matching: {query}")
            return pd.DataFrame()
        
        return matches.head(top_n)[['title', 'type', 'rating', 'release_year']]
    
    def get_recommendations(self, top_n=10, rating_filter=None):
        """Get recommendations based on user profile"""
        if self.user_profile is None or len(self.user_liked_titles) == 0:
            self.logger.warning("Not enough preference data")
            return pd.DataFrame()
        
        # This method must be implemented by derived classes
        scored_items = self._get_scored_items()
        
        # Use integration service to filter recommendations
        all_interacted = self.user_liked_titles.union(self.user_disliked_titles)
        recommendation_indices = self.integration_service.filter_recommendations(
            self.df, scored_items, all_interacted, rating_filter, top_n
        )
        
        if not recommendation_indices:
            self.logger.warning("No suitable recommendations found")
            return pd.DataFrame()
        
        # Format recommendations as DataFrame
        rec_indices = [idx for idx, _ in recommendation_indices]
        rec_scores = [score for _, score in recommendation_indices]
        
        recommendations = self.df.iloc[rec_indices][
            ['title', 'type', 'rating', 'release_year', 'description']
        ].copy()
        
        recommendations['similarity_score'] = rec_scores
        recommendations['rank'] = range(1, len(recommendations) + 1)
        
        return recommendations
    
    def _get_scored_items(self):
        """Calculate similarity scores between user profile and items"""
        # This method must be implemented by derived classes
        raise NotImplementedError("Subclasses must implement _get_scored_items()")
    

    def get_user_preferences(self):
        """Get information about user's current preferences"""
        liked_df = self.df[self.df['title'].isin(self.user_liked_titles)]
        disliked_df = self.df[self.df['title'].isin(self.user_disliked_titles)]
        
        self.logger.info(f"You have liked {len(liked_df)} titles and disliked {len(disliked_df)} titles.")
        
        # Display liked titles
        if not liked_df.empty:
            self.logger.info("\n--- Titles You've Liked ---")
            for i, (_, row) in enumerate(liked_df.iterrows(), 1):
                self.logger.info(f"{i}. {row['title']} ({row['type']}, {row['rating']})")
        
        # Display disliked titles
        if not disliked_df.empty:
            self.logger.info("\n--- Titles You've Disliked ---")
            for i, (_, row) in enumerate(disliked_df.iterrows(), 1):
                self.logger.info(f"{i}. {row['title']} ({row['type']}, {row['rating']})")
        
        # Use integration service for preference analysis
        preference_analysis = self.integration_service.analyse_user_preferences(liked_df)
        
        if preference_analysis:
            self.logger.info("\n--- Your Favorite Genres ---")
            for genre, count in preference_analysis['favorite_genres'].items():
                self.logger.info(f"• {genre} ({count} titles)")
        
        return liked_df, disliked_df
        
    def reset_preferences(self):
        """Reset all user preferences"""
        self.user_liked_titles = set()
        self.user_disliked_titles = set()
        self.user_profile = np.zeros(self.tfidf_matrix.shape[1])
        
        # Use integration service to reset profile
        self.integration_service.reset_preferences()
        
        self.logger.info("User preferences have been reset.")