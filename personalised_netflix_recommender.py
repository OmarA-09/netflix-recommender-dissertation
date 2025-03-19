import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class PersonalizedNetflixRecommender:
    def __init__(self):
        """Initialize the personalized Netflix recommender system"""
        self.df = None
        self.tfidf_matrix = None
        self.user_liked_titles = set()
        self.user_disliked_titles = set()
        self.user_profile = None
        self.profile_path = "user_profile.pkl"
        
    def load_data(self, filepath):
        """Load the Netflix dataset"""
        self.df = pd.read_csv(filepath)
        print(f"Loaded dataset with {self.df.shape[0]} titles")
        
        # Handle missing values
        self.df['description'] = self.df['description'].fillna('')
        self.df['director'] = self.df['director'].fillna('')
        self.df['cast'] = self.df['cast'].fillna('')
        self.df['listed_in'] = self.df['listed_in'].fillna('')
        
        # Create a combined feature for content-based filtering
        self.df['combined_features'] = (
            self.df['description'] + ' ' + 
            self.df['director'] + ' ' + 
            self.df['cast'] + ' ' + 
            self.df['listed_in']
        )
        
        # Load existing user profile if available
        self._load_user_profile()
        
        return self
    
    def preprocess(self):
        """Create the TF-IDF matrix for content similarity"""
        print("Building content similarity matrix...")
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.df['combined_features'])
        print(f"Created TF-IDF matrix with shape: {self.tfidf_matrix.shape}")
        
        # Initialize user profile if not already loaded
        if self.user_profile is None:
            self.user_profile = np.zeros(self.tfidf_matrix.shape[1])
            
        return self
    
    def _load_user_profile(self):
        """Load user's profile from file if it exists"""
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, 'rb') as f:
                    profile_data = pickle.load(f)
                    self.user_liked_titles = profile_data.get('liked', set())
                    self.user_disliked_titles = profile_data.get('disliked', set())
                    self.user_profile = profile_data.get('profile', None)
                print(f"Loaded user profile with {len(self.user_liked_titles)} liked and {len(self.user_disliked_titles)} disliked titles")
            except Exception as e:
                print(f"Error loading user profile: {e}")
                self.user_liked_titles = set()
                self.user_disliked_titles = set()
                self.user_profile = None
    
    def _save_user_profile(self):
        """Save user's profile to file"""
        profile_data = {
            'liked': self.user_liked_titles,
            'disliked': self.user_disliked_titles,
            'profile': self.user_profile
        }
        try:
            with open(self.profile_path, 'wb') as f:
                pickle.dump(profile_data, f)
            print("User profile saved successfully")
        except Exception as e:
            print(f"Error saving user profile: {e}")
    
    def update_user_profile(self):
        """Update user profile based on liked and disliked titles"""
        if not self.user_liked_titles and not self.user_disliked_titles:
            print("No liked or disliked titles to build profile from")
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
        
        # Update user profile based on liked and disliked content
        if liked_indices:
            liked_profile = np.mean(self.tfidf_matrix[liked_indices].toarray(), axis=0)
        else:
            liked_profile = np.zeros(self.tfidf_matrix.shape[1])
            
        if disliked_indices:
            disliked_profile = np.mean(self.tfidf_matrix[disliked_indices].toarray(), axis=0)
            # Subtract disliked profile with lower weight (0.5)
            self.user_profile = liked_profile - 0.5 * disliked_profile
        else:
            self.user_profile = liked_profile
            
        # Normalize profile
        profile_norm = np.linalg.norm(self.user_profile)
        if profile_norm > 0:
            self.user_profile = self.user_profile / profile_norm
            
        self._save_user_profile()
        print("User profile updated based on preferences")
    
    def like_title(self, title):
        """User likes a title"""
        matches = self.df[self.df['title'] == title]
        if matches.empty:
            print(f"Title not found: {title}")
            return False
        
        title_info = matches.iloc[0]
        print(f"You liked: {title_info['title']} ({title_info['type']}, {title_info['rating']})")
        
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
            print(f"Title not found: {title}")
            return False
        
        title_info = matches.iloc[0]
        print(f"You disliked: {title_info['title']} ({title_info['type']}, {title_info['rating']})")
        
        # Add to disliked set and remove from liked if present
        self.user_disliked_titles.add(title)
        self.user_liked_titles.discard(title)
        
        # Update user profile
        self.update_user_profile()
        return True
    
    def get_recommendations(self, top_n=10, rating_filter=None):
        """Get personalized recommendations based on user profile"""
        if self.user_profile is None or len(self.user_liked_titles) == 0:
            print("Not enough preference data. Please like some titles first!")
            return pd.DataFrame()
        
        # Calculate similarity between user profile and all items
        user_profile_matrix = self.user_profile.reshape(1, -1)
        cosine_similarities = cosine_similarity(user_profile_matrix, self.tfidf_matrix)[0]
        
        # Create a list of (index, score) tuples
        scored_items = list(enumerate(cosine_similarities))
        
        # Sort by similarity score (highest first)
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out titles the user has already interacted with
        all_interacted = self.user_liked_titles.union(self.user_disliked_titles)
        recommendation_indices = []
        
        for idx, score in scored_items:
            title = self.df.iloc[idx]['title']
            
            # Skip titles the user has already interacted with
            if title in all_interacted:
                continue
            
            # Apply rating filter if specified
            if rating_filter and self.df.iloc[idx]['rating'] != rating_filter:
                continue
                
            recommendation_indices.append((idx, score))
            if len(recommendation_indices) >= top_n:
                break
        
        if not recommendation_indices:
            print("No suitable recommendations found with the current filters")
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
    
    def search_titles(self, query, top_n=5):
        """Search for titles in the dataset"""
        # Simple case-insensitive search in title
        matches = self.df[self.df['title'].str.lower().str.contains(query.lower())]
        
        if matches.empty:
            print(f"No titles found matching: {query}")
            return pd.DataFrame()
        
        return matches.head(top_n)[['title', 'type', 'rating', 'release_year']]
    
    def get_user_preferences(self):
        """Get information about user's current preferences"""
        liked_df = self.df[self.df['title'].isin(self.user_liked_titles)]
        disliked_df = self.df[self.df['title'].isin(self.user_disliked_titles)]
        
        print(f"You have liked {len(liked_df)} titles and disliked {len(disliked_df)} titles.")
        
        if not liked_df.empty:
            print("\n--- Titles you've liked ---")
            for _, row in liked_df.iterrows():
                print(f"• {row['title']} ({row['type']}, {row['rating']})")
        
        if not disliked_df.empty:
            print("\n--- Titles you've disliked ---")
            for _, row in disliked_df.iterrows():
                print(f"• {row['title']} ({row['type']}, {row['rating']})")
        
        # Get preferred genres if enough data
        if len(liked_df) >= 3:
            genres = []
            for _, row in liked_df.iterrows():
                if isinstance(row['listed_in'], str):
                    genres.extend([g.strip() for g in row['listed_in'].split(',')])
            
            genre_counts = pd.Series(genres).value_counts()
            print("\n--- Your favorite genres ---")
            for genre, count in genre_counts.head(5).items():
                if count > 1:
                    print(f"• {genre} ({count} titles)")
        
        return liked_df, disliked_df
    
    def reset_preferences(self):
        """Reset all user preferences"""
        self.user_liked_titles = set()
        self.user_disliked_titles = set()
        self.user_profile = np.zeros(self.tfidf_matrix.shape[1])
        self._save_user_profile()
        print("User preferences have been reset.")

# Example usage
if __name__ == "__main__":
    # Initialize the recommender
    recommender = PersonalizedNetflixRecommender()
    
    # Load data
    recommender.load_data('netflix_titles.csv')
    
    # Preprocess the data
    recommender.preprocess()
    
    # Example interactions
    recommender.like_title("Stranger Things")
    recommender.like_title("Breaking Bad")
    recommender.dislike_title("The Crown")
    
    # Get recommendations
    recommendations = recommender.get_recommendations(top_n=10)
    print("\nPersonalized Recommendations:")
    print(recommendations[['title', 'rating', 'similarity_score']])