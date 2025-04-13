from src.recommenders.CosineNetflixRecommender import CosineNetflixRecommender

def create_profile(recommender):
    """Add seed likes to an existing recommender"""
    # Reset preferences
    recommender.reset_preferences()
    
    # Hard-coded titles for teens
    teens_titles = [
        "Stranger Things",
        "The Flash",
        "Teen Wolf",
        "Riverdale",
        "On My Block"
    ]
    
    # Like each title
    for title in teens_titles:
        recommender.like_title(title)
    
    return recommender

if __name__ == "__main__":
    # This allows the script to be run standalone for testing
    test_recommender = CosineNetflixRecommender()
    test_recommender.load_data('data/netflix_titles.csv')
    test_recommender.preprocess()
    create_profile(test_recommender)
    print("Seed profile created successfully!")