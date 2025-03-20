from src.personalised_netflix_recommender import PersonalisedNetflixRecommender


def create_profile(recommender=None):
    # If no recommender provided, create one
    if recommender is None:
        recommender = PersonalisedNetflixRecommender()
        recommender.load_data('data/netflix_titles.csv')
        recommender.preprocess()
    
    # Reset preferences
    recommender.reset_preferences()
    
    # Hard-coded titles
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
    # This allows the script to be run standalone
    create_profile()