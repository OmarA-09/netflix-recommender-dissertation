from src.recommenders.CosineNetflixRecommender import CosineNetflixRecommender
# Allows the script to be run standalone for testing

def create_profile(recommender):
    recommender.reset_preferences()
    
    liked_titles = [
        # TV-PG, TV-Y7, PG titles from our dataset
        "Our Planet",
        "Planet Earth II",
        "NOVA: Bird Brain",
        "NOVA: Black Hole Apocalypse",
        "NOVA: Eclipse Over America", 
        "NOVA: Death Dive to Saturn",
        "Octonauts",
        "Paw Patrol",
        "Power Rangers Dino Charge",
        "Power Rangers Ninja Steel",
        "Pokémon the Series",
        "Pokémon: Indigo League",
        "Pocoyo",
        "Pablo",
        "Penguins of Madagascar",
        "Paddleton"
    ]
    
    # DISLIKES - Mature, explicit content from our dataset
    disliked_titles = [
        "Narcos",
        "Nymphomaniac: Volume 1",
        "Nymphomaniac: Volume II",
        "Nocturnal Animals",
        "Nurse Jackie",
        "Naked",
        "Orange Is the New Black",
        "Ozark",
        "Outlaw King",
        "Only God Forgives",
        "Paranormal Activity",
        "Pulp Fiction",
        "Primal Fear",
        "Piercing",
        "Platoon",
        "Pineapple Express",
        "Psycho",
        "Pan's Labyrinth",
        "Polar",
        "Project Power",
        "Pieces of a Woman",
        "Panic Room",
        "Peaky Blinders",
        "Pawn Stars",
        "Perfect Stranger",
        "Philadelphia"
    ]
    
    # Like the liked list
    successful_likes = 0
    for title in liked_titles:
        try:
            success = recommender.like_title(title)
            if success:
                successful_likes += 1
                print(f"Liked: {title}")
            else:
                print(f"Failed to like: {title}")
        except Exception as e:
            print(f"Error liking title {title}: {str(e)}")
    
    # Dislike the disliked list
    successful_dislikes = 0
    for title in disliked_titles:
        try:
            success = recommender.dislike_title(title)
            if success:
                successful_dislikes += 1
                print(f"Disliked: {title}")
            else:
                print(f"Failed to dislike: {title}")
        except Exception as e:
            print(f"Error disliking title {title}: {str(e)}")
    
    print(f"Successfully liked {successful_likes} titles and disliked {successful_dislikes} titles")
    
    return recommender

if __name__ == "__main__":
    test_recommender = CosineNetflixRecommender()
    test_recommender.load_data('data/netflix_titles.csv')
    test_recommender.preprocess()
    create_profile(test_recommender)
    print("Seed profile created successfully!")