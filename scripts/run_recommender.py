# from src.personalised_netflix_recommender import PersonalisedNetflixRecommender
# from src.knn_recommender import KNNNetflixRecommender  

from src.recommenders.CosineNetflixRecommender import CosineNetflixRecommender
from src.recommenders.KNNNetflixRecommender import KNNNetflixRecommender



def main():
    print("====================================================")
    print("   Netflix Content-Based Personalized Recommender   ")
    print("====================================================")
    print("\nInitializing recommender system...\n")
    
    # Initialize the recommender: Choose from 2 options below by uncommenting

    recommender = KNNNetflixRecommender()
    # recommender = CosineNetflixRecommender()
    
    # Load the Netflix data
    recommender.load_data('data/netflix_titles.csv')
    
    # Preprocess the data
    recommender.preprocess()
    
    print("\nWelcome to your personalized Netflix recommender!")
    print("This system will learn your preferences as you like or dislike titles.")
    
    while True:
        print("\n" + "="*60)
        print("MENU OPTIONS:")
        print("1. Search for a title")
        print("2. Like a title")
        print("3. Dislike a title")
        print("4. Get personalized recommendations")
        print("5. View my preferences")
        print("6. Reset my preferences")
        print("7. Exit")
        print("="*60)
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            query = input("Enter search term: ").strip()
            results = recommender.search_titles(query)
            if not results.empty:
                print("\nSearch results:")
                for i, (_, row) in enumerate(results.iterrows(), 1):
                    print(f"{i}. {row['title']} ({row['type']}, {row['rating']}, {row['release_year']})")
        
        elif choice == '2':
            title = input("Enter the exact title you want to like: ").strip()
            recommender.like_title(title)
        
        elif choice == '3':
            title = input("Enter the exact title you want to dislike: ").strip()
            recommender.dislike_title(title)
        
        elif choice == '4':
            rating_option = input("Filter by rating? (y/n): ").strip().lower()
            rating_filter = None
            if rating_option == 'y':
                rating_filter = input("Enter specific rating (e.g., TV-14, PG-13, etc.): ").strip()
            
            top_n = 10
            try:
                n_input = input(f"Number of recommendations (default: {top_n}): ").strip()
                if n_input:
                    top_n = int(n_input)
            except ValueError:
                print("Using default value of 10")
            
            recommendations = recommender.get_recommendations(top_n=top_n, rating_filter=rating_filter)
            if not recommendations.empty:
                print("\nYour personalized recommendations:")
                for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                    print(f"{i}. {row['title']} ({row['type']}, {row['rating']}, {row['release_year']})")
                    print(f"   Similarity: {row['similarity_score']:.4f}")
                    print(f"   Description: {row['description'][:100]}..." if len(row['description']) > 100 
                          else f"   Description: {row['description']}")
                    print()
        
        elif choice == '5':
            recommender.get_user_preferences()
        
        elif choice == '6':
            confirm = input("Are you sure you want to reset all your preferences? (y/n): ").strip().lower()
            if confirm == 'y':
                recommender.reset_preferences()
        
        elif choice == '7':
            print("\nThank you for using the Netflix recommender system! Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter a number from 1 to 7.")

if __name__ == "__main__":
    main()