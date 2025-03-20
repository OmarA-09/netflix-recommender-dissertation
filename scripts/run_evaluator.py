# evaluation_script.py
from src.personalised_netflix_recommender import PersonalisedNetflixRecommender
from src.evaluator import RecommenderEvaluator

# Step 1: Initialize your recommender
recommender = PersonalisedNetflixRecommender()
recommender.load_data('data/netflix_titles.csv')
recommender.preprocess()

# Step 2: Initialize the evaluator
evaluator = RecommenderEvaluator()

# Step 3: Find titles that actually exist in your dataset
print("Finding titles in the dataset...")
# Let's find some titles that actually exist in your dataset
kids_ratings = ['TV-Y', 'TV-Y7', 'TV-G', 'G']
teens_ratings = ['TV-PG', 'PG', 'TV-14', 'PG-13']
adults_ratings = ['TV-MA', 'R', 'NC-17']

# Sample 3 titles from each age group
kids_titles = recommender.df[recommender.df['rating'].isin(kids_ratings)].sample(3)['title'].tolist()
teens_titles = recommender.df[recommender.df['rating'].isin(teens_ratings)].sample(3)['title'].tolist()
adults_titles = recommender.df[recommender.df['rating'].isin(adults_ratings)].sample(3)['title'].tolist()

print(f"Kids titles: {kids_titles}")
print(f"Teens titles: {teens_titles}")
print(f"Adults titles: {adults_titles}")

# Step 4: Compare different methods or configurations
weighted_recommender = PersonalisedNetflixRecommender()
weighted_recommender.content_weight = 0.5
weighted_recommender.rating_weight = 0.5

methods = {
    'baseline': recommender,
    'weighted': weighted_recommender
}

# Load data for each method
for method_name, method_recommender in methods.items():
    if method_name != 'baseline':  # Skip baseline as it's already loaded
        method_recommender.load_data('data/netflix_titles.csv')
        method_recommender.preprocess()

# Helper function to handle empty results
def print_results_safely(results, columns):
    if results is not None and not results.empty:
        print(results[columns])
    else:
        print("No valid results were generated. Check if recommendations were produced.")

# Step 5: Run evaluation
print("\nEvaluating kids profile...")
kids_results = evaluator.compare_methods(kids_titles, 'kids', methods)
print_results_safely(kids_results, ['Method', 'Accuracy', 'Precision', 'F1 Score', 'RMSE'])

print("\nEvaluating teens profile...")
teens_results = evaluator.compare_methods(teens_titles, 'teens', methods)
print_results_safely(teens_results, ['Method', 'Accuracy', 'Precision', 'F1 Score', 'RMSE'])

print("\nEvaluating adults profile...")
adults_results = evaluator.compare_methods(adults_titles, 'adults', methods)
print_results_safely(adults_results, ['Method', 'Accuracy', 'Precision', 'F1 Score', 'RMSE'])