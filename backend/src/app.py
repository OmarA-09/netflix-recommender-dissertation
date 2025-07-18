from flask import Flask, request, jsonify
from src.recommenders.CosineNetflixRecommender import CosineNetflixRecommender
from src.recommenders.KNNNetflixRecommender import KNNNetflixRecommender
from src.evaluator import RecommenderEvaluator
from flask_cors import CORS
import os
from seed_user import create_profile
import math
import numpy as np

app = Flask(__name__)

CORS(app, 
     resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     supports_credentials=True,  # This is important for some CORS scenarios
     max_age=3600
)

###############################################
# IMPORTANT: Config the below:
# Init the recommender system: Uncomment below to switch between the different recommender types.
###############################################

recommender = KNNNetflixRecommender()
# recommender = CosineNetflixRecommender()
recommender.load_data('data/netflix_titles.csv')
recommender.preprocess()

# Check if user profile exists, if not create it
if not os.path.exists(recommender.profile_path) or os.path.getsize(recommender.profile_path) == 0:
    print(f"No existing profile found at {recommender.profile_path}. Creating seed profile...")
    # Create the profile using the imported function
    create_profile(recommender)
else:
    print(f"Existing profile found at {recommender.profile_path}. Using saved preferences.")


# Initialize evaluator
evaluator = RecommenderEvaluator(recommender)

@app.route('/')
def home():
    return jsonify({
        'status': 'active',
        'message': 'Netflix Age-Appropriate Recommender API',
        'endpoints': [
            '/profile',
            '/like (POST)',
            '/dislike (POST)',
            '/recommendations',
            '/search',
            '/evaluation'
        ]
    })

@app.route('/profile', methods=['GET'])
def get_profile():
    liked_titles = list(recommender.user_liked_titles)
    disliked_titles = list(recommender.user_disliked_titles)
    
    return jsonify({
        'liked_titles': liked_titles,
        'disliked_titles': disliked_titles,
        'profile_status': 'active' if len(liked_titles) > 0 else 'new',
        'liked_count': len(liked_titles),
        'disliked_count': len(disliked_titles)
    })


@app.route('/like', methods=['POST', 'OPTIONS'])
def like_title():
    # Handle CORS
    if request.method == 'OPTIONS':
        response = jsonify(success=True)
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,GET,OPTIONS')
        return response

    print("DEBUG: Incoming Request")
    print("Method:", request.method)
    print("Headers:", request.headers)
    print("Content Type:", request.content_type)
    
    try:
        data = request.get_json(force=True, silent=False)
        print("Parsed Data:", data)
    except Exception as e:
        print("JSON Parsing Error:", str(e))
        return jsonify({'error': 'Invalid JSON', 'details': str(e)}), 400
    
    title = data.get('title')
    if not title:
        return jsonify({'error': 'Title parameter is required'}), 400
    
    success = recommender.like_title(title)
    if success:
        return jsonify({
            'status': 'success',
            'message': f"Added '{title}' to liked titles"
        })
    else:
        return jsonify({'error': 'Title not found'}), 404

@app.route('/dislike', methods=['POST'])
def dislike_title():
    data = request.get_json()
    title = data.get('title') if data else None
    
    if not title:
        return jsonify({'error': 'Title parameter is required'}), 400
    
    success = recommender.dislike_title(title)
    if success:
        return jsonify({
            'status': 'success',
            'message': f"Added '{title}' to disliked titles"
        })
    else:
        return jsonify({'error': 'Title not found'}), 404

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    count = request.args.get('count', default=10, type=int)
    rating_filter = request.args.get('rating_filter', default=None)
    
    # Check if we have enough data
    if len(recommender.user_liked_titles) == 0:
        return jsonify({
            'message': 'Not enough preference data. Please like some titles first!',
            'recommendations': []
        })
    
    recommendations = recommender.get_recommendations(top_n=count, rating_filter=rating_filter)
    
    if recommendations.empty:
        return jsonify({
            'message': 'No suitable recommendations found with the current filters',
            'recommendations': []
        })
    
    # Convert recommendations -> list of dictionaries
    recs_list = recommendations.to_dict(orient='records')
    
    return jsonify({
        'status': 'success',
        'count': len(recs_list),
        'recommendations': recs_list
    })

@app.route('/search', methods=['GET'])
def search_titles():
    query = request.args.get('query', '')
    rating = request.args.get('rating', None)
    
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400
    
    # Basic search in title
    results = recommender.df[recommender.df['title'].str.lower().str.contains(query.lower())]
    
    # Apply filter if specified
    if rating:
        results = results[results['rating'] == rating]
    
    # Convert -> list of dictionaries
    results_list = results[['title', 'type', 'rating', 'release_year']].head(20).to_dict(orient='records')
    
    return jsonify({
        'status': 'success',
        'count': len(results_list),
        'results': results_list
    })


@app.route('/evaluation', methods=['GET'])
def get_evaluation():
    if len(recommender.user_liked_titles) == 0:
        return jsonify({
            'error': 'Not enough preference data. Please like some titles first!'
        }), 400
    
    recommendations = recommender.get_recommendations(top_n=20)
    
    if recommendations.empty:
        return jsonify({
            'error': 'Could not generate recommendations for evaluation'
        }), 400
    
    kids_metrics = evaluator.evaluate_recommendations(recommendations, 'kids')
    teens_metrics = evaluator.evaluate_recommendations(recommendations, 'teens')
    adults_metrics = evaluator.evaluate_recommendations(recommendations, 'adults')
    
    return jsonify({
        'status': 'success',
        'metrics': {
            'kids': {
                'accuracy': kids_metrics['accuracy'],
                'precision': kids_metrics['precision'],
                'f1_score': kids_metrics['f1_score'],
                'rmse': kids_metrics['rmse']
            },
            'teens': {
                'accuracy': teens_metrics['accuracy'],
                'precision': teens_metrics['precision'],
                'f1_score': teens_metrics['f1_score'],
                'rmse': teens_metrics['rmse']
            },
            'adults': {
                'accuracy': adults_metrics['accuracy'],
                'precision': adults_metrics['precision'],
                'f1_score': adults_metrics['f1_score'],
                'rmse': adults_metrics['rmse']
            }
        }
    })

@app.route('/user/preferences', methods=['GET'])
def get_user_preferences():
    try:
        liked_df, disliked_df = recommender.get_user_preferences()
        
        # Convert DataFrames -> list of dictionaries
        liked_titles = liked_df[['title', 'type', 'rating', 'release_year']].to_dict(orient='records')
        disliked_titles = disliked_df[['title', 'type', 'rating', 'release_year']].to_dict(orient='records')
        
        preference_analysis = recommender.integration_service.analyse_user_preferences(liked_df)
        
        return jsonify({
            'liked_titles': liked_titles,
            'disliked_titles': disliked_titles,
            'liked_count': len(liked_titles),
            'disliked_count': len(disliked_titles),
            'genre_insights': preference_analysis.get('favorite_genres', {}) if preference_analysis else {}
        }), 200
    except Exception as e:
        print(f"Error in user preferences endpoint: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve user preferences',
            'details': str(e)
        }), 500

@app.route('/reset', methods=['POST'])
def reset_preferences():
    try:
        recommender.reset_preferences()
        return jsonify({
            'message': 'User preferences have been successfully reset'
        }), 200
    except Exception as e:
        logger.error(f"Error resetting preferences: {str(e)}")
        return jsonify({'error': 'Failed to reset preferences'}), 500

@app.route('/films', methods=['GET'])
def list_films():
    """
    Endpoint to list films with optional search and pagination
    
    Query Parameters:
    - query: Optional search term to filter titles
    - page: Page number for pagination (default: 1)
    - limit: Number of results per page (default: 20, max: 100)
    - type: Optional filter for 'Movie' or 'TV Show'
    - rating: Optional filter for content rating
    """
    try:
        query = request.args.get('query', '').strip()
        page = max(1, request.args.get('page', 1, type=int))
        limit = min(100, max(1, request.args.get('limit', 20, type=int)))
        content_type = request.args.get('type', '').strip()
        rating = request.args.get('rating', '').strip()
        
        filtered_df = recommender.df.copy()
        
        # search filter
        if query:
            filtered_df = filtered_df[
                filtered_df['title'].str.contains(query, case=False, na=False)
            ]
        
        # type filter
        if content_type:
            filtered_df = filtered_df[filtered_df['type'].str.lower() == content_type.lower()]
        
        # rating filter
        if rating:
            filtered_df = filtered_df[filtered_df['rating'] == rating]
        
        # Sort by release year
        filtered_df = filtered_df.sort_values('release_year', ascending=False)
        
        # Pagination
        total_count = len(filtered_df)
        total_pages = math.ceil(total_count / limit)
        
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        page_df = filtered_df.iloc[start_idx:end_idx]
        
        results = page_df[['title', 'type', 'rating', 'release_year', 'description']].to_dict(orient='records')
        
        return jsonify({
            'status': 'success',
            'total_count': total_count,
            'page': page,
            'total_pages': total_pages,
            'limit': limit,
            'results': results
        }), 200
    
    except Exception as e:
        print(f"Error in films list endpoint: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve film list',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)