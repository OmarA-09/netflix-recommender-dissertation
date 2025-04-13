# netflix-recommender-dissertation

From netflix-recommender-dissertation:

## Install required packages:

`pip install -r requirements.txt`

## For Terminal View: Run the run_recommender script:

`python ./scripts/run_recommender.py`

- Note: Ensure you select the correct recommender type for init before executing app.py

## To run Backend with endpoints:

`python ./src/app.py`

- Note: Ensure you select the correct recommender type before executing app.py

## Test with API curl commands in another terminal!

### Test the home endpoint

curl http://127.0.0.1:5000/

curl http://127.0.0.1:5000/profile

# 3. Like a Title
curl -X POST http://127.0.0.1:5000/like -H "Content-Type: application/json" -d '{"title": "Stranger Things"}'

# 4. Dislike a Title
curl -X POST http://127.0.0.1:5000/dislike -H "Content-Type: application/json" -d '{"title": "The Crown"}'

# 5. Get Recommendations (Default)
curl http://127.0.0.1:5000/recommendations


# 6. Get Recommendations with Count 
curl "http://127.0.0.1:5000/recommendations?count=5"

# 7. Get Recommendations with Rating Filter
curl "http://127.0.0.1:5000/recommendations?rating_filter=PG"

# 8. Search Titles
curl "http://127.0.0.1:5000/search?query=stranger"

# 9. Search Titles with Rating Filter
curl "http://127.0.0.1:5000/search?query=drama&rating=TV-14"

# 10. Get Evaluation Metrics
curl http://127.0.0.1:5000/evaluation