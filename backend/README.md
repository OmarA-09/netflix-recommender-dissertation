# netflix-recommender-dissertation

From netflix-recommender-dissertation:

## Install required packages:

`pip install -r requirements.txt`

## For Terminal View: Run the run_recommender script from backend folder:

`python ./scripts/run_recommender.py`

- Note: Ensure you select the correct recommender type for init before executing app.py

## To run Backend with endpoints:

`python ./src/app.py`

- Note: Ensure you select the correct recommender type before executing app.py

## Test with API curl commands in another terminal!

### Test the home endpoint

curl http://127.0.0.1:5000/

curl http://127.0.0.1:5000/profile

### 3. Like a Title
curl -X POST http://127.0.0.1:5000/like -H "Content-Type: application/json" -d '{"title": "Stranger Things"}'

### 4. Dislike a Title
curl -X POST http://127.0.0.1:5000/dislike -H "Content-Type: application/json" -d '{"title": "The Crown"}'

### 5. Get Recommendations (Default)
curl http://127.0.0.1:5000/recommendations

### 6. Get Recommendations with Count 
curl "http://127.0.0.1:5000/recommendations?count=5"

### 7. Get Recommendations with Rating Filter
curl "http://127.0.0.1:5000/recommendations?rating_filter=PG"

### 8. Search Titles
curl "http://127.0.0.1:5000/search?query=stranger"

### 9. Search Titles with Rating Filter
curl "http://127.0.0.1:5000/search?query=drama&rating=TV-14"

### 10. Get Evaluation Metrics
curl http://127.0.0.1:5000/evaluation

### 11. Get User Preferences
curl http://127.0.0.1:5000/user/preferences

### 12. Reset User Preferences
curl -X POST http://127.0.0.1:5000/reset

## List films
curl http://127.0.0.1:5000/films

### Search for films with "Luv" in the title
curl "http://127.0.0.1:5000/films?query=Luv"

### Filter for Movies
curl "http://127.0.0.1:5000/films?type=Movie"

### Filter for TV Shows
curl "http://127.0.0.1:5000/films?type=TV%20Show"

### Paginate to second page of results
curl "http://127.0.0.1:5000/films?page=2"

### Limit results to 5 per page
curl "http://127.0.0.1:5000/films?limit=5"

### Filter by rating (e.g., TV-MA)
curl "http://127.0.0.1:5000/films?rating=TV-MA"

### Combine multiple filters
curl "http://127.0.0.1:5000/films?query=Lupin&type=Movie&rating=PG-13"

### Search Indian movies
curl "http://127.0.0.1:5000/films?query=India&type=Movie"