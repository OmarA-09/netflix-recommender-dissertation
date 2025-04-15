import React, { useState, useEffect } from 'react';

// Base API URL (can be changed to environment variable)
const API_BASE_URL = 'http://127.0.0.1:5000';

function NetflixRecommenderApp() {
  // State management
  const [films, setFilms] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [selectedFilm, setSelectedFilm] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  
  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  
  // Filtering state
  const [typeFilter, setTypeFilter] = useState('');
  const [ratingFilter, setRatingFilter] = useState('');

  // Fetch films with pagination and filters
  const fetchFilms = async (page = 1) => {
    try {
      // Construct query parameters
      const params = new URLSearchParams({
        page: page,
        query: searchQuery,
        type: typeFilter,
        rating: ratingFilter,
        limit: 12  // Adjust number of items per page
      });

      const response = await fetch(`${API_BASE_URL}/films?${params}`);
      const data = await response.json();
      
      setFilms(data.results);
      setCurrentPage(data.page);
      setTotalPages(data.total_pages);
    } catch (error) {
      console.error('Error fetching films:', error);
    }
  };

  // Initial data fetch
  useEffect(() => {
    fetchFilms();
  }, []);

  // Refetch films when filters or search changes
  useEffect(() => {
    fetchFilms(1);
  }, [searchQuery, typeFilter, ratingFilter]);

  // Like a film
  const likeFilm = async (title) => {
    try {
      await fetch(`${API_BASE_URL}/like`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title })
      });
      alert(`Liked: ${title}`);
    } catch (error) {
      console.error('Error liking film:', error);
    }
  };

  // Dislike a film
  const dislikeFilm = async (title) => {
    try {
      await fetch(`${API_BASE_URL}/dislike`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title })
      });
      alert(`Disliked: ${title}`);
    } catch (error) {
      console.error('Error disliking film:', error);
    }
  };

  // Get recommendations
  const getRecommendations = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/recommendations`);
      const data = await response.json();
      setRecommendations(data.recommendations);
    } catch (error) {
      console.error('Error getting recommendations:', error);
    }
  };

  // Pagination controls
  const changePage = (newPage) => {
    if (newPage > 0 && newPage <= totalPages) {
      fetchFilms(newPage);
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-4">
      <h1 className="text-3xl font-bold text-center mb-6">Netflix Recommender</h1>
      
      {/* Search and Filters Container */}
      <div className="flex flex-col md:flex-row gap-4 mb-6">
        {/* Search Input */}
        <div className="flex-grow">
          <input 
            type="text" 
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search films..."
            className="w-full p-2 border rounded"
          />
        </div>

        {/* Type Filter */}
        <select 
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value)}
          className="p-2 border rounded"
        >
          <option value="">All Types</option>
          <option value="Movie">Movies</option>
          <option value="TV Show">TV Shows</option>
        </select>

        {/* Rating Filter */}
        <select 
          value={ratingFilter}
          onChange={(e) => setRatingFilter(e.target.value)}
          className="p-2 border rounded"
        >
          <option value="">All Ratings</option>
          <option value="TV-Y">TV-Y</option>
          <option value="TV-Y7">TV-Y7</option>
          <option value="G">G</option>
          <option value="PG">PG</option>
          <option value="TV-14">TV-14</option>
          <option value="PG-13">PG-13</option>
          <option value="TV-MA">TV-MA</option>
          <option value="R">R</option>
        </select>

        {/* Recommendations Button */}
        <button 
          onClick={getRecommendations}
          className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
        >
          Get Recommendations
        </button>
      </div>

      {/* Films Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-6">
        {films.map((film) => (
          <div 
            key={film.title} 
            className="border p-2 rounded shadow hover:bg-gray-100 cursor-pointer"
            onClick={() => setSelectedFilm(film)}
          >
            <h2 className="font-bold text-sm truncate">{film.title}</h2>
            <p className="text-xs">{film.type} ({film.release_year})</p>
            <p className="text-xs">Rating: {film.rating}</p>
          </div>
        ))}
      </div>

      {/* Pagination Controls */}
      <div className="flex justify-center items-center space-x-4 mb-6">
        <button 
          onClick={() => changePage(currentPage - 1)}
          disabled={currentPage === 1}
          className="px-4 py-2 border rounded disabled:opacity-50"
        >
          Previous
        </button>
        <span>Page {currentPage} of {totalPages}</span>
        <button 
          onClick={() => changePage(currentPage + 1)}
          disabled={currentPage === totalPages}
          className="px-4 py-2 border rounded disabled:opacity-50"
        >
          Next
        </button>
      </div>

      {/* Selected Film Modal */}
      {selectedFilm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4">
          <div className="bg-white p-6 rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <h2 className="text-2xl font-bold mb-4">{selectedFilm.title}</h2>
            
            <div className="space-y-4">
              <p><strong>Type:</strong> {selectedFilm.type}</p>
              <p><strong>Release Year:</strong> {selectedFilm.release_year}</p>
              <p><strong>Rating:</strong> {selectedFilm.rating}</p>
              
              <div>
                <strong>Description:</strong>
                <p className="mt-2">{selectedFilm.description}</p>
              </div>
            </div>
            
            <div className="flex justify-between mt-6">
              <button 
                onClick={() => {
                  likeFilm(selectedFilm.title);
                  setSelectedFilm(null);
                }}
                className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
              >
                Like
              </button>
              <button 
                onClick={() => {
                  dislikeFilm(selectedFilm.title);
                  setSelectedFilm(null);
                }}
                className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600"
              >
                Dislike
              </button>
              <button 
                onClick={() => setSelectedFilm(null)}
                className="bg-gray-500 text-white px-4 py-2 rounded hover:bg-gray-600"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Recommendations Section */}
      {recommendations.length > 0 && (
        <div className="mt-8">
          <h2 className="text-2xl font-bold mb-4">Recommendations</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            {recommendations.map((rec) => (
              <div 
                key={rec.title} 
                className="border p-2 rounded shadow"
              >
                <h3 className="font-bold text-sm truncate">{rec.title}</h3>
                <p className="text-xs">{rec.type} ({rec.release_year})</p>
                <p className="text-xs">Rating: {rec.rating}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default NetflixRecommenderApp;