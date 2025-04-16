import React, { useState, useEffect } from 'react';

const API_BASE_URL = 'http://127.0.0.1:5000';

function ProfilePage({ onBackToHome }) {
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch user profile
  const fetchProfile = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/profile`);
      if (!response.ok) {
        throw new Error('Failed to fetch profile');
      }
      const data = await response.json();
      setProfile(data);
      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  // Reset preferences
  const resetPreferences = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/reset`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw new Error('Failed to reset preferences');
      }
      
      // Refetch profile after reset
      fetchProfile();
      
      alert('Preferences have been reset successfully');
    } catch (err) {
      alert(`Error: ${err.message}`);
    }
  };

  // Fetch profile on component mount
  useEffect(() => {
    fetchProfile();
  }, []);

  // Loading state
  if (loading) {
    return (
      <div className="max-w-4xl mx-auto p-4 text-center">
        <h1 className="text-2xl font-bold">Loading Profile...</h1>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="max-w-4xl mx-auto p-4 text-center text-red-500">
        <h1 className="text-2xl font-bold">Error Loading Profile</h1>
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-4">
      <h1 className="text-3xl font-bold text-center mb-6">User Preferences</h1>
      
      {/* Preferences Summary */}
      <div className="bg-gray-100 p-4 rounded-lg mb-6">
        <div className="flex justify-between">
          <div>
            <h2 className="font-bold">Liked Titles</h2>
            <p>{profile.liked_count} titles</p>
          </div>
          <div>
            <h2 className="font-bold">Disliked Titles</h2>
            <p>{profile.disliked_count} titles</p>
          </div>
        </div>
      </div>
      
      {/* Liked Titles Section */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-4">Liked Titles</h2>
        {profile.liked_titles.length > 0 ? (
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            {profile.liked_titles.map((title, index) => (
              <div 
                key={index} 
                className="border p-2 rounded shadow"
              >
                <h3 className="font-bold text-sm truncate">{title}</h3>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-500">No liked titles yet</p>
        )}
      </div>
      
      {/* Disliked Titles Section */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-4">Disliked Titles</h2>
        {profile.disliked_titles.length > 0 ? (
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            {profile.disliked_titles.map((title, index) => (
              <div 
                key={index} 
                className="border p-2 rounded shadow"
              >
                <h3 className="font-bold text-sm truncate">{title}</h3>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-500">No disliked titles yet</p>
        )}
      </div>
      
      {/* Action Buttons */}
      <div className="flex justify-between">
        <button 
          onClick={onBackToHome}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        >
          Back to Recommendations
        </button>
        <button 
          onClick={resetPreferences}
          className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600"
        >
          Reset Preferences
        </button>
      </div>
    </div>
  );
}

export default ProfilePage;