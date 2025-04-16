import React, { useState } from 'react';
import NetflixRecommenderApp from './NetflixRecommenderApp';
import ProfilePage from './ProfilePage';

function App() {
  const [currentPage, setCurrentPage] = useState('home');

  const switchToProfile = () => {
    setCurrentPage('profile');
  };

  const switchToHome = () => {
    setCurrentPage('home');
  };

  return (
    <div>
      {/* Navigation */}
      <nav className="border-b border-gray-200 p-4">
        <div className="max-w-6xl mx-auto flex justify-between items-center">
          <h1 className="text-3xl font-bold">Netflix Recommender</h1>
          {currentPage === 'home' ? (
            <button 
              onClick={switchToProfile}
              className="px-4 py-2 text-blue-600 hover:bg-blue-50 rounded-lg flex items-center"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
              </svg>
              Profile
            </button>
          ) : (
            <button 
              onClick={switchToHome}
              className="px-4 py-2 text-green-600 hover:bg-green-50 rounded-lg flex items-center"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
              </svg>
              Home
            </button>
          )}
        </div>
      </nav>

      {/* Main Content */}
      {currentPage === 'home' ? (
        <NetflixRecommenderApp onProfileClick={switchToProfile} />
      ) : (
        <ProfilePage onBackToHome={switchToHome} />
      )}
    </div>
  );
}

export default App;