# Netflix Recommender Frontend

## Prerequisites
- Node.js (v14 or later)
- npm or yarn
- Backend Flask application running on `http://127.0.0.1:5000`

## Setup

1. Clone the repository
2. Install dependencies:
```bash
npm install
# or
yarn install
```

3. Install Tailwind CSS (if not already installed):
```bash
npm install -D tailwindcss
npx tailwindcss init
```

4. Configure Tailwind in `tailwind.config.js`:
```javascript
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

5. Add Tailwind directives toyour CSS file (e.g., `src/index.css`):
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

## Running the Application

1. Start the development server:
```bash
npm start
# or
yarn start
```

2. Open `http://localhost:5000` in your browser

## Key Features
- Browse Netflix titles
- Search films
- Like/Dislike individual titles
- Get personalized recommendations

## Configuration

Modify the `API_BASE_URL` in the main React component if your backend runs on a different host/port.

## Troubleshooting
- Ensure CORS is enabled on your Flask backend
- Check that the backend is running on the expected port
- Verify network connectivity between frontend and backend

## Build for Production
```bash
npm run build
# or
yarn build
```

## Deployment Notes
- The app uses Tailwind CSS for styling
- Requires a modern browser with ES6 support
- Tested with React 17+