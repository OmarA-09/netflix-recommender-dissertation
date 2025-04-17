# Netflix Recommender Frontend User Guide

**From /frontend:**

## Prerequisites
Check if you have the following prerequisites installed and install them if needed:

### 1. Node.js (v14 or later)
**Check installation:**
```bash
node --version
```

**Install if needed:**
- Windows: Download and install from [nodejs.org](https://nodejs.org/)
- macOS: 
  ```bash
  brew install node
  ```
- Linux (Ubuntu/Debian):
  ```bash
  sudo apt update
  sudo apt install nodejs npm
  ```

### 2. npm or yarn
**Check installation:**
```bash
# For npm - if Node.js is installed then use this npm command
npm --version

# For yarn
yarn --version
```

**Install yarn if needed:**
```bash
npm install -g yarn
```

### 3. Backend Flask Application
**Check if running:**
```bash
curl http://127.0.0.1:5000
```
You should receive a response if the backend is running.

**Install and run the backend if needed:**
```bash
# Navigate to the backend directory
cd ../backend

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask application
python app.py
```

Make sure the backend Flask application is running on `http://127.0.0.1:5000` before starting the frontend.

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
- Get personalised recommendations

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
- App uses Tailwind CSS for styling
- Requires modern browser with ES6 support
- Tested with React 17+