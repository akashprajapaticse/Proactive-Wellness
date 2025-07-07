✨ Proactive Wellness App ✨
Tired of feeling drained after a long day at your desk? 😴 The Proactive Wellness App is your personal AI-powered companion, designed to revolutionize your work-life balance! This desktop application leverages cutting-edge computer vision and smart activity monitoring to keep you at your peak performance and prevent burnout. Get ready for personalized insights and actionable recommendations that truly make a difference.

🌟 Key Features That Keep You Thriving
👁️ Real-time Wellness Monitoring:

Computer Vision (Webcam): Tracks your eye gaze stability, facial emotions, body posture, ambient lighting conditions, and even your distance from the screen.

Activity Monitoring: Keeps tabs on your mouse movement (Pixels Per Second - PPS) and keyboard typing speed (Words Per Minute - WPM).

Simulated Metrics: Includes a dynamic heart rate and hydration level (with manual logging and automatic detection of drinking actions for convenience!).

🧠 AI-Powered Fatigue Prediction:

A custom Machine Learning model (trained on synthetic data for demonstration purposes) intelligently analyzes all your real-time wellness and activity metrics to predict your current fatigue score (0-100).

💡 Intelligent Recommendation System:

Receive dynamic, context-aware suggestions tailored just for you! From "Take a break!" to "Adjust posture!" or "Drink water!", our system guides you to better habits.

Need a moment? Use the "Dismiss for a While" option to temporarily suppress suggestions.

Feeling a bit low? Our app might even surprise you with a light-hearted "joke" to uplift your mood! 😄

⏱️ Interactive Wellness Tools:

Water Tracker: Easily monitor your water intake with manual logging and smart detection of drinking actions via your webcam. Stay hydrated effortlessly!

Nap Timer: Set a customizable countdown for quick power naps to boost alertness and recharge.

Exercise Timer: Schedule short, invigorating exercise breaks to combat inactivity and keep your energy flowing.

🌈 Dynamic UI Feedback: The application's background color isn't just pretty – it dynamically changes to instantly reflect your wellness state, providing immediate visual cues for critical issues like high fatigue or urgent alarms.

⚙️ How It Works Under The Hood
The Proactive Wellness App operates on a seamless client-server architecture:

🐍 Python Flask Backend (backend/app.py):

The powerhouse for data collection and AI processing.

Utilizes OpenCV and MediaPipe for robust real-time computer vision (face mesh for gaze/emotion/distance, pose estimation for posture/drinking).

Employs pynput to precisely monitor your mouse and keyboard activity.

Integrates FER (Face Emotion Recognition) for accurate emotion detection.

Loads a pre-trained scikit-learn Machine Learning model (saved via joblib) to predict fatigue based on all collected metrics.

Exposes a clean REST API endpoint (/predict_wellness) to deliver all this rich, real-time wellness data to the frontend.

Runs the webcam and activity monitoring in a separate, efficient thread to ensure the API remains lightning-fast and responsive.

⚛️ React Frontend (frontend/src/App.js):

Your beautiful, responsive user interface.

Built with React and elegantly styled with Tailwind CSS for a modern and intuitive experience.

Periodically fetches the latest wellness data from the Python backend.

Intelligently processes the received data to drive the dynamic recommendation system.

Manages your hydration tracking and powers the interactive timers for naps and exercises.

Dynamically updates the UI with clear visual feedback and helpful notifications.

🚀 Get Started: Setup & Installation
Ready to take control of your wellness? Follow these simple steps to get the app running locally:

Prerequisites
Python 3.8+

Node.js and npm (or yarn)

Backend Setup
Clone the repository:

git clone https://github.com/your-username/proactive-wellness-app.git
cd proactive-wellness-app/backend

Create and activate a virtual environment:

python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

Install Python dependencies:

pip install -r requirements.txt

(Ensure your requirements.txt includes: Flask, Flask-Cors, opencv-python, mediapipe, fer, pynput, scikit-learn, pandas, joblib)

Generate the ML model:
This crucial step creates the fatigue_model.joblib file.

python generate_and_train_fatigue_model.py

Run the Flask backend:

python app.py

Keep this terminal open! A new OpenCV window showing your webcam feed should pop up.

Frontend Setup
Navigate to the frontend directory:

cd ../frontend

Install Node.js dependencies:

npm install
# or yarn install

Run the React frontend:

npm start
# or yarn start

This will magically open the application in your default web browser (usually http://localhost:3000).

🎮 How to Use Your Wellness Companion
Launch Both: Make sure both the Python backend and React frontend are actively running.

Open in Browser: Navigate to http://localhost:3000.

Start Monitoring: Click the "Start Working Time" button to kick off your wellness journey.

Observe & Interact: Watch your real-time metrics update, receive smart suggestions, and use the nap, exercise, and water logging features to stay on track.

Quit: To close the webcam feed, simply press Q in the OpenCV window or stop the backend process in your terminal.

🤝 Contributing to a Healthier Workflow
We welcome contributions of all kinds! Whether it's bug fixes, new features, or documentation improvements, your help makes this app even better.

Fork this repository.

Create your feature branch (git checkout -b feature/AmazingFeature).

Commit your changes (git commit -m 'Add some AmazingFeature').

Push to the branch (git push origin feature/AmazingFeature).

Open a Pull Request.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.