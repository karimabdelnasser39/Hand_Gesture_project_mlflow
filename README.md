# Hand_Gesture_project_mlflow

Real-Time Hand Gesture Recognition with MLflow
This repository contains a research-focused implementation of a real-time hand gesture recognition system. The project utilizes MediaPipe for hand landmark extraction and evaluates multiple machine learning classifiers to achieve high-accuracy gesture detection.

Project Overview
The core of this project is a pipeline that captures video from a webcam, extracts 21 hand landmarks (63 3D coordinates), normalizes the data, and classifies the gesture into one of 18 distinct classes.

Key Features
Real-Time Detection: Processes live video feeds using OpenCV and MediaPipe.

Coordinate Normalization: Landmarks are centered around the wrist and scaled by the hand's size to ensure consistent predictions regardless of hand distance or position.

Experiment Tracking: Integrated with MLflow to track hyperparameters, performance metrics, and model artifacts.

Ensemble Logic: Implements a majority voting system across multiple models to ensure stable and robust predictions during live inference.

Experimentation & Model Comparison
We conducted a comprehensive experiment named Gesture_Recognition_Research to compare four different classification architectures.

Performance Summary
Based on the experiments tracked in MLflow:

AdaBoostClassifier emerged as the top-performing model, achieving near-perfect accuracy and F1-scores across 18 gesture classes (such as call, fist, peace, rock, and stop).

The models were evaluated on a dataset of 25,675 samples with 64 features.

Project Structure
my_project.ipynb: The main research notebook containing data visualization, model training, MLflow logging, and the real-time inference loop.

mlflow_utils.py: A utility module designed to handle MLflow experiment initialization, logging of confusion matrices, and model registration.

mlruns/: Directory containing the MLflow tracking database (parameters, metrics, and models).

screenshots/: Visual evidence from the MLflow UI, including metric charts and the Model Registry.

Installation & Usage
Dependencies: Install the required libraries:

Training: Run the my_project.ipynb notebook to execute the full training pipeline and log results to MLflow.

Real-Time Inference: Use the final cell in the notebook to launch the webcam interface. Press 'q' to exit the live feed.

🏆 Model Selection
The AdaBoostClassifier was selected for production (Registered Model: Final_Gesture_Recognition_Model) due to its superior handling of high-dimensional landmark data and its consistent performance in cross-validation.
