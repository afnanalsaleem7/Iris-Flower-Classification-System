# Iris-Flower-Classification-System
An end-to-end Machine Learning desktop application built in Python to classify Iris flower species based on their morphological features. The project follows a modular, object-oriented software architecture, cleanly separating data pipelines, model management, and the user interface.

📌 Project Overview
The system predicts the specific species of an Iris flower (*Iris setosa*, *Iris virginica*, or *Iris versicolor*) by analyzing four core botanical measurements:
* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

The pipeline evaluates multiple algorithmic approaches using Cross-Validation, ultimately utilizing a Support Vector Machine (SVC) to deliver high-accuracy desktop predictions.

📂 Repository Architecture
The codebase is strictly modularized into four distinct Python classes to ensure high maintainability and prevent component coupling:

* **`IrisApp.py`**: The main orchestration class and entry point. It controls the operational flow: triggering data ingestion, launching the model benchmarking routine, and deploying the user interface.
* **`DataLoader.py`**: Handles local dataset exploration (shape analysis, descriptive statistics) and generates exploratory visualizations via Matplotlib and Seaborn (Violin plots and Pairplots).
* **`ModelManager.py`**: Controls the machine learning workflow. It handles stratified data splitting, benchmarks multiple algorithms (Logistic Regression, KNN, Decision Trees, and SVM) via 10-fold Stratified Cross-Validation, trains the optimized model, and exposes the inference API.
* **`IrisGUI.py`**: A lightweight desktop graphical user interface built with Tkinter. Includes embedded exception handling to validate user inputs against non-numeric data types before model testing.
💻 Tech Stack & Libraries
  Development Environment: Visual Studio
  Core Language: Python 3
  Data Engineering & Visualization: Pandas, NumPy, Seaborn, Matplotlib
  Machine Learning Framework:Scikit-learn (Algorithms & Metrics Evaluation)
  GUI Deployment: Tkinter

  📊 Model Evaluation Results
The pipeline performs a robust 10-Fold Stratified Cross-Validation benchmark across multiple classification baselines:
* Logistic Regression (LR)
* K-Nearest Neighbors (KNN)
* Classification and Regression Trees (CART)
* Support Vector Classifier (SVC)

The final predictive state is backed by the optimized **Support Vector Classifier (SVC)**, outputting full metric transparency including Accuracy, Precision, Recall, and F1-score upon initialization.
