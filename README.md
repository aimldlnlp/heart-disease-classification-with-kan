# Heart Disease Classification with Kolmogorov-Arnold Networks (KAN)

## Project Description
This project implements Kolmogorov-Arnold Networks (KAN) for the classification of heart disease using a publicly available dataset. The goal is to leverage the KAN model, a novel interpretable machine learning approach, to accurately predict the presence of heart disease based on patient data. The project includes data preprocessing, model training with and without hyperparameter tuning, evaluation using multiple metrics, and visualization of results.

## Features
- Implementation of KANClassifier from the `imodelsx` library for heart disease classification.
- Data preprocessing including scaling and train-test splitting.
- Hyperparameter tuning for KAN model parameters such as hidden layers, activation, entropy, ridge, and spline order.
- Model evaluation using accuracy, Cohen's kappa score, confusion matrix, and classification report.
- Visualization of data distributions and ROC curve for model performance.
- GPU support for accelerated model training (using CUDA).

## Installation Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/aimldlnlp/heart-disease-classification-with-kan.git
   cd heart-disease-classification-with-kan
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) If you have a CUDA-enabled GPU, ensure you have the appropriate CUDA drivers installed to leverage GPU acceleration.

## Usage Examples
1. Open the Jupyter notebook `heart_disease_classification_with_kan.ipynb`.
2. Run the notebook cells sequentially to:
   - Load and visualize the heart disease dataset.
   - Preprocess the data and split into training and testing sets.
   - Train the KAN model with default parameters.
   - Perform hyperparameter tuning to optimize model performance.
   - Evaluate the model using various metrics.
   - Visualize the ROC curve to assess classification performance.
