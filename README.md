🩺 Deep Learning Project for Skin Cancer Detection and Prediction Using CNN
1.	Project Overview
Skin cancer, particularly melanoma, is one of the most dangerous forms of cancer if not detected early. This project presents a deep learning–based approach for automatic skin cancer detection using Convolutional Neural Networks (CNNs). The system classifies dermoscopic skin images into benign and malignant categories to assist in early diagnosis.
The project focuses purely on the machine learning pipeline including data preprocessing, model training, evaluation, and inference, without any frontend or deployment components.
________________________________________
2.	 Authors
•	Aun Mustansar Hussain
•	M Zohaib Shahid
Degree: MS Data Science
University: Superior University, Gold Campus Lahore
________________________________________
3.	Objectives
•	To develop a CNN-based deep learning model for skin cancer detection
•	To classify skin lesion images into benign and malignant categories
•	To evaluate model performance using advanced metrics beyond accuracy
•	To demonstrate model inference on unseen images
________________________________________
4.	Dataset Description
•	Dataset Type: Dermoscopic skin lesion images
•	Classes:
o	Benign
o	Malignant
•	Total Images: ~10,605
o	Training images: 9,605
o	Testing images: 1,000
•	Directory Structure:
data/melanoma_cancer_dataset/
 ├── train/
 │   ├── benign
 │   └── malignant
 └── test/
     ├── benign
     └── malignant
Note: Data imputation is not required as the dataset consists of image files.
________________________________________
5.	Model Architecture
The model is a custom CNN architecture implemented using TensorFlow/Keras, consisting of:
•	Convolutional layers (Conv2D)
•	Batch Normalization
•	Max Pooling
•	Dropout for regularization
•	Fully connected (Dense) layers
•	Sigmoid activation for binary classification
Total Parameters: ~11.1 million
________________________________________
6.	Technologies Used
•	Programming Language: Python
•	Deep Learning Framework: TensorFlow / Keras
•	Libraries: NumPy, OpenCV, scikit-learn, Matplotlib
•	Environment: Conda (CPU-based execution)
GPU is optional. The project runs successfully on CPU.
________________________________________
7.	Model Training
•	Images are normalized using rescaling
•	Data augmentation is applied to improve generalization
•	Binary cross-entropy loss is used
•	Adam optimizer is employed
•	Training history is saved for analysis
________________________________________
8.	Model Evaluation
The model is evaluated using multiple performance metrics to ensure reliability:
•	Accuracy: 90%
•	Precision: 0.8931
•	Recall: 0.9020
•	F1-score: 0.8975
•	ROC-AUC: 0.9637
A confusion matrix and classification report are also generated for detailed analysis.
________________________________________
9.	Model Inference
The trained model can predict the class of a new, unseen skin image.
Example Output:
Prediction score: 1.0000
Predicted class: Malignant
Inference is implemented in a separate script, ensuring clear separation from training and evaluation.
________________________________________
10.	Project Structure
skin-cancer-detection-cnn/
 ├── data/
 ├── utils/
 │   ├── data_loader.py
 │   └── metrics.py
 ├── train.py
 ├── model.py
 ├── evaluate.py
 ├── inference.py
 ├── transfer_learning.py
 ├── requirements.txt
 ├── README.md
________________________________________
11.	How to Run the Project
1. Install dependencies
pip install -r requirements.txt
2. Train the model
python train.py
3. Evaluate the model
python evaluate.py
4. Run inference
python inference.py
________________________________________
12.	Key Highlights
•	Clear separation of training, evaluation, and inference
•	Uses advanced evaluation metrics (not accuracy only)
•	CPU-compatible implementation
•	Clean and modular code structure
•	Designed for academic learning and demonstration
________________________________________
13.	Conclusion
This project demonstrates the effective use of deep learning for medical image classification. The CNN model achieves strong performance in detecting malignant skin lesions and can serve as a foundation for further research or real-world clinical decision support systems.
________________________________________
