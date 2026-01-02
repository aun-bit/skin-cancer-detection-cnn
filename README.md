# Skin Cancer Detection using Deep Learning (CNN)

## Project Overview
Deep Learning project for skin cancer detection and prediction using Convolutional Neural Networks (CNN).

## Dataset
- **Name:** Melanoma Skin Cancer Dataset
- **Year:** 2023
- **Size:** 10,605 images
- **Classes:** Benign vs Malignant
- **Split:** 
  - Training: 8,165 images
  - Validation: 1,440 images
  - Test: 1,000 images

## Week 1 Progress ✅
- Environment setup complete
- Dataset downloaded and explored
- Data split into train/val/test sets
- Sample visualizations created

## Technologies
- Python 3.8+
- TensorFlow/Keras
- OpenCV
- Pandas, NumPy
- Matplotlib

## Project Timeline
- **Week 1:** Data preparation ✅
- **Week 2:** Baseline CNN model (in progress)
- **Week 3:** Model improvements
- **Week 4:** Final evaluation and documentation


## Week 1 Summary
**Completed:** December 29, 2025

### Achievements:
- ✅ Downloaded Melanoma dataset (10,605 images from 2023)
- ✅ Explored class distribution (benign vs malignant)
- ✅ Created train/validation/test splits (8165/1440/1000)
- ✅ Visualized sample images from both classes
- ✅ Confirmed dataset balance and quality

### Files Created:
- `week1_setup_and_exploration.ipynb` - Data exploration notebook

### Next Steps (Week 2):
- Build baseline CNN model
- Implement data augmentation
- Train and evaluate model

## Week 2 Progress ✅
**Completed:** January 1, 2025

### Achievements:
- ✅ Built baseline CNN model (3 convolutional layers)
- ✅ Trained for 10 epochs with data augmentation
- ✅ Achieved 90.5% test accuracy
- ✅ Created confusion matrix and evaluation metrics
- ✅ Saved trained model

### Results:
- Training Accuracy: 90.53%
- Validation Accuracy: 90.76%
- Test Accuracy: 90.50%
- Benign Recall: 92%
- Malignant Precision: 92%

### Files Created:
- `week2_baseline_model.ipynb` - Model training notebook
- `skin_cancer_model.h5` - Saved trained model


## Week 3 Progress ✅
**Completed:** January 2, 2026

### Achievements:
- ✅ Implemented transfer learning with MobileNetV2
- ✅ Trained model with pre-trained ImageNet weights
- ✅ Improved accuracy from 90.5% to 91.7%
- ✅ Created confusion matrix and evaluation
- ✅ Saved improved model

### Results:
- Training Accuracy: Higher than baseline
- Validation Accuracy: 91.40%
- Test Accuracy: 91.70% (+1.20% improvement)
- Benign Recall: 96%
- Malignant Precision: 95%

### Model Comparison:
| Model | Test Accuracy | Improvement |
|-------|---------------|-------------|
| Week 2 - Baseline CNN | 90.50% | - |
| Week 3 - MobileNetV2 | 91.70% | +1.20% |

### Files Created:
- `week3_transfer_learning.ipynb` - Transfer learning notebook
- `skin_cancer_mobilenet.h5` - Trained MobileNetV2 model

### Next Steps (Week 4):
- Add Grad-CAM visualization
- Final model evaluation
- Create project demo
- Complete documentation
