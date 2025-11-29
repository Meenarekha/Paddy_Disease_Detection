
# Paddy Disease Detection Using Deep Learning (CNN)

## Project Overview

This project aims to build a low-cost, AI-powered tool to detect and classify paddy leaf diseases using deep learning techniques. By leveraging Convolutional Neural Networks (CNNs) 
and real-world paddy leaf image data, the model helps identify crop diseases early, enabling farmers to take preventive actions.

Goal: Improve paddy crop yield through early disease detection using smartphone-captured leaf images.

## Why This Project?

- Agriculture is vital to global food security—especially in rice-dependent regions.
- Farmers lack access to rapid, expert-level crop diagnostics.
- Traditional methods are manual, slow, and error-prone.
- Our goal: use AI + Computer Vision to enable automated disease detection.

## Dataset Description

- Total Images: 16,225
- Classes: 13 (12 diseases + 1 healthy)
- Resolution: 480×640 pixels
- Split: 75% training, 25% testing
- Source: Real-world paddy fields in India

## CNN Architecture

Input Image → Conv2D → MaxPooling → Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense (128) → Dropout → Output Layer (Softmax)

| Layer               | Description                                      |
|--------------------|--------------------------------------------------|
| Conv2D Layers       | Extract low, mid, and high-level image features |
| MaxPooling2D Layers | Reduce spatial dimensions, retain key features  |
| Flatten             | Convert 3D feature maps to 1D vector             |
| Dense (128)         | Fully connected for high-level reasoning         |
| Dropout             | Prevent overfitting                              |
| Dense Output (Softmax) | Output prediction probabilities             |

## Model Comparison – Ablation Study

| Model           | Accuracy | F1-Score | Remarks            |
|-----------------|----------|----------|--------------------|
| CNN             | 51.48%   | 0.64     | Best performer     |
| SVM             | 36.35%   | N/A      | Moderate accuracy  |
| Random Forest   | 33.66%   | N/A      | Lower performance  |

CNNs excel due to their ability to learn spatial features, while traditional ML models require manual feature engineering.

## Societal Impact

This AI system brings real-world value to agriculture by:

- Empowering rural farmers with smartphone-based tools
- Reducing pesticide overuse with targeted intervention
- Minimizing crop loss and boosting food security
- Bridging the digital divide in rural communities

## Technical Highlights

- Implemented in Python using TensorFlow/Keras
- Image classification with CNNs
- Compared ML and DL models (SVM, RF, CNN)
- Architecture optimized for real-world image recognition
- Accepts leaf images as input for prediction

## Conclusion

CNN-based models offer a scalable, accurate, and automated solution to paddy disease classification. This project contributes to the intersection of AI and sustainable agriculture, 
offering long-term benefits to rural farming communities.

Ideal for integration with mobile apps for real-time crop health monitoring.


