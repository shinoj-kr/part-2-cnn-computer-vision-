## Dataset Source
https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing

## Task 1: Problem Identification

The given dataset represents a Multi-Class Image Classification problem in the field of Computer Vision.

The dataset contains four image categories:

- dent
- normal
- scratch
- stain

Each image belongs to only one class, and the goal of the CNN model is to correctly classify the manufacturing surface image into its corresponding defect category.

This problem is classified as Image Classification because:
- The model predicts a single label for each image.
- There are no bounding boxes for object localization.
- There are no segmentation masks for pixel-level classification.

Therefore, a CNN-based image classification approach is appropriate for this dataset.

## Task 2: Dataset Exploration

### Number of Classes

The dataset contains four classes:

- dent
- normal
- scratch
- stain

### Number of Images per Class

Each class contains 120 images.

dent: 120 images
normal: 120 images
scratch: 120 images
stain: 120 images

### Sample Images from Each Class

Sample images from the dataset are visualized to understand the defect patterns present in each category.

Observations:

- dent images contain circular dent-like defects.
- normal images represent clean product surfaces without major defects.
- scratch images contain line-shaped scratch marks.
- stain images contain colored stain-like spots.

The visual differences between classes are clear, making the dataset suitable for CNN-based image classification.

### Image Dimensions

A sample image from the dataset is analyzed to understand the image dimensions.

Image Dimensions

Image Shape: (96, 96, 3)

96 - image height
96 - image width
3 - RGB color channels

### Dataset Balance Observation

The dataset is balanced because all four classes contain an equal number of images. This helps the CNN model learn fairly from all categories and reduces prediction bias toward any single class.

## Task 3: Image Preprocessing

### Resizing Images to a Fixed Size

All images are processed using a fixed image size of 96 × 96 pixels. Resizing is included as part of the standard preprocessing pipeline to ensure uniformity and future compatibility, even though the dataset already contained fixed-size images.

### Normalizing Pixel Values

The image pixel values are normalized by dividing all pixel values by 255.0. Before normalization, pixel values ranged from 0 to 255. After normalization, the values are scaled to a smaller range between approximately 0.17 and 1.0. Normalization helps the CNN model train more efficiently and improves numerical stability during learning.

### Splitting into Training and Testing Sets

The dataset is divided into training and testing sets using an 80:20 ratio.

- 80% of the images are used for training the CNN model.
- 20% of the images are used for testing the model performance on unseen data.

The random_state=42 parameter is used to ensure stable and reproducible dataset splitting across multiple runs.

Output:

- Training Images: 384
- Testing Images: 96

### Applying Data Augmentation

Light data augmentation techniques are configured using TensorFlow’s ImageDataGenerator.

The following augmentation techniques are applied:

- Small image rotations (rotation_range=10)
- Horizontal flipping (horizontal_flip=True)

Data augmentation helps improve model generalization by exposing the CNN model to slightly modified versions of the training images. Since the dataset is already balanced and visually clear, only minimal augmentation is used.

## Task 4: CNN Model Creation

A Convolutional Neural Network (CNN) model is created using TensorFlow/Keras Sequential API for multi-class image classification.

The CNN architecture includes:

- Convolution layers (Conv2D)
- ReLU activation functions
- Max Pooling layers (MaxPooling2D)
- Flatten layer
- Dense (fully connected) layers
- Softmax output layer

### CNN Architecture Summary

1. First Convolution Layer:
   - 32 filters
   - Kernel size: (3 × 3)
   - ReLU activation

2. First Max Pooling Layer:
   - Pool size: (2 × 2)

3. Second Convolution Layer:
   - 64 filters
   - Kernel size: (3 × 3)
   - ReLU activation

4. Second Max Pooling Layer:
   - Pool size: (2 × 2)

5. Flatten Layer:
   - Converts feature maps into a one-dimensional vector

6. Dense Layer:
   - 128 neurons
   - ReLU activation

7. Output Layer:
   - 4 output neurons
   - Softmax activation function for multi-class classification

## Task 5 — Model Training and Evaluation

The CNN model was trained using the Adam optimizer and sparse categorical crossentropy loss function for multi-class image classification.

### Model Training

The CNN model is trained using the training dataset for 10 epochs with a batch size of 32.

Training Configuration:

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Epochs: 10
- Batch Size: 32
- Validation Split: 20%

### Training Observation

During training, the model accuracy improved steadily while the loss value decreased gradually.

Final Results:

- Training Accuracy: approximately 98%
- Validation Accuracy: approximately 76%
- Training Loss: approximately 0.14
- Validation Loss: approximately 0.52

The model successfully learned the visual patterns of different defect categories. A slight gap between training and validation accuracy suggests mild overfitting, which is common for small datasets.

### Accuracy and Loss Curves

Training and validation accuracy/loss curves are plotted to visualize the learning performance of the CNN model over multiple epochs.

Observations:

- Training accuracy increased steadily during training.
- Training loss decreased gradually across epochs.
- Validation accuracy also improved over time, indicating that the model learned meaningful image features.
- A small gap between training and validation accuracy suggests mild overfitting, which is common in small datasets.

The accuracy and loss curves confirm that the CNN model successfully learned the defect classification patterns from the dataset.

### Testing Performance

The trained CNN model is evaluated using the testing dataset containing unseen images.

Testing Results:

- Test Accuracy: approximately 83%
- Test Loss: approximately 0.46

The model achieved good performance in identifying different defect categories from the image dataset. The testing accuracy indicates that the CNN model was able to generalize reasonably well on unseen data.

### Confusion Matrix Analysis

A confusion matrix is generated to analyze the prediction performance of the CNN model across all defect categories.

Observations:

- The model classified normal and stain images very accurately with almost no misclassifications.
- Minor confusion is observed between dent and scratch classes.
- The scratch category showed the highest number of misclassifications, indicating that some scratch patterns visually resemble  
  other defect types. 

Overall, the confusion matrix shows that the CNN model learned the major visual defect patterns effectively while experiencing slight difficulty distinguishing certain complex scratch features.

### Sample Predictions on Test Images

Sample predictions are generated using unseen test images from the dataset.

For each test image:

- The actual defect category is displayed.
- The CNN model prediction is displayed.

The sample predictions showed that the model is able to correctly identify most defect categories, especially normal and stain images. Minor misclassifications are observed in certain scratch images due to visual similarity between some defect patterns.

## TASK 6 : CNN Concept Explanation

### What is Convolution?

Convolution is a mathematical operation used in CNNs to detect important visual patterns from images, such as edges, textures, shapes, and defects. A small filter (kernel) moves across the image and extracts useful features by performing element-wise multiplication between the filter and image pixels. This helps the CNN learn meaningful image features automatically during training.

### Why is Pooling Used?

Pooling is used in CNNs to reduce the size of feature maps while preserving the most important information. It helps to decrease computational complexity, reduces memory usage, and improves model efficiency. Pooling also helps to reduce overfitting by simplifying feature representations. In this Max Pooling is used to retain the strongest visual features from the images.

### Why is ReLU Commonly Used in CNNs?

ReLU (Rectified Linear Unit) is an activation function widely used in CNNs because it helps the model learn faster and improves computational efficiency. ReLU converts all negative values to zero while keeping positive values unchanged. This introduces non-linearity into the neural network, allowing the CNN to learn complex image patterns effectively. ReLU also helps reduce the vanishing gradient problem during training.

### Why are CNNs Better than Regular Feed-Forward Networks for Image Data?

CNNs are specifically designed for image processing tasks. They automatically learn spatial features such as edges, textures, shapes, and patterns directly from images. Unlike regular feed-forward neural networks, CNNs preserve the spatial structure of images using convolution and pooling operations. This makes CNNs more efficient, accurate, and scalable for computer vision problems. CNNs also require fewer parameters compared to fully connected networks when working with image data, reducing computational complexity and improving learning efficiency.

## Task 7: Business Use Case Mapping

This CNN-based computer vision solution can be used in the manufacturing industry for automated surface defect detection and quality inspection. Manufacturing companies can use cameras and CNN models to automatically identify defects such as dents, scratches, and stains on product surfaces during production. This helps improve product quality, reduce manual inspection effort, minimize human error, and increase manufacturing efficiency. 

Automated defect detection systems can be applied in industries such as:

- Automobile manufacturing
- Metal surface inspection
- Electronic device production
- Packaging quality control

Using CNN-based defect detection can help manufacturers detect faulty products early and maintain consistent quality standards.