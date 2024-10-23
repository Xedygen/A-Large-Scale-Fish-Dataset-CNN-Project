# A-Large-Scale-Fish-Dataset-CNN-Project
This project aims to classify various fish species using a deep learning model built with TensorFlow and Keras. The dataset consists of a large-scale collection of fish images organized into different classes. The model utilizes convolutional neural networks (CNNs) to learn from the images and make predictions on unseen data.

**Notebook Link:** https://www.kaggle.com/code/xedygen/global-ai-hub-project-latest-version-burak-basol    

**Dataset Link:** https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset

**Instagram:** @xedygen

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)

## Overview

The project implements a CNN using TensorFlow to classify images of fish into nine different classes. Key aspects of the implementation include data preprocessing, model architecture, training with callbacks for resource management, and evaluation of model performance through accuracy, loss, confusion matrix, and classification report visualizations.

## Dependencies

Ensure you have the following libraries installed:

```bash
pip install numpy pandas tensorflow matplotlib seaborn scikit-learn
```

## Dataset

The dataset contains images of fish, and each class is represented by a folder containing images of that specific fish type.  

**Dataset Link:** https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset

## Model Architecture

The model is a simple sequential CNN architecture defined as follows:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])
```
**Input Layer:** Accepts images of size 224x224 with three color channels (RGB).    

**Convolutional Layers:** Extract features from the images using filters of size 3x3.    

**MaxPooling Layers:** Downsample the feature maps to reduce spatial dimensions.    

**Flatten Layer:** Converts the 2D feature maps into a 1D vector for the dense layers.    

**Dense Layers:** Fully connected layers that learn to classify the features into one of the nine classes

![Image is not loading.](visulization_sample.png)

## Training

The model is compiled and trained using the following configuration:

```python
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
        ClearMemory()  
    ]
)
```

**Callbacks**

**EarlyStopping:** Stops training if the validation loss does not improve for 5 epochs.    

**ReduceLROnPlateau:** Reduces the learning rate if the validation loss does not improve for 3 epochs.    

**ClearMemory:** Custom callback to clear the session and collect garbage to manage memory effectively.    

## Evaluation

The model's performance is evaluated using confusion matrices and classification reports.

**Confusion Matrix**
```python
confusion_mtx = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.show()
```

**Classification Report**

```python
classification_rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

classification_df = pd.DataFrame(classification_rep).transpose()

plt.figure(figsize=(10, 6))
sns.heatmap(classification_df.iloc[:-1, :-1], annot=True, cmap='Blues', fmt='.2f')
plt.title('Classification Report', fontsize=16)
plt.ylabel('Classes', fontsize=14)
plt.xlabel('Metrics', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
```

## Results

The trained model is saved for future use in kaggle notebook.    

**Trained Model:** https://www.kaggle.com/code/xedygen/global-ai-hub-project-latest-version-burak-basol/output

```python
model.save('model.keras')
```

## Usage
To use this model for predictions, load the saved model and pass in new images for classification.

```python
model = tf.keras.models.load_model('model.keras')
predictions = model.predict(new_images)
```
