# Deep Learning for Image Classification Assessment

## Overview
This assessment involves building an image classifier using Keras and Convolutional Neural Networks (CNNs) for the Fashion MNIST dataset. The dataset includes 10 labels representing different clothing types, with 28x28 grayscale images. The task is to create a model that can classify these images accurately.

## Dataset Description
The Fashion MNIST dataset consists of:
- **Training set:** 60,000 images
- **Test set:** 10,000 images

### Labels and Descriptions:
0. T-shirt/top  
1. Trouser  
2. Pullover  
3. Dress  
4. Coat  
5. Sandal  
6. Shirt  
7. Sneaker  
8. Bag  
9. Ankle boot  

## Tasks
### Task 1: Download the Dataset
```python
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
```

### Task 2: Visualize the Data
```python
import matplotlib.pyplot as plt

label_description = {
    0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 
    5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'
}

index = 0  # Change this as needed
image = x_train[index]
label = y_train[index]

plt.imshow(image, cmap='gray')
plt.title(f'Label: {label} - {label_description[label]}')
plt.show()
```

### Task 3: Normalize the Data
```python
x_train = x_train / 255.0
x_test = x_test / 255.0
```

### Task 4: Reshape the Data
```python
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
```

### Task 5: One-Hot Encoding the Labels
```python
from keras.utils import to_categorical

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```

### Task 6: Build and Compile the Model
```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (4,4), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()
```

### Train the Model
```python
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
```

### Task 8: Evaluate the Model
```python
from sklearn.metrics import classification_report
import numpy as np

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred_classes, target_names=[label_description[i] for i in range(10)]))
```

### Model Summary:
	•	Model: Sequential
	•	Layers: Conv2D, MaxPooling2D, Flatten, Dense
	•	Total Parameters: 591,786
	•	Epochs: 10
	•	Training Accuracy: Up to 95.75%
	•	Validation Accuracy: Up to 91.11%

### Performance Metrics

	•	Accuracy: 90%
	•	Precision, Recall, F1-Score: Detailed in the classification report for each class.

 ### Conclusion

This assessment guides you through building a CNN model to classify images from the Fashion MNIST dataset. By following the steps, you will preprocess the data, build and compile a model, train it, and evaluate its performance using various metrics.
