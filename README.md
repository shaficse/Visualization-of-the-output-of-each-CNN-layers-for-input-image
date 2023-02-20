# Visualization-of-the-output-of-each-CNN-layers-for-input-image
 
The primary aim of this small project to visualize the representaiton of each of the CNN layer's output.
 
Convolutional neural networks are highly effective for image classification and recognition tasks, as they employ a variety of filters in each layer to learn features from the training images. The features that are learned at each convolutional layer tend to vary significantly. It has been observed that the earlier layers tend to capture low-level features such as edges, orientation, and colors in the image. As the number of layers increases, the CNN is able to capture more high-level features, which aid in distinguishing between different classes of images. To better understand how convolutional neural networks learn spatial and temporal dependencies in images, it is possible to visualize the different features that are captured at each layer.

## In step 1:
The dataset is loaded and the data is preprocessed. This is achieved by using Keras ImageDataGenerator to load the training and validation images into a data generator. The class mode is set as 'Binary' and a batch size of 20 is used. The size of the image is fixed at (150, 150) as the target size.

## In step 2:
The model's architecture is defined by adding two-dimensional convolutional layers, max-pooling layers, and a dense classification layer. The final dense layer uses the Sigmoid activation function since the problem is a two-class classification problem.

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 148, 148, 32)      896       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 74, 74, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 72, 72, 64)        18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 36, 36, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 34, 34, 128)       73856     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 17, 17, 128)       0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 15, 15, 128)       147584    
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 7, 7, 128)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 6272)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               3211776   
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 513       
=================================================================
Total params: 3, 453, 121
Trainable params: 3, 453, 121
Non-trainable params: 0
```

## In step 3:
In this step, the model is compiled with binary cross-entropy as the loss function, RMSprop as the optimizer, and accuracy as the metric. The model is then trained using the cat and dog dataset.

## In Step 4: 
The process of visualizing the output of each layer by considering a test image is described. An image that was not used for training is selected from the test dataset.

## Observation
The early layers in a neural network tend to capture and preserve the basic features of the input image, and are generally more easy to interpret. As the layers become deeper, the features extracted become more abstract and class-specific, focusing less on general features of the image.

## Reference
https://keras.io/api/models/model/
https://www.kaggle.com/c/dogs-vs-cats
https://www.geeksforgeeks.org/introduction-convolution-neural-network/
https://www.tensorflow.org/guide/data_performance
https://www.geeksforgeeks.org/visualizing-representations-of-outputs-activations-of-each-cnn-layer/
