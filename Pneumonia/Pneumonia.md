---
layout: default
title: Classifying Pneumonia
---

# Fully Convolutional Networks and Pneumonia Classification
For this project, I will be performing a binary classification on x-ray image data. Mainly this exercise will run through how to quickly set up an image classification pipeline
using high level Keras tools. This problem helps showcase helpful techniques to deal with large image datasets with different dimensions, as well as how to turn our raw data into trainable batches. I will also run through helpful ways to evaluate medical data and give some intuition on architecture design.

## Dataset: Chest X-rays
Our data comes courtesy of UCSD and is based off the paper "*Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*" by Kermany et al. The Data:

Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “*Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*”, Mendeley Data, V2, doi: 10.17632/rscbjbr9sj.2

Provides thousands of chest x-rays from Guangzhou Women and Children’s Medical Center, where patients between the age of 1 and 5 were examined for Pneumonia-related diseases. The images were graded by up to 3 physicians, determining whether each patient could be diagnosed with some form of pneumonia, or as 'healthy.' The question we are presented with is thus "Can machine learning help provide insight into a proper diagnosis?

To answer this question, I will seek to perform a streamlined initial analysis by correctly separating sick from healthy patients. More specifically, I will construct a neural network to predict whether or not an x-ray indicates a healthy patient or one afflicted with pneumonia. 


```python
from PIL import Image
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
```


```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
```

## Exploring our Data

To begin, let's take a look at our data. First, we'll open a healthy child's x-ray for reference


```python
Image.open('./Data/chest_xray/train/normal/IM-0147-0001.jpeg').convert('RGB')
```




![png](output_4_0.png)



We can clearly see a full x-ray image of the upper torso, and the spine, heart, ribs, and shoulder are all easily visible. Of course, what we are really interested in are the lungs--you can faintly see their outline and some bronchi behind the ribs. Our network will need to pinpoint the lungs (and how healthy they are) specifically so we can give our patients a proper diagnosis.

You may have noticed that the x-ray seems quite large. In fact, the bulk of our images are well over 1000x1000 pixels in size! This size is definitely overkill for our classification task (not to mention the strain it puts on our poor GPU), so we can definitely afford to cut down the dimensions substantially. Thankfully, Pillow's **thumbnail** lets us easily reduce the image size (I'll use a factor of 1/8 per side), and includes a nice **Antialias** processing to help keep the image quality in the reduced dimensions.

Now let's take a look at a sick patient's x-ray


```python
Image.open('./Data/chest_xray/train/pneumonia/person4_bacteria_14.jpeg').convert('RGB')
```




![png](output_6_0.png)



This patient looks quite a bit worse than the last. You can easily see the cloudy, inflamed lungs (careful though--the white lump on the right side is the heart--hopefully our network will be able to tell the difference!).

Looking at this image, we see there's a bit of a problem with our data. *The aspect ratios of the two previous images are totally different!* In fact, our entire dataset is a mishmash of various dimensions and sizes. While it's (hopefully) easy for a person to see that the two images above are of the same object (namely, the chest), our network may have some problems depending on how we deal with these discrepancies.

One option to deal with the matter would be to force all images into the same dimensions (say, a 216x216 square). While the uniformity would be easy to feed into our network, the result of this strategy would be that each image would be stretched and distorted in an unpredictable manner. Naturally, this would make it very difficult for the machine to be able to learn any helpful features from the data, and our accuracy would take a nosedive. (Even for the human eye, it can be a problem. Try it yourself: take your picture and stretch it horizontally or vertically--your face becomes practically unrecognizeable)

Rather than take the prior approach, I will instead seek to keep the images in their original aspect ratios (after size reduction). Then, I will utilize a **Fully Convolutional Network** (FCN) to handle the images. FCNs are similar to the standard Convolutional Network with one exception: *there are no fully connected layers!* Instead, FCNs stack convolutional blocks continually from input to output until the desired dimensions are achieved. This allows FCNs to handle input data of different dimensions, as the number of neuron connections (normally constant due to the fully connected layer constraints) can vary based on input size. With this architecture in mind, let's continue to process our images.

## Image Preprocessing and Data Generation

Right now, our images are still in their original forms, all stored as .jpeg files in the repository folder. We would like to implement some of the changes we discussed earlier and store them in a data format that we can feed into our FCN. We could simply process and store the whole dataset in one chunk, putting all the images in a single tensor or numpy array and then cutting up batches as need be. However, the size of these images is pretty massive as it is, and would needlessly tax our system to produce. Furthermore, setting up the batches, shuffling, and splitting procedures in this way is a bit of an ugly way to proceed, and goes against the high-level structure of Keras.

As such, I will employ a generator to take care of the image processing and batch storing all in one go. We can then pass this along to our Keras model, which will be able to use the generator to acquire the necessary data for each run. First, start with some preliminary parameters and paths:


```python
# Determine training parameters
EPOCHS = 30
BATCHSIZE = 32
VALSPLIT = 0.9

path_ntr = './Data/chest_xray/train/normal/'
path_ptr = './Data/chest_xray/train/pneumonia/'

path_ntst = './Data/chest_xray/test/normal/'
path_ptst = './Data/chest_xray/test/pneumonia/'
```

To keep track of the images, I'm going to store the filenames and labels (0 for healthy patients, 1 for those with pneumonia) in 2 dictionaries. Note how I'm also employing my chosen validation split here, so we can already set aside those images and keep them out of the training loop.


```python
norm_list = listdir(path_ntr)
pneu_list = listdir(path_ptr)
                    
ntrain_n = int(len(norm_list) * VALSPLIT)
ntrain_p = int(len(pneu_list) * VALSPLIT)

filenames = {'train': [], 'val': []}
labels = {}

# Loop through the normal xrays and store a label of '0'
ptr = 0
for file in norm_list:
    if ptr < ntrain_n:
        filenames['train'].append(file)
    else:
        filenames['val'].append(file)
    labels[file] = 0
    ptr += 1

# Now loop through the pneumonia xrays and give a label of '1'
ptr = 0
for file in pneu_list:
    if ptr < ntrain_p:
        filenames['train'].append(file)
    else:
        filenames['val'].append(file)
    labels[file] = 1
    ptr += 1
```

I also want to make a final dictionary for our test images. I'm keeping them separate as they won't be needed during the training runs.


```python
norm_list_t = listdir(path_ntst)
pneu_list_t = listdir(path_ptst)
                    
ntest_n = len(norm_list_t)
ntest_p = len(pneu_list_t)

test_names = {'test': []}
test_labels = {}

for file in norm_list_t:
    test_names['test'].append(file)
    test_labels[file] = 0

for file in pneu_list_t:
    test_names['test'].append(file)
    test_labels[file] = 1
```

Next I set up the data generator as a keras sequence. This will handle training, testing, and validation batches, and will shuffle the images on each epoch to avoid unnecessary bias.

The image batch processing will be conducted as follows: First images will be opened with 3 color channels and then resized by (1/8, 1/8). The values will be stored in a numpy array and each rescaled to be between 0 and 1 for gradient stability. Finally, we will 0-pad the images until they are the same size as the largest image for that particular batch (this allows our gradients to have the same dimensions for each particular batch.


```python
class Generator(Sequence):
    def __init__(self, IDs, labels, path_norm, path_pneu, batch_size=32, n_channels = 3, shuffle=True):
        self.IDs = IDs
        self.labels = labels
        self.path_norm = path_norm
        self.path_pneu = path_pneu
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.shuffle_batches()
    
    def __len__(self):
        '''
        For Keras data generator processing, return the number of batches per epoch
        '''
        return int(len(self.IDs) // self.batch_size)
    
    def __getitem__(self, idx):
        '''
        For Keras generator processing, this returns the current batch to the model with
        the appropriate labels.
        '''
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_IDs = [self.IDs[i] for i in indices]
        X, y = self.make_batch(batch_IDs, self.n_channels)
        
        return X, y
    
    def shuffle_batches(self):
        '''
        After each epoch, we will shuffle the images so we get different batches each time
        '''
        self.indices = np.arange(len(self.IDs))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def make_batch(self, batch_IDs, n_channels):
        '''
        This handles our processing and batch production. Returns a numpy array of
        size [batch_size, im_size, im_size, n_channels]
        
        batch_IDs (list): list of image filenames for the data
        n_channels (int): number of channels in each image
        '''
        paths = [self.path_norm, self.path_pneu]

        y = np.empty((self.batch_size), dtype=int)
        
        Xtemp = {}
        dims = 0
        
        # Open each image, resize to a thumbnail, and rescale to [0, 1]
        for i, ID in enumerate(batch_IDs):
            y_i = self.labels[ID]
            
            im = Image.open(paths[y_i] + ID).convert('RGB')
            im.thumbnail((im.width // 8, im.height // 8), Image.ANTIALIAS) # Antialias preserves jpeg quality
            if max(im.width, im.height) > dims:
                dims = max(im.width, im.height)
            
            im = np.asarray(im, dtype=np.float32)
            im /= 255.0
            Xtemp[i] = im
            y[i] = y_i
        
        # Now we find out the biggest image for our batch and pad the remaining ones to match
        X = np.empty((self.batch_size, dims, dims, n_channels), dtype=np.float32)
        for i in Xtemp:
            x_i = Xtemp[i]
            pad = ((0,dims-x_i.shape[0]), (0, dims-x_i.shape[1]), (0, 0))
            x_i = np.pad(x_i, pad_width=pad, mode='constant', constant_values=0)
            X[i, :, :, :] = x_i
        
        return X, tf.keras.utils.to_categorical(y, num_classes=2) # We'll return our labels as one-hot encoded variables
```

## Network Architecture

Now it's time to design and implement our FCN. FCN designs still follow the basics of creating a good CNN classifier; piece together several convolutional blocks, increasing the number
of features the network detects as you go. In each block, pool the results to downsize, apply your activation (somewhat optional, but I'm just using leaky ReLU to avoid neuron death), then normalize and push to the next block.

Input height and width dimensions are set to **None** to account for different sizes in each batch. The convolution kernels will start as 5x5 to learn larger features within the image, then slowly decrease to learn fine grain differences. The last two blocks use a 1x1 kernel--essentially, we are performing the same work as a fully connected layer but with the convolutional operation. Max Pooling is used in the initial layers to pick out the most imortant components of each convolutional output, while in the final block, we have a Global Average Pool to compress the final output into a single class prediction for each image. 


```python
def make_FCN():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 5), input_shape=[None, None, 3]))
    model.add(layers.MaxPooling2D())
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.MaxPooling2D())
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(128, (3, 3)))
    model.add(layers.MaxPooling2D())
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(256, (3, 3)))
    model.add(layers.MaxPooling2D())
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(512, (1, 1)))
    model.add(layers.MaxPooling2D())
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(2, (1, 1)))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Activation('softmax'))

    return model
```

## Weighing and Performance Metrics

Before we fit our FCN, we should note that there are roughly 3 times the number of sick patient x-rays as there are healthy x-rays


```python
fig = plt.figure()
patients = ['healthy', 'sick']
numbers = [ntrain_n, ntrain_p]
plt.bar(patients, numbers)
plt.show()
```


![png](output_18_0.png)


This class imbalance isn't too egregious, but we should account for it regardless. I'm a bit loathe to cut the sick samples down (particularly since we only have <10,000 training samples) and trying to artificially generate new healthy samples introduces too many extra factors for such a small imbalance. Instead, I'm going to weigh the classes so that the FCN penalizes different class losses in proportion to their numbers in the dataset.


```python
class_weight = {
    0: round((ntrain_n+ntrain_p)/(2*ntrain_n), 2),
    1: round((ntrain_n+ntrain_p)/(2*ntrain_p), 2)
}
```

Next we want to think about our performance metrics. Accuracy on its own might seem like a great way to determine how well our network is doing, but it doesn't tell the full story given our objective. In particular, since we are attempting to diagnose patients with a disease, we may like to know what proportion of positive (or negative) cases of pneumonia are correctly discovered. If, for example, we had a batch of 100 cases involving 98 healthy patients and 2 sick cases, by guessing that all are healthy, our network could achieve 98% accuracy *while completely missing both pneumonia patients.* 

To that end, medical research often uses Sensitivity and Specificity to evaluate a diagnosis. Sensitivity (true positives / true positives + false negatives) measures the proportion of sick patients that are correctly flagged by the network. Specificity (true negatives / true negatives + false positives) measures the proportion of healthy patients that are correctly designated. Thus, for instance, if we see our network displaying a low sensitivity, we can determine that the network may be putting out too many false negatives and a 'healthy' diagnosis wouldn't hold much water. Since Keras doesn't include Sensitivity and Specificity in its metrics library, I can code them from scratch and feed them into the model.


```python
def sensitivity(y_true, y_pred):
    y_true = tf.math.argmax(y_true, axis=1)
    y_pred = tf.math.argmax(y_pred, axis=1)
    yp_neg = 1 - y_pred # 1 if predicted data is negative
    tp = K.sum(tf.math.multiply(y_true,y_pred))
    fn = K.sum(tf.math.multiply(yp_neg,y_true))
    sensitivity = tp / (tp + fn)
    return sensitivity

def specificity(y_true, y_pred):
    y_true = tf.math.argmax(y_true, axis=1)
    y_pred = tf.math.argmax(y_pred, axis=1)
    y_neg = 1 - y_true # 1 if true data is negative
    yp_neg = 1 - y_pred # 1 if predicted data is negative
    fp = K.sum(tf.math.multiply(y_neg,y_pred))
    tn = K.sum(tf.math.multiply(y_neg,yp_neg))
    specificity = tn / (tn + fp)
    return specificity
```

## Compiling Model and Training

From here on out, all there is to do is train the model. Using Adam to update the weights, and with the appropriate loss and metrics selected, we can generate our batches and begin our learning.


```python
FCN = make_FCN()
FCN.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy', sensitivity, specificity],
            run_eagerly=True)
```


```python
training_generator = Generator(filenames['train'], labels, path_ntr, path_ptr)
val_generator = Generator(filenames['val'], labels, path_ntr, path_ptr)
test_generator = Generator(test_names['test'], test_labels, path_ntst, path_ptst)
```


```python
history = FCN.fit(x=training_generator,
                  epochs=EPOCHS,
                  verbose=2,
                  validation_data=val_generator,
                 class_weight=class_weight)
```

    Epoch 1/30
    147/147 - 123s - loss: 0.3695 - categorical_accuracy: 0.8259 - sensitivity: 0.7951 - specificity: 0.9324 - val_loss: 0.6700 - val_categorical_accuracy: 0.7500 - val_sensitivity: 1.0000 - val_specificity: 0.0137
    Epoch 2/30
    147/147 - 67s - loss: 0.2996 - categorical_accuracy: 0.8650 - sensitivity: 0.8348 - specificity: 0.9684 - val_loss: 0.9986 - val_categorical_accuracy: 0.2539 - val_sensitivity: 0.0000e+00 - val_specificity: 1.0000
    Epoch 3/30
    147/147 - 68s - loss: 0.2743 - categorical_accuracy: 0.8848 - sensitivity: 0.8592 - specificity: 0.9752 - val_loss: 0.9197 - val_categorical_accuracy: 0.2539 - val_sensitivity: 0.0000e+00 - val_specificity: 1.0000
    Epoch 4/30
    147/147 - 68s - loss: 0.2514 - categorical_accuracy: 0.8980 - sensitivity: 0.8751 - specificity: 0.9783 - val_loss: 0.4610 - val_categorical_accuracy: 0.8496 - val_sensitivity: 0.8720 - val_specificity: 0.7827
    Epoch 5/30
    147/147 - 68s - loss: 0.2393 - categorical_accuracy: 0.9045 - sensitivity: 0.8835 - specificity: 0.9789 - val_loss: 0.3891 - val_categorical_accuracy: 0.8750 - val_sensitivity: 0.8805 - val_specificity: 0.8536
    Epoch 6/30
    147/147 - 66s - loss: 0.2247 - categorical_accuracy: 0.9137 - sensitivity: 0.8946 - specificity: 0.9814 - val_loss: 0.3438 - val_categorical_accuracy: 0.8711 - val_sensitivity: 0.9072 - val_specificity: 0.7692
    Epoch 7/30
    147/147 - 67s - loss: 0.2115 - categorical_accuracy: 0.9199 - sensitivity: 0.9028 - specificity: 0.9813 - val_loss: 0.3706 - val_categorical_accuracy: 0.8809 - val_sensitivity: 0.8855 - val_specificity: 0.8629
    Epoch 8/30
    147/147 - 67s - loss: 0.1999 - categorical_accuracy: 0.9230 - sensitivity: 0.9069 - specificity: 0.9809 - val_loss: 0.3318 - val_categorical_accuracy: 0.8809 - val_sensitivity: 0.9092 - val_specificity: 0.7979
    Epoch 9/30
    147/147 - 66s - loss: 0.1911 - categorical_accuracy: 0.9273 - sensitivity: 0.9122 - specificity: 0.9817 - val_loss: 0.3052 - val_categorical_accuracy: 0.8906 - val_sensitivity: 0.9400 - val_specificity: 0.7458
    Epoch 10/30
    147/147 - 66s - loss: 0.1856 - categorical_accuracy: 0.9296 - sensitivity: 0.9161 - specificity: 0.9796 - val_loss: 0.3131 - val_categorical_accuracy: 0.8848 - val_sensitivity: 0.9248 - val_specificity: 0.7710
    Epoch 11/30
    147/147 - 66s - loss: 0.1754 - categorical_accuracy: 0.9362 - sensitivity: 0.9237 - specificity: 0.9816 - val_loss: 0.4070 - val_categorical_accuracy: 0.8711 - val_sensitivity: 0.8728 - val_specificity: 0.8620
    Epoch 12/30
    147/147 - 66s - loss: 0.1660 - categorical_accuracy: 0.9384 - sensitivity: 0.9247 - specificity: 0.9872 - val_loss: 0.2896 - val_categorical_accuracy: 0.8965 - val_sensitivity: 0.9433 - val_specificity: 0.7632
    Epoch 13/30
    147/147 - 66s - loss: 0.1588 - categorical_accuracy: 0.9409 - sensitivity: 0.9295 - specificity: 0.9843 - val_loss: 0.2757 - val_categorical_accuracy: 0.8945 - val_sensitivity: 0.9433 - val_specificity: 0.7542
    Epoch 14/30
    147/147 - 66s - loss: 0.1553 - categorical_accuracy: 0.9415 - sensitivity: 0.9303 - specificity: 0.9836 - val_loss: 0.2909 - val_categorical_accuracy: 0.9004 - val_sensitivity: 0.9298 - val_specificity: 0.8106
    Epoch 15/30
    147/147 - 66s - loss: 0.1474 - categorical_accuracy: 0.9469 - sensitivity: 0.9360 - specificity: 0.9868 - val_loss: 0.3841 - val_categorical_accuracy: 0.8887 - val_sensitivity: 0.8880 - val_specificity: 0.8916
    Epoch 16/30
    147/147 - 66s - loss: 0.1427 - categorical_accuracy: 0.9464 - sensitivity: 0.9358 - specificity: 0.9863 - val_loss: 0.2774 - val_categorical_accuracy: 0.8945 - val_sensitivity: 0.9380 - val_specificity: 0.7710
    Epoch 17/30
    147/147 - 66s - loss: 0.1358 - categorical_accuracy: 0.9515 - sensitivity: 0.9431 - specificity: 0.9854 - val_loss: 0.2907 - val_categorical_accuracy: 0.9121 - val_sensitivity: 0.9322 - val_specificity: 0.8477
    Epoch 18/30
    147/147 - 66s - loss: 0.1325 - categorical_accuracy: 0.9522 - sensitivity: 0.9438 - specificity: 0.9858 - val_loss: 0.4811 - val_categorical_accuracy: 0.8691 - val_sensitivity: 0.8884 - val_specificity: 0.8103
    Epoch 19/30
    147/147 - 66s - loss: 0.1273 - categorical_accuracy: 0.9541 - sensitivity: 0.9453 - specificity: 0.9874 - val_loss: 0.2621 - val_categorical_accuracy: 0.9023 - val_sensitivity: 0.9457 - val_specificity: 0.7779
    Epoch 20/30
    147/147 - 65s - loss: 0.1213 - categorical_accuracy: 0.9592 - sensitivity: 0.9515 - specificity: 0.9892 - val_loss: 0.2944 - val_categorical_accuracy: 0.9199 - val_sensitivity: 0.9375 - val_specificity: 0.8650
    Epoch 21/30
    147/147 - 66s - loss: 0.1173 - categorical_accuracy: 0.9588 - sensitivity: 0.9516 - specificity: 0.9875 - val_loss: 0.2585 - val_categorical_accuracy: 0.9023 - val_sensitivity: 0.9460 - val_specificity: 0.7764
    Epoch 22/30
    147/147 - 65s - loss: 0.1147 - categorical_accuracy: 0.9590 - sensitivity: 0.9519 - specificity: 0.9864 - val_loss: 0.2853 - val_categorical_accuracy: 0.9102 - val_sensitivity: 0.9248 - val_specificity: 0.8650
    Epoch 23/30
    147/147 - 66s - loss: 0.1097 - categorical_accuracy: 0.9619 - sensitivity: 0.9549 - specificity: 0.9898 - val_loss: 0.2464 - val_categorical_accuracy: 0.9102 - val_sensitivity: 0.9460 - val_specificity: 0.8058
    Epoch 24/30
    147/147 - 65s - loss: 0.1067 - categorical_accuracy: 0.9641 - sensitivity: 0.9578 - specificity: 0.9890 - val_loss: 0.3519 - val_categorical_accuracy: 0.8984 - val_sensitivity: 0.9161 - val_specificity: 0.8387
    Epoch 25/30
    147/147 - 66s - loss: 0.1028 - categorical_accuracy: 0.9641 - sensitivity: 0.9574 - specificity: 0.9905 - val_loss: 0.2331 - val_categorical_accuracy: 0.9082 - val_sensitivity: 0.9536 - val_specificity: 0.7717
    Epoch 26/30
    147/147 - 65s - loss: 0.0995 - categorical_accuracy: 0.9658 - sensitivity: 0.9599 - specificity: 0.9899 - val_loss: 0.2916 - val_categorical_accuracy: 0.9023 - val_sensitivity: 0.9216 - val_specificity: 0.8392
    Epoch 27/30
    147/147 - 65s - loss: 0.0959 - categorical_accuracy: 0.9683 - sensitivity: 0.9625 - specificity: 0.9919 - val_loss: 0.2325 - val_categorical_accuracy: 0.9004 - val_sensitivity: 0.9562 - val_specificity: 0.7339
    Epoch 28/30
    147/147 - 65s - loss: 0.0945 - categorical_accuracy: 0.9679 - sensitivity: 0.9621 - specificity: 0.9919 - val_loss: 0.3087 - val_categorical_accuracy: 0.9043 - val_sensitivity: 0.9304 - val_specificity: 0.8261
    Epoch 29/30
    147/147 - 66s - loss: 0.0905 - categorical_accuracy: 0.9700 - sensitivity: 0.9647 - specificity: 0.9917 - val_loss: 0.2283 - val_categorical_accuracy: 0.9219 - val_sensitivity: 0.9512 - val_specificity: 0.8379
    Epoch 30/30
    147/147 - 66s - loss: 0.0860 - categorical_accuracy: 0.9722 - sensitivity: 0.9669 - specificity: 0.9933 - val_loss: 0.2437 - val_categorical_accuracy: 0.9238 - val_sensitivity: 0.9433 - val_specificity: 0.8616
    

As you can see, we achieve a very nice accuracy, sensitivity, and specificity very quickly. I cut off the training after 30 epochs, but we could certainly push the performance even higher with longer training. The validation performance also is very close to the training runs, which is a good sign that we are not overfitting our data. Visually, we can look at the performance below.


```python
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']

sens = history.history['sensitivity']
val_sens = history.history['val_sensitivity']

spec = history.history['specificity']
val_spec = history.history['val_specificity']

epochs_range = range(EPOCHS)

plt.figure(figsize=(10, 10))
plt.subplot(3, 1, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy')
#plt.ylim([0.65, 1.1])

plt.subplot(3, 1, 2)
plt.plot(epochs_range, sens, label='Training Sensitivity')
plt.plot(epochs_range, val_sens, label='Validation Sensitivity')
plt.legend(loc='lower right')
plt.title('Sensitivity')
#plt.ylim([0.6, 1.1])

plt.subplot(3, 1, 3)
plt.plot(epochs_range, spec, label='Training Specificity')
plt.plot(epochs_range, val_spec, label='Validation Specificity')
plt.legend(loc='lower right')
plt.title('Specificity')
#plt.ylim([0.4, 1.1])
plt.show()
```


![png](output_28_0.png)


After a bit of spurious initial epochs, we see the FCN very quickly achieve a nice stable level of performance.

Finally, let's see how we perform on the test data.


```python
results = FCN.evaluate(test_generator)
```

    19/19 [==============================] - 7s 370ms/step - loss: 0.2891 - categorical_accuracy: 0.8931 - sensitivity: 0.9888 - specificity: 0.7398
    

For comparison, in the paper "*Diagnostic Accuracy of Chest x-Ray and Ultrasonography in Detection of Community Acquired Pneumonia; a Brief Report*" by Taghizadieh, Ala, Rahmani, and Nadi, medical diagnoses of Pneumonia using chest X-rays showed a sensitivity of 93.1% using traditional methods (compare to our FCN's 98.88% sensitivity!)

In conclusion, we can see how effectively an FCN can classify images of various sizes and how relatively straightforward its implementation is using the Tensorflow Keras Library.


```python

```
