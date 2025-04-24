#!/usr/bin/env python
# coding: utf-8

# Import all the necessary libraries from tensorflow,sklearn

# In[ ]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
import matplotlib.pyplot as plt
import numpy as np
import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score
from tensorflow.keras.regularizers import l2


# <h4 color='black'>Load the dataset From tensorflow dataset</h4>

# In[ ]:


(X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data()



# Analyize the shape of training and test data

# In[ ]:


print(X_train.shape)
y_train.shape


# split train dataset into train,valid dataset

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(
   X_train, y_train, test_size=0.1, random_state=42)


# In[ ]:


print(X_valid.shape)
y_valid.shape


# In[ ]:


print(X_test.shape)
y_test.shape


# There are 50000 training images and 10000 test images

# In[ ]:


y_train = y_train.reshape(-1,)
y_valid=y_valid.reshape(-1,)
y_train[:5]


# In[ ]:


y_test = y_test.reshape(-1,)
y_test.shape


# In[ ]:


classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# In[ ]:


def plot_Images(X, y, index):
    plt.figure(figsize = (5,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])


# plotting some training images

# In[ ]:


for i in range(1,3):
   plot_Images(X_train, y_train, i)



# <h4 style="color:purple">Normalizing the training data</h4>

# In[ ]:


X_train = X_train / 255.0
X_test = X_test / 255.0
X_valid=X_valid/255.0


# <h4 style="color:"> Building a custom convolutional neural network to train our images</h4>

# In[ ]:


cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(4096, activation='relu',kernel_regularizer=l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(4096, activation='relu',kernel_regularizer=l2(0.001)),
    layers.BatchNormalization(),
    layers.Dense(10, activation='softmax')
])


# In[ ]:


cnn.summary()


# In[ ]:


cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


history = cnn.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_valid, y_valid))


# 

# Accuracy of custom_cnn

# In[ ]:


test_loss, test_acc = cnn.evaluate(X_test, y_test)
print(test_acc*100)


# <h4 style="color:"> Implementing a alext_net_architecture convolutional neural network to train our images</h4>

# In[ ]:


alex_net_cnn = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(96, (11,11), strides=(4,4), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'),

        tf.keras.layers.Conv2D(256, (5,5), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'),

        tf.keras.layers.Conv2D(384, (3,3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(384, (3,3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu',kernel_regularizer=l2(0.001)),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(4096, activation='relu',kernel_regularizer=l2(0.001)),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(10, activation='softmax')
     ])


# In[ ]:


alex_net_cnn.summary()


# In[ ]:


alex_net_cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


history1 = alex_net_cnn.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_valid, y_valid))


# Accuracy of Alex-Net

# In[ ]:


test_loss, test_acc = alex_net_cnn.evaluate(X_test, y_test)
print(test_acc*100)


# <h4 style="color:"> Implementing a lenet convolutional neural network to train our images</h4>

# In[ ]:


le_net_cnn = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        tf.keras.layers.Conv2D(16, (5, 5), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(84, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(10, activation='softmax')
    ])


# In[ ]:


le_net_cnn.summary()


# In[ ]:


le_net_cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history1 = le_net_cnn.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_valid, y_valid))


# Accuracy of le_net

# In[ ]:


test_loss, test_acc = le_net_cnn.evaluate(X_test, y_test)
print(test_acc*100)


# custom_cnn_Architecture
# 
# Evaluating the results of training data  using accuracy,confusion_matrix, classification report

# In[ ]:


from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
y_pred = cnn.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]
class_labels = np.unique(np.concatenate((y_test, y_pred_classes)))
conf_matrix = confusion_matrix(y_test, y_pred_classes)
# Print confusion matrix with labels
print(classes)
print("Confusion Matrix:")
print("\t" + "\t".join(str(label) for label in class_labels))
for i, row in enumerate(conf_matrix):
    print(f"{class_labels[i]}\t" + "\t".join(str(count) for count in row))






# In[ ]:


print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes,target_names=classes))


# In[ ]:


for i in range(11,20):

   (plot_Images(X_test, y_pred_classes, i))


# Alex _Net_Architecture
# 
# Evaluating the results of training data  using accuracy,confusion_matrix, classification report

# In[ ]:


from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
y_pred = alex_net_cnn.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]
class_labels = np.unique(np.concatenate((y_test, y_pred_classes)))
conf_matrix = confusion_matrix(y_test, y_pred_classes)
# Print confusion matrix with labels
print(classes)
print("Confusion Matrix:")
print("\t" + "\t".join(str(label) for label in class_labels))
for i, row in enumerate(conf_matrix):
    print(f"{class_labels[i]}\t" + "\t".join(str(count) for count in row))






# In[ ]:


print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes,target_names=classes))


# Le_Net_Architecture
# 
# Evaluating the results of training data  using accuracy,confusion_matrix, classification report

# In[ ]:


from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
y_pred = le_net_cnn.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]
class_labels = np.unique(np.concatenate((y_test, y_pred_classes)))
conf_matrix = confusion_matrix(y_test, y_pred_classes)
# Print confusion matrix with labels
print(classes)
print("Confusion Matrix:")
print("\t" + "\t".join(str(label) for label in class_labels))
for i, row in enumerate(conf_matrix):
    print(f"{class_labels[i]}\t" + "\t".join(str(count) for count in row))






# In[ ]:


print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes,target_names=classes))


# In[ ]:


for i in range(11,20):

   (plot_Images(X_test, y_pred_classes, i))

