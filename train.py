import numpy as np
import pickle
#import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Import necessary items from Keras
from keras.models import Model
from keras.layers import Activation, Dropout, UpSampling2D, concatenate, Input
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras import regularizers

# Load training images
a = open("full_CNN_train.p", "rb")
train_images = pickle.load(a)

# Load image labels
labels = pickle.load(open("full_CNN_labels.p", "rb" ))

# Make into arrays as the neural network wants these
train_images = np.array(train_images)
labels = np.array(labels)



# Normalize labels - training images get normalized to start in the network
labels = labels / 255

# Shuffle images along with their labels, then split into training/validation sets
train_images, labels = shuffle(train_images, labels)
# Test size may be 10% or 20%
X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1)

# Batch size, epochs and pool size below are all paramaters to fiddle with for optimization
batch_size = 16
epochs = 15
pool_size = (2, 2)
#input_shape = X_train.shape[1:]

### Here is the actual neural network ###
# Normalizes incoming inputs. First layer needs the input shape to work
#BatchNormalization(input_shape=input_shape)
Inputs = Input(batch_shape=(None, 80, 160, 3))

# Below layers were re-named for easier reading of model summary; this not necessary
# Conv Layer 1
Conv1 = Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu')(Inputs)
Bat1 = BatchNormalization()(Conv1)

# Conv Layer 2
Conv2 = Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu')(Conv1)
Bat2 = BatchNormalization()(Conv2)

# Pooling 1
Pool1 = MaxPooling2D(pool_size=pool_size)(Conv2)

# Conv Layer 3
Conv3 = Conv2D(32, (3, 3), padding = 'valid', strides=(1,1), activation = 'relu')(Pool1)
#Drop3 = Dropout(0.2)(Conv3)
Bat3 = BatchNormalization()(Conv3)

# Conv Layer 4
Conv4 = Conv2D(32, (3, 3), padding = 'valid', strides=(1,1), activation = 'relu')(Bat3)
#Drop4 = Dropout(0.5)(Conv4)
Bat4 = BatchNormalization()(Conv4)

# Conv Layer 5
Conv5 = Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu')(Bat4)
#Drop5 = Dropout(0.2)(Conv5)
Bat5 = BatchNormalization()(Conv5)

# Pooling 2
Pool2 = MaxPooling2D(pool_size=pool_size)(Bat5)

# Conv Layer 6
Conv6 = Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu')(Pool2)
#Drop6 = Dropout(0.2)(Conv6)
Bat6 = BatchNormalization()(Conv6)

# Conv Layer 7
Conv7 = Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu')(Bat6)
#Drop7 = Dropout(0.2)(Conv7)
Bat7 = BatchNormalization()(Conv7)

# Pooling 3
Pool3 = MaxPooling2D(pool_size=pool_size)(Bat7)

# Conv Layer 8
Conv8 = Conv2D(128, (3, 3), padding='valid', strides=(1,1), activation = 'relu')(Pool3)
#Drop8 = Dropout(0.2)(Conv8)
Bat8 = BatchNormalization()(Conv8)

# Conv Layer 9
Conv9 = Conv2D(128, (3, 3), padding='valid', strides=(1,1), activation = 'relu')(Bat8)
#Drop9 = Dropout(0.2)(Conv9)
Bat9 = BatchNormalization()(Conv9)

# Pooling 4
Pool4 = MaxPooling2D(pool_size=pool_size)(Bat9)


# Upsample 1 to Deconv 1
Deconv1 = Conv2DTranspose(128, (2, 2), padding='valid', strides=(2,2), activation = 'relu')(Pool4)
#Up1 = UpSampling2D(size=pool_size)(Pool4)
Mer1 = concatenate([Deconv1, Bat9], axis=-1)

# Deconv 2
Deconv2 = Conv2DTranspose(128, (3, 3), padding='valid', strides=(1,1), activation = 'relu')(Mer1)
DBat2 = BatchNormalization()(Deconv2)

# Deconv 3
Deconv3 = Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu')(DBat2)
DBat3 = BatchNormalization()(Deconv3)

# Upsample 2 to Deconv 4
Deconv4 = Conv2DTranspose(64, (2, 2), padding='valid', strides=(2,2), activation = 'relu')(DBat3)
#Up2 = UpSampling2D(size=pool_size)(DBat2)
Mer2 = concatenate([Deconv4, Bat7], axis=-1)

# Deconv 5
Deconv5 = Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu')(Mer2)
DBat5 = BatchNormalization()(Deconv5)

# Deconv 6
Deconv6 = Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu')(DBat5)
DBat6 = BatchNormalization()(Deconv6)

# Upsample 3 to Deconv 7
Deconv7 = Conv2DTranspose(32, (2, 2), padding='valid', strides=(2,2), activation = 'relu')(DBat6)
#Up3 = UpSampling2D(size=pool_size)(DBat4)
Mer3 = concatenate([Deconv7, Bat5], axis=-1)

# Deconv 8
Deconv8 = Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu')(Mer3)
DBat8 = BatchNormalization()(Deconv8)

# Deconv 9
Deconv9 = Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu')(DBat8)
DBat9 = BatchNormalization()(Deconv9)

# Deconv 10
Deconv10 = Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu')(DBat9)
DBat10 = BatchNormalization()(Deconv10)

# Upsample 4 to Deconv 11
Deconv11 = Conv2DTranspose(16, (2, 2), padding='valid', strides=(2,2), activation = 'relu')(DBat10)
#Up4 = UpSampling2D(size=pool_size)(DBat7)
Mer4 = concatenate([Deconv11, Bat2], axis=-1)

# Deconv 12
Deconv12 = Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu')(Mer4)
DBat12 = BatchNormalization()(Deconv12)

# Deconv 13
Deconv13 = Conv2DTranspose(8, (3, 3), padding='valid', strides=(1,1), activation = 'relu')(DBat12)
DBat13 = BatchNormalization()(Deconv13)

# Final layer - only including one channel so 1 filter
Final = Conv2DTranspose(1, (3, 3), padding='same', strides=(1,1), activation = 'relu')(DBat13)

### End of network ###
model = Model(inputs=Inputs, outputs=Final)


# Using a generator to help the model use less data
# Channel shifts help with shadows slightly
datagen = ImageDataGenerator(channel_shift_range=0.2)
datagen.fit(X_train)

# Compiling and training the model
model.compile(optimizer='Adam', loss='mean_squared_error')
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=len(X_train)/batch_size,
epochs=epochs, verbose=1, validation_data=(X_val, y_val))

# Freeze layers since training is done
model.trainable = False
model.compile(optimizer='Adam', loss='mean_squared_error')

# Save model architecture and weights
model.save('full_CNN_model_HYe15.h5')

# Show summary of model
model.summary()
plot_model(model, to_file='model.png')
