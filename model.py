import csv
import cv2
import numpy as np
import sklearn
import math 

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Cropping2D

def get_csv_log_file_data(folder_list):
    """ The Function takes a list of folders and returns combined list of entries 
    and the folder of the entry, taken from from the driving_log.csv file."""
    
    csv_lines = []
    
    # For the driving_log.csv file from imput list of folders:
    # In this case ['training_data_middle', 'training_data_opposite', 'training_data_recover']
    # The first folder has training samples to train the network to drive car in the middle of the road
    # The second folder has data by driving the car in the clock wise direction on track one
    # The third folder has samples to teach car to recover to middle of road from sides.
    for val in folder_list:
        print('./{}/driving_log.csv'.format(val))
        with open('./{}/driving_log.csv'.format(val)) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                csv_lines.append([line, './{}/'.format(val)])
    return csv_lines

def generator(samples, batch_size=32):
    """ Generator function that provides center, right, left and their flipped images and their steering angles """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            correction = 0.2
            for batch_sample in batch_samples:
                # Center Images and their steering angles
                center_name = batch_sample[1] + 'IMG/' + batch_sample[0][0].split('/')[-1]
                center_image = cv2.cvtColor(cv2.imread(center_name), cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[0][3])
                
                # Left Images and their corrected steering angles
                left_name = batch_sample[1] + 'IMG/' + batch_sample[0][1].split('/')[-1]
                left_image = cv2.cvtColor(cv2.imread(left_name), cv2.COLOR_BGR2RGB)
                left_angle = float(batch_sample[0][3]) + correction
                
                # Right Images and their corrected steering angles
                right_name = batch_sample[1] + 'IMG/' + batch_sample[0][2].split('/')[-1]
                right_image = cv2.cvtColor(cv2.imread(right_name), cv2.COLOR_BGR2RGB)
                right_angle = float(batch_sample[0][3]) - correction
                
                images.append(center_image)
                # Center Images are fliped 
                images.append(cv2.flip(center_image, 1))
                angles.append(center_angle)
                angles.append(-1 * center_angle)
                
                images.append(left_image)
                # Left Images are flipped
                images.append(cv2.flip(left_image, 1))
                angles.append(left_angle)
                angles.append(-1 * left_angle)
                
                images.append(right_image)
                # Right Images are flipped
                images.append(cv2.flip(right_image, 1))
                angles.append(right_angle)
                angles.append(-1 * right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
def get_train_and_validation_samples():
    """ Function splits data into training and validation datasets """
    train_samples, validation_samples = train_test_split(get_csv_log_file_data(['training_data_middle', 'training_data_opposite', 'training_data_recover']), test_size=0.3)
    return train_samples, validation_samples
    
def get_generator_results():
    """ Function provides the data yield of generator, and the length of training and validation sets """
    batch_size=32
    train_samples, validation_samples = get_train_and_validation_samples()
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)
    return train_generator, validation_generator, len(train_samples), len(validation_samples)

def create_and_get_model():
    """ Function creates a network model and provides the model """
    model = Sequential()
    
    # Normalization Layer
    model.add(Lambda( lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    
    # Cropping Layers, all the images are cropped
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
    
    # Convolutional Layer with 5x5 filter and stride of 2x2 followed by RELU Activation
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    
    # Convolutional Layer with 5x5 filter and stride of 2x2 followed by RELU Activation
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    
    # Convolutional Layer with 5x5 filter and stride of 2x2 followed by RELU Activation
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    
    # Convolutional Layer with 3x3 filter and stride of 1x1 followed by RELU Activation
    model.add(Convolution2D(64,3,3,activation="relu"))
    
    # Convolutional Layer with 3x3 filter and stride of 1x1 followed by RELU Activation
    model.add(Convolution2D(64,3,3,activation="relu"))
    
    # Flatten Layer
    model.add(Flatten())
    
    # Fully Connected layer follwed by a Droupout layer with probability of 0.5
    model.add(Dense(576,activation="relu"))
    model.add(Dropout(0.5))
    
    # Fully Connected layer follwed by a Droupout layer with probability of 0.5
    model.add(Dense(100,activation="relu"))
    model.add(Dropout(0.5))
    
    # Fully Connected layer
    model.add(Dense(50,activation="relu"))
    
    # Output layer: provides steering Angle
    model.add(Dense(1))
    
    return model

def get_compiled_model(loss_string, optimizer_string):
    """ The model cerated by above function is compiled with provided loss function and optimizer """
    model = create_and_get_model()
    
    # In this case Mean Squared Error is used as loss function and AdamOptimizer is used.
    model.compile(loss=loss_string, optimizer=optimizer_string)
    return model
    
def fit_and_save_model():
    """ Function trains the network and provides training and validation loss """
    batch_size =  32
    
    # Compiles the model to use Mean Squared Error as loss function and AdamOptimizer as the Optimizer
    model = get_compiled_model('mse', 'adam')
    train_generator, validation_generator, samples_per_epoch, nb_val_samples = get_generator_results()
    model.fit_generator(train_generator, steps_per_epoch=math.ceil((samples_per_epoch*3*2)/batch_size), validation_data=validation_generator, validation_steps=math.ceil((nb_val_samples*3*2)/batch_size), epochs=5, verbose=1)
    
    # Model saved to file model.h5
    model.save('model.h5')

    print('Model Trained and saved!!!!!!!')

    # Print Model Summary
    model.summary()
    
#  Call function to train and save the network model.
fit_and_save_model()