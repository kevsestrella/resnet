def build_model():
    import numpy as np
    from keras import layers
    from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
    from keras.models import Model, load_model
    from keras.preprocessing import image
    from keras.utils import layer_utils
    from keras.utils.data_utils import get_file
    from keras.applications.imagenet_utils import preprocess_input
    from IPython.display import SVG
    from keras.utils.vis_utils import model_to_dot
    from keras.utils import plot_model
    from keras.initializers import glorot_uniform
    import scipy.misc
    from matplotlib.pyplot import imshow
    #%matplotlib inline

    import keras.backend as K
    K.set_image_data_format('channels_last')
    K.set_learning_phase(1)

    import tensorflow as tf
    import h5py 

    def identity_block(X, f, filters, stage, block):
        """
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network

        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value. You'll need this later to add back to the main path. 
        X_shortcut = X

        # First component of main path
        X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path
        X = Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding='same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path
        X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid', name = conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name = bn_name_base + '2c')(X)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X = layers.Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    def convolutional_block(X, f, filters, stage, block, s = 2):
        """
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        s -- Integer, specifying the stride to be used

        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value
        X_shortcut = X


        ##### MAIN PATH #####
        # First component of main path 
        X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path
        X = Conv2D(filters=F2,kernel_size=(f,f),strides=(1,1),padding='same', name = conv_name_base + '2b', kernel_initializer= glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path
        X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1),name= conv_name_base+'2c', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base+ '2c')(X)

        ##### SHORTCUT PATH ####
        X_shortcut = Conv2D(filters = F3, kernel_size=(1,1), strides=(s,s), name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name=bn_name_base+'1')(X_shortcut)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X = layers.Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    def ResNet50(input_shape = (64, 64, 3), classes = 6):
        """
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes

        Returns:
        model -- a Model() instance in Keras
        """

        # Define the input as a tensor with shape input_shape
        X_input = Input(input_shape)


        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input)

        # Stage 1
        X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
        X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
        X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

        # Stage 3
        X = convolutional_block(X, f=3 , filters=[128,128,512], stage = 3, block='a' )
        X = identity_block(X,f = 3,filters=[128,128,512], stage=3, block='b')
        X = identity_block(X,f = 3,filters=[128,128,512], stage=3, block='c')
        X = identity_block(X,f = 3,filters=[128,128,512], stage=3, block='d')

        # Stage 4
        X = convolutional_block(X, f=3 , filters=[256,256,1024], stage = 4, block='a' )
        X = identity_block(X,f = 3,filters=[256,256,1024], stage=4, block='b')
        X = identity_block(X,f = 3,filters=[256,256,1024], stage=4, block='c')
        X = identity_block(X,f = 3,filters=[256,256,1024], stage=4, block='d')
        X = identity_block(X,f = 3,filters=[256,256,1024], stage=4, block='e')
        X = identity_block(X,f = 3,filters=[256,256,1024], stage=4, block='f')

        # Stage 5
        X = convolutional_block(X, f=3 , filters=[512,512,2048], stage = 5, block='a' )
        X = identity_block(X,f = 3,filters=[512,512,2048], stage=5, block='b')
        X = identity_block(X,f = 3,filters=[512,512,2048], stage=5, block='c')

        # AVGPOOL. Use "X = AveragePooling2D(...)(X)"
        X = AveragePooling2D((2,2))(X)

        # output layer
        X = Flatten()(X)
        X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)


        # Create model
        model = Model(inputs = X_input, outputs = X, name='ResNet50')

        return model

    model = ResNet50(input_shape = (64, 64, 3), classes = 6)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def load_dataset():
        train_dataset = h5py.File('./datasets/train_signs.h5', "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

        test_dataset = h5py.File('./datasets/test_signs.h5', "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
        test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

        classes = np.array(test_dataset["list_classes"][:]) # the list of classes

        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    # Normalize image vectors
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.

    # Convert training and test labels to one hot matrices
    Y_train = np.eye(6)[Y_train_orig.reshape(-1)]
    Y_test = np.eye(6)[Y_test_orig.reshape(-1)]

    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))

    model.fit(X_train, Y_train, epochs = 2, batch_size = 32)

    preds = model.evaluate(X_test, Y_test)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))
