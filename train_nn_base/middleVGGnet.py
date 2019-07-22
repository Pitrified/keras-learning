from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


class MiddleVGGNet:
    @staticmethod
    def build(
        width,
        height,
        depth,
        classes,
        fully_connected_size,
        do_batch_normalization=True,
        do_dropout=True,
    ):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # (CONV => RELU) * 2 => POOL layer set
        model.add(Conv2D(64, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        if do_batch_normalization:
            model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        if do_batch_normalization:
            model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        if do_dropout:
            model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL layer set
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        if do_batch_normalization:
            model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        if do_batch_normalization:
            model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        if do_dropout:
            model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL layer set
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Activation("relu"))
        if do_batch_normalization:
            model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Activation("relu"))
        if do_batch_normalization:
            model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        if do_dropout:
            model.add(Dropout(0.25))

        # (CONV => RELU) * 3 => POOL layer set
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Activation("relu"))
        if do_batch_normalization:
            model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Activation("relu"))
        if do_batch_normalization:
            model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Activation("relu"))
        if do_batch_normalization:
            model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        if do_dropout:
            model.add(Dropout(0.25))

        # (CONV => RELU) * 3 => POOL layer set
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Activation("relu"))
        if do_batch_normalization:
            model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Activation("relu"))
        if do_batch_normalization:
            model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Activation("relu"))
        if do_batch_normalization:
            model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        if do_dropout:
            model.add(Dropout(0.25))

        # FLATTEN
        model.add(Flatten())

        # two set of FC => RELU layers
        #  model.add(Dense(512))
        model.add(Dense(fully_connected_size))
        #  model.add(Dense(4096)) # sadly this kills my machine
        model.add(Activation("relu"))
        if do_batch_normalization:
            model.add(BatchNormalization())
        if do_dropout:
            model.add(Dropout(0.5))

        model.add(Dense(fully_connected_size))
        model.add(Activation("relu"))
        if do_batch_normalization:
            model.add(BatchNormalization())
        if do_dropout:
            model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model

