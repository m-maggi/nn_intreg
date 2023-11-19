import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras import Model

# custom Loss as in https://stackoverflow.com/questions/64223840/use-additional-trainable-variables-in-keras-tensorflow-custom-loss-function
class MyLoss(Layer):
    def __init__(self, var1, var2):
        super(MyLoss, self).__init__()
        self.var1 = K.variable(var1) # or tf.Variable(var1) etc.
        self.var2 = K.variable(var2)
    
    def get_vars(self):
        return self.var1, self.var2
    
    def custom_loss(self, y_true, y_pred):
        return self.var1 * K.mean(K.square(y_true-y_pred)) + self.var2 ** 2
    
    def call(self, y_true, y_pred):
        self.add_loss(self.custom_loss(y_true, y_pred))
        return y_pred


X_train = np.zeros((10, 5))
inputs = Input(shape=(X_train.shape[1],))
y_input = Input(shape=(1,))
hidden1 = Dense(10)(inputs)
output = Dense(1)(hidden1)
my_loss = MyLoss(0.5, 0.5)(y_input, output) # here can also initialize those var1, var2
model = Model(inputs=[inputs, y_input], outputs=my_loss)

model.compile(optimizer= 'adam')