from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer

# Read the event count matrix and the event order matrix from text files
event_count_normal = np.loadtxt('data/count_0.txt', dtype=int)
event_order_normal = np.loadtxt('data/order_0.txt', dtype=int)
event_count_anomal = np.loadtxt('data/count_1.txt', dtype=int)
event_order_anomal = np.loadtxt('data/order_1.txt', dtype=int)

# --- Data preparation ---
# Load and concatenate your matrices
count_0 = event_count_normal
order_0 = event_order_normal
count_1 = event_count_anomal
order_1 = event_order_anomal

# scaled matrices
scaler = MinMaxScaler()
count_0_scaled = scaler.fit_transform(count_0)
count_1_scaled = scaler.transform(count_1)

scaler = MinMaxScaler()
order_0_scaled = scaler.fit_transform(order_0)
order_1_scaled = scaler.transform(order_1)

X = np.concatenate((count_0_scaled, order_0_scaled), axis=1)
X_test_1 = np.concatenate((count_1_scaled, order_1_scaled), axis=1)
y_test_1 = np.ones(count_1_scaled.shape[0])

X_train, X_test_0, y_train, y_test_0 = train_test_split(X, np.zeros(X.shape[0]), test_size=0.065, random_state=1)

X_test = np.concatenate((X_test_0, X_test_1), axis=0)
y_test = np.concatenate((y_test_0, y_test_1), axis=0)

input_data = tf.stack(X_train, axis=0)
input_data.shape


def get_error_term(v1, v2, _rmse=True):
    if _rmse:
        return np.sqrt(np.mean((v1 - v2) ** 2, axis=1))
    #return MAE
    return np.mean(abs(v1 - v2), axis=1)

# The reparameterization trick

def sample(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

original_dim =count_0.shape[1]+order_0.shape[1]
input_shape = (original_dim,)
intermediate_dim = int(original_dim / 2)
latent_dim = int(original_dim / 3)

# encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
# use the reparameterization trick and get the output from the sample() function
z = Lambda(sample, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
encoder = Model(inputs, z, name='encoder')
encoder.summary()

# decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)
# Instantiate the decoder model:
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# full VAE model
outputs = decoder(encoder(inputs))
vae_model = Model(inputs, outputs, name='vae_mlp')
vae_model.load_weights('model_46_30__100_256.h5')

# the KL loss function:
def vae_loss(x, x_decoded_mean):
    # compute the average MSE error, then scale it up, ie. simply sum on all axes
    reconstruction_loss = K.sum(K.square(x - x_decoded_mean))
    # compute the KL loss
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.square(K.exp(z_log_var)), axis=-1)
    # return the average loss over all
    total_loss = K.mean(reconstruction_loss + kl_loss)
    #total_loss = reconstruction_loss + kl_loss
    return total_loss

opt = optimizers.legacy.Adam(learning_rate=0.0001, clipvalue=0.5)
#opt = optimizers.RMSprop(learning_rate=0.0001)

vae_model.compile(optimizer=opt, loss=vae_loss)
vae_model.summary()
# Finally, we train the model:
with tf.device('/device:GPU:0'):
  results = vae_model.fit(input_data, input_data,
                          shuffle=True,
                          epochs=500,
                          batch_size=256, steps_per_epoch=1000)

vae_model.save('model_46_30__200_256.h5') #dense 46 latent 30 500 epochs 256 batch size

X_train_pred = vae_model.predict(X_train)
