import math
import os 
import argparse
import json 

import tensorflow as tf
tf.test.is_gpu_available()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import netCDF4

import keras
from keras import layers
from keras import backend as K
from keras.models import Model
from keras.losses import binary_crossentropy, mse
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

class AnnealingCallback(keras.callbacks.Callback):
    def __init__(self, epochs):
        super(AnnealingCallback, self).__init__()
        self.epochs = epochs 
        
    def on_epoch_begin(self, epoch, logs={}):
        new_kl_weight = epoch/self.epochs 
        K.set_value(self.model.kl_weight, new_kl_weight)
        print("Using updated KL Weight:", K.get_value(self.model.kl_weight))

class Sampling(keras.layers.Layer):
    def call(self, inputs):
        """
        TODO 
        """
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var/2) + mean

def ELBO_Test(z_log_var, z_mean, vae):
    def _ELBO_Test(true, pred):
        """
        TODO 
        """
        true = tf.reshape(true, [-1, 128 * 30])

        x_mu = pred[:, :128*30]
        x_log_var = pred[:, 128*30:]

        # Gaussian reconstruction loss
        
        #griffins edits for anomaly detection
        x_mu = x_mu.astype('float64') 
        x_log_var = x_log_var.astype('float64')
        #end griffins edits
        
        mse = -0.5 * K.sum(K.square(true - x_mu)/K.exp(x_log_var), axis=1)
        var_trace = -0.5 * K.sum(x_log_var, axis=1)
        log2pi = -0.5 * 128 * 30 * np.log(2 * np.pi)
        
        log_likelihood = mse + var_trace + log2pi
        print("log likelihood shape", log_likelihood.shape)

        # NOTE: We don't take a mean here, since we first want to add the KL term
        reconstruction_loss = -log_likelihood

        # KL divergence loss
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=1)
        kl_loss *= -0.5

        #for anon dectation
        kl_loss = tf.cast(kl_loss, dtype=tf.float64)
        return -0.5*(reconstruction_loss + kl_loss)
        #end anon detection edits

    return _ELBO_Test
    
    
def kl_reconstruction_loss(z_log_var, z_mean, vae):
    def _kl_reconstruction_loss(true, pred):
        """
        TODO 
        """
        
        ##changed from 128*30 to 128*30*3 x,y,z in our case
        true = tf.reshape(true, [-1, 128*30*3 ])
        
        ##changed from 128 * 30 to 80 * 80 * 37 
        x_mu = pred[:, :128*30*3]
        x_log_var = pred[:, 128*30*3:]

        # Gaussian reconstruction loss
        
        #griffins edits for anomaly detection
        #x_mu = x_mu.astype('float64') 
        #x_log_var = x_log_var.astype('float64')
        #end griffins edits
        
        mse = -0.5 * K.sum(K.square(true - x_mu)/K.exp(x_log_var), axis=1)
        var_trace = -0.5 * K.sum(x_log_var, axis=1)
        ## change from 128 * 30 to 80 * 80 * 37 
        log2pi = -0.5 * 128 * 30 * 3 * np.log(2 * np.pi)

        
        log_likelihood = mse + var_trace + log2pi
        print("log likelihood shape", log_likelihood.shape)

        # NOTE: We don't take a mean here, since we first want to add the KL term
        reconstruction_loss = -log_likelihood

        # KL divergence loss
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=1)
        kl_loss *= -0.5

        return K.mean(reconstruction_loss + vae.kl_weight * kl_loss)
        
        #for anon dectation
        #kl_loss = tf.cast(kl_loss, dtype=tf.float64)
        #return -0.5*(reconstruction_loss + kl_loss)
        #end anon detection edits

    return _kl_reconstruction_loss

def kl(z_log_var, z_mean):
    def _kl(true, pred):
        """
        TODO 
        """
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        # kl_loss = K.print_tensor(kl_loss, message='EULA PEULA')
        #return vae.kl_weight
        return K.mean(kl_loss)
    
    return _kl

def reconstruction(true, pred):
    """
    TODO
    """
    ## changed dimensions from 128 * 30 to  128*30*3
    true = tf.reshape(true, [-1, 128 * 30 * 3])

    x_mu = pred[:, :128*30*3]
    x_log_var = pred[:, 128*30*3:]
    
    mse = -0.5 * K.sum(K.square(true - x_mu)/K.exp(x_log_var), axis=1)
    var_trace = -0.5 * K.sum(x_log_var, axis=1)
    ## changed dimensions here 
    log2pi = -0.5 * 128 * 30 * 3 * np.log(2 * np.pi)

    
    log_likelihood = mse + var_trace + log2pi
    print("log likelihood shape", log_likelihood.shape)

    return K.mean(-log_likelihood)

def encoder_gen(input_shape: tuple, encoder_config: dict, id):
    """
    Create the architecture for the VAE encoder. 
    """

    class EncoderResult():
        pass 

    encoder_result = EncoderResult()

    # Construct VAE Encoder layers
    
    ##update VAE encoder add inputs 
    inputs = keras.layers.Input(shape=[input_shape[0], input_shape[1], input_shape[2], 1])
    ##figure out the padding 
    ##30 --> should go to 32 (32, 64, 128...)
    ##32 --> should go to 30 in decoding 
    ## 30 --> 32 ( 1, 0, 0 )
    zero_padded_inputs = keras.layers.ZeroPadding3D(padding=(1, 0, 0))(inputs)#padding out x, y, keeping z 

     ##update from 2dconv to 3dconv 
   
    print("shape of input after padding", zero_padded_inputs.shape)
    #if the shape of input goes from 30 to 32, then you are okay
    #otherwise fix zero padding 
    # break to check shape of input 
    z = keras.layers.convolutional.Conv3D(
        encoder_config["conv_1"]["filter_num"], 
        tuple(encoder_config["conv_1"]["kernel_size"]), 
        padding='same', 
        activation=encoder_config["activation"], 
        strides=encoder_config["conv_1"]["stride"]
    )(zero_padded_inputs)
    print("shape after first convolutional layer", z.shape)

    z = keras.layers.convolutional.Conv3D(
        encoder_config["conv_2"]["filter_num"], 
        tuple(encoder_config["conv_2"]["kernel_size"]), 
        padding='same', 
        activation=encoder_config["activation"], 
        strides=encoder_config["conv_2"]["stride"]
    )(z)

    print("shape after second convolutional layer", z.shape)

    z = keras.layers.convolutional.Conv3D(
        encoder_config["conv_3"]["filter_num"], 
        tuple(encoder_config["conv_3"]["kernel_size"]), 
        padding='same', 
        activation=encoder_config["activation"], 
        strides=encoder_config["conv_3"]["stride"]
    )(z)

    print("shape after third convolutional layer", z.shape)

    z_mean = keras.layers.convolutional.Conv3D(
        encoder_config["conv_mu"]["filter_num"], 
        tuple(encoder_config["conv_mu"]["kernel_size"]), 
        padding='same', 
        strides=encoder_config["conv_mu"]["stride"]
    )(z)
    

    z_log_var = keras.layers.convolutional.Conv3D(
        encoder_config["conv_log_var"]["filter_num"], 
        tuple(encoder_config["conv_log_var"]["kernel_size"]), 
        padding='same', 
        strides=encoder_config["conv_log_var"]["stride"]
    )(z)

    z_mean = keras.layers.Flatten()(z_mean)
    z_log_var = keras.layers.Flatten()(z_log_var)

    print("z mean shape", z_mean._keras_shape)
    print("z log var shape", z_log_var._keras_shape)
    

    z = Sampling()([z_mean, z_log_var])

    # Instantiate Keras model for VAE encoder 
    vae_encoder = keras.Model(inputs=[inputs], outputs=[z_mean, z_log_var, z])
    plot_model(vae_encoder, to_file='./model_graphs/model_diagrams/encoder_{}.png'.format(id), show_shapes=True)
    # Package up everything for the encoder
    encoder_result.inputs = inputs
    encoder_result.z_mean = z_mean
    encoder_result.z_log_var = z_log_var
    encoder_result.z = z
    encoder_result.vae_encoder = vae_encoder 
    return encoder_result

def decoder_gen(
    original_input: tuple,
    decoder_config: dict
):
    """
    Create the architecture for the VAE decoder 
    """
    decoder_inputs = keras.layers.Input(shape=[decoder_config["latent_dim"]])
    print("decoder_inputs", decoder_inputs._keras_shape)
    # Reshape input to be an image 
    #for original case - for latent space size 64 - config 38
    #x = keras.layers.Reshape((2, 8, 4))(decoder_inputs)
    #superior for 1024 - works best - config 35
    #x = keras.layers.Reshape((2, 8, 64))(decoder_inputs)
    
    ## reshape additional dimension 
    x = keras.layers.Reshape((decoder_config["latent_reshape"]["dim_1"], decoder_config["latent_reshape"]["dim_2"], decoder_config["latent_reshape"]["dim_3"],  decoder_config["latent_reshape"]["dim_4"]))(decoder_inputs)
    #for foster arch - config 34
    #x = keras.layers.Reshape((2, 8, 32))(decoder_inputs)
    
    ## update to 3d conv transpose 
    
    x = keras.layers.convolutional.Conv3DTranspose(
        decoder_config["conv_t_0"]["filter_num"], 
        tuple(decoder_config["conv_t_0"]["kernel_size"]), 
        padding='same', 
        activation=decoder_config["activation"], 
        strides=decoder_config["conv_t_0"]["stride"]
    )(x)

    # Start tranpose convolutional layers that upsample the image
    print("shape at beginning of decoder", x.shape)

    x = keras.layers.Conv3DTranspose(
        decoder_config["conv_t_1"]["filter_num"], 
        tuple(decoder_config["conv_t_1"]["kernel_size"]), 
        padding='same', 
        activation=decoder_config["activation"], 
        strides=decoder_config["conv_t_1"]["stride"]
    )(x)
    print("shape after first convolutional transpose layer", x._keras_shape)

    x = keras.layers.Conv3DTranspose(
        decoder_config["conv_t_2"]["filter_num"], 
        tuple(decoder_config["conv_t_2"]["kernel_size"]), 
        padding='same', 
        strides=decoder_config["conv_t_2"]["stride"],
        activation=decoder_config["activation"]
    )(x)
    print("shape after second convolutional transpose layer", x._keras_shape)
    x_mu = keras.layers.Conv3DTranspose(
        decoder_config["conv_mu"]["filter_num"], 
        tuple(decoder_config["conv_mu"]["kernel_size"]), 
        padding='same',  
        strides=decoder_config["conv_mu"]["stride"],
        activation=decoder_config["conv_mu"]["activation"]
    )(x)
    print("shape after convolutional mu layer", x_mu._keras_shape)

    x_log_var = keras.layers.Conv3DTranspose(
        decoder_config["conv_log_var"]["filter_num"], 
        tuple(decoder_config["conv_log_var"]["kernel_size"]), 
        padding='same',  
        strides=decoder_config["conv_log_var"]["stride"],
        activation=decoder_config["conv_log_var"]["activation"]
    )(x)
    print("shape after convolutional log var layer", x_mu._keras_shape)
    
    ## check params here 
    ## change cropping 
    ## might be able to just use first line for cropping
    ## adds 0 to each end 
#     x_mu = keras.layers.Cropping3D(cropping=(1, 0, 0))(x_mu)
    x_mu = keras.layers.Cropping3D(cropping=((1,1), (0,0), (6,7)))(x_mu)
    print("shape after cropping", x_mu._keras_shape)
    
    # break statement here to figure out paramas 

    ## check params here 
    ## might be able to just use first line for cropping
    ## 128, 32 --> 30 , 3 
    ## (0, 2, 0) 
#     x_log_var = keras.layers.Cropping3D(cropping=(1, 0, 0))(x_log_var)
    x_log_var = keras.layers.Cropping3D(cropping=((1,1), (0,0), (6,7)))(x_log_var)
    print("shape after cropping", x_log_var._keras_shape)

    # break statement here to figure out paramas 

    x_mu = keras.layers.Flatten()(x_mu)
    x_log_var = keras.layers.Flatten()(x_log_var)
   
    x_mu_log_var = keras.layers.Concatenate(axis=1)([x_mu, x_log_var])
    
    print("x_mu======= ", x_mu)
    print("x log var ======= ", x_log_var)
    print("x mu log var ========== ", x_mu_log_var) 
    
    variational_decoder = keras.Model(inputs=[decoder_inputs], outputs=[x_mu_log_var])
    
    return variational_decoder

def plot_training_losses(h, id):
    """
    Plot training loss graphs for 
        (1) KL term
        (2) Reconstruction term
        (3) Total ELBO loss  
    """
    hdict = h.history
    print(hdict)

    train_reconstruction_losses = hdict['reconstruction']
    valid_reconstruction_losses = hdict['val_reconstruction']

    kl_train_losses = hdict['_kl']
    kl_valid_losses = hdict['val__kl']

    total_train_losses = hdict['_kl_reconstruction_loss']
    total_valid_losses = hdict['val__kl_reconstruction_loss']

    epochs = range(1, len(train_reconstruction_losses) + 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.8, 4.8))

    # Plot combined loss 
    ax1.plot(epochs, total_train_losses, 'b', label='Train')
    ax1.plot(epochs, total_valid_losses, 'r', label='Valid')
    ax1.set(xlabel="Epochs", ylabel="Loss")
    ax1.legend(prop={'size': 10})
    ax1.set_title("Combined Loss")

    # Plot KL 
    ax2.plot(epochs, kl_train_losses, 'b', label='Train')
    ax2.plot(epochs, kl_valid_losses, 'r', label='Valid')
    ax2.set(xlabel="Epochs", ylabel="Loss")
    ax2.legend(prop={'size': 10})
    ax2.set_title("KL Loss")

    # Plot reconstruction loss 
    ax3.plot(epochs, train_reconstruction_losses, 'b', label='Train')
    ax3.plot(epochs, valid_reconstruction_losses, 'r', label='Valid')
    ax3.set(xlabel="Epochs", ylabel="Loss")
    ax3.legend(prop={'size': 10})
    ax3.set_title("Reconstruction Loss")
#     ax3.set_ylim(-25000, 10000)
    
    plt.tight_layout()

    plt.savefig('./model_graphs/losses/model_losses_{}.png'.format(id))

def main():
    args = argument_parsing()
    print("Command line args:", args)

    f = open("./model_config/config_{}.json".format(args.id))
    model_config = json.load(f)
    f.close()
    
    #concatenating the data sets 
     
    train_data_T = np.load(model_config["data"]["training_data_path_T"])
    train_data_Q = np.load(model_config["data"]["training_data_path_Q"])
    train_data_W = np.load(model_config["data"]["training_data_path_W"])
    
 
    
    test_data_T = np.load(model_config["data"]["test_data_path_T"])
    test_data = test_data_T
    test_data_Q = np.load(model_config["data"]["test_data_path_Q"])
    test_data_W = np.load(model_config["data"]["test_data_path_W"])
    
    #concatentation
#     train_data_c = np.concatenate((train_data_T, train_data_Q), axis = 2)
#     train_data = np.concatenate((train_data_c, train_data_W), axis = 2)
    
#     test_data_d = np.concatenate((test_data_T, test_data_Q), axis = 1)
#     test_data = np.concatenate((test_data_d, test_data_W), axis = 1)
    
    train_data = np.stack((train_data_T, train_data_Q, train_data_W), axis=1, out=None)
    test_data = np.stack((test_data_T, test_data_Q, test_data_W), axis=1, out=None)
     
     
    
    print("train data shape ========", train_data.shape)
    print("test data shape ==========", test_data.shape)
       
    img_width = train_data.shape[1]
    img_height = train_data.shape[2]
    img_depth = train_data.shape[3];
    print("Image shape:", img_width, img_height, img_depth) 
    
    print("BREAK====================================lsjflsjdlfkjs")
    # Construct VAE Encoder 
    encoder_result = encoder_gen((img_width, img_height, img_depth), model_config["encoder"], args.id)
    # Construct VAE Decoder 
    vae_decoder = decoder_gen( (img_width, img_height, img_depth),model_config["decoder"])
    
    plot_model(vae_decoder, to_file='./model_graphs/model_diagrams/decoder_{}.png'.format(args.id), show_shapes=True)
    _, _, z = encoder_result.vae_encoder(encoder_result.inputs)
    x_mu_log_var = vae_decoder(z)
    vae = keras.Model(inputs=[encoder_result.inputs], outputs=[x_mu_log_var])
    plot_model(vae, to_file='./model_graphs/model_diagrams/full_vae_{}.png'.format(args.id), show_shapes=True)
    vae.kl_weight = K.variable(model_config["kl_weight"])

    # Specify the optimizer 
    optimizer = keras.optimizers.Adam(lr=model_config['optimizer']['lr'])

    # Compile model 
    vae.compile(
        # loss=reconstruction, 
        loss=kl_reconstruction_loss(
            encoder_result.z_log_var, 
            encoder_result.z_mean, 
            vae
            # stat_weight
        ), 
        optimizer=optimizer, 
        metrics=[
            reconstruction, 
            kl(
                encoder_result.z_log_var, 
                encoder_result.z_mean
            ), 
            kl_reconstruction_loss(
                encoder_result.z_log_var, 
                encoder_result.z_mean, 
                vae
                # stat_weight
            )#,
#             constrainer(
#                 encoder_result.z_log_var, 
#                 encoder_result.z_mean,
#                 stat_weight
#             )
        ]
    )
    vae.summary()

    train_data = train_data.reshape(train_data.shape+(1,))
    test_data = test_data.reshape(test_data.shape+(1,))

    print("train data shape", train_data.shape)
    print("test data shape", test_data.shape)


    checkpoint = ModelCheckpoint(
        './models/model_{}.th'.format(args.id), 
        monitor='val_loss', 
        verbose=1,
        save_best_only=True,
        save_weights_only=True 
    )
    callbacks_list = [checkpoint]

    if model_config["annealing"]:
        kl_weight_annealing = AnnealingCallback(model_config["train_epochs"])
        callbacks_list.append(kl_weight_annealing)

    h = vae.fit(
        x=train_data, 
        y=train_data, 
        epochs=model_config["train_epochs"], 
        batch_size=model_config["batch_size"], 
        validation_data=[test_data, test_data],
        callbacks=callbacks_list
    )

    plot_training_losses(h, args.id)

def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, help='This option specifies the config file to use to construct and train the VAE.')

    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    main()
