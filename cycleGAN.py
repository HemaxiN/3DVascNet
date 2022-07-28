##https://machinelearningmastery.com/cyclegan-tutorial-with-keras/

#dataset directory
#images_3D
#testA desired domain images
#testB new images to be converted
#trainA desired domain images
#trainB new images to be converted

# example of training a cyclegan on the horse2zebra dataset
from random import random
from numpy import load
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv3D
from keras.layers import Conv3DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from matplotlib import pyplot
import tifffile
import numpy as np
import os
import pandas as pd

#from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
# define layer
layer = InstanceNormalization(axis=-1)

def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_image = Input(shape=image_shape)
    # C64
    d = Conv3D(32, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv3D(64, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv3D(128, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv3D(256, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv3D(512, (4,4,4), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    patch_out = Conv3D(1, (4,4,4), padding='same', kernel_initializer=init)(d)
    # define model
    model = Model(in_image, patch_out)
    # compile model
    model.compile(loss='mse', optimizer=Adam(lr=0.0001, beta_1=0.5), loss_weights=[0.5])
    return model



# generator a resnet block
def resnet_block(n_filters, input_layer):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # first layer convolutional layer
    g = Conv3D(n_filters, (3,3,3), padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # second convolutional layer
    g = Conv3D(n_filters, (3,3,3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    # concatenate merge channel-wise with input layer
    g = Concatenate()([g, input_layer])
    return g


# define the standalone generator model
def define_generator(image_shape, n_resnet=2):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # c7s1-64
    g = Conv3D(8, (3,3,3), padding='same', kernel_initializer=init)(in_image)  #16/06/2022, from (7,7,7) to (3,3,3)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d128
    g = Conv3D(16, (3,3,3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d256
    g = Conv3D(32, (3,3,3), strides=(2,2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # R256
    for _ in range(n_resnet):
        g = resnet_block(64, g)
    # u128
    g = Conv3DTranspose(32, (3,3,3), strides=(2,2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # u64
    g = Conv3DTranspose(16, (3,3,3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # c7s1-3
    g = Conv3D(1, (3,3,3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model



# define a composite model for updating generators by adversarial and cycle loss
def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
    # ensure the model we're updating is trainable
    g_model_1.trainable = True
    # mark discriminator as not trainable
    d_model.trainable = False
    # mark other generator model as not trainable
    g_model_2.trainable = False
    # discriminator element
    input_gen = Input(shape=image_shape)
    gen1_out = g_model_1(input_gen)
    output_d = d_model(gen1_out)
    # identity element
    input_id = Input(shape=image_shape)
    output_id = g_model_1(input_id)
    # forward cycle
    output_f = g_model_2(gen1_out)
    # backward cycle
    gen2_out = g_model_2(input_id)
    output_b = g_model_1(gen2_out)
    # define model graph
    model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
    # define optimization algorithm configuration
    opt = Adam(lr=0.0001, beta_1=0.5)
    # compile model with weighting of least squares loss and L1 loss
    model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
    return model


def load_real_samples2(ix, img_shape, dir_):
    X1=zeros((len(ix),img_shape[0],img_shape[1],img_shape[2],img_shape[3]),dtype='float32')
    X2=zeros((len(ix)img_shape[0],img_shape[1],img_shape[2],img_shape[3]),dtype='float32')
    k=0
    for i in ix:
        image = tifffile.imread(os.path.join(dir_, 'images/'+str(i)+'.tif')) # RGB image
        mask = tifffile.imread(os.path.join(dir_, 'masks/'+str(i)+'.tif')) # RGB image

        image = np.expand_dims(image, axis=-1)
        mask = np.expand_dims(mask, axis=-1)


        X1[k,:]=(image-127.5) /127.5
        X2[k,:]=(mask-127.5) /127.5
        k=k+1
    return [X1, X2]


# select a batch of random samples, returns images and target
#def generate_real_samples(dataset, n_samples, patch_shape):
def generate_real_samples(n_patches, n_samples, patch_shape, dir_, image_shape):
    # unpack dataset
    # choose random instances
    ix = randint(0, n_patches, n_samples)
    # retrieve selected images
    X1, X2 = load_real_samples2(ix, image_shape, dir_)
    # generate 'real' class labels (1)
    y = ones((n_samples, 4, patch_shape, patch_shape, 1))
    return X1, X2, y

# select a batch of random samples, returns images and target
#def generate_real_samples(dataset, n_samples, patch_shape):
def generate_real_samples2(ix, n_samples, patch_shape, dir_, image_shape):
    # unpack dataset
    #trainA, trainB = dataset
    # choose random instances
    # retrieve selected images
    X1, X2 = load_real_samples2(ix, image_shape, dir_)
    # generate 'real' class labels (1)
    y = ones((n_samples, 4, patch_shape, patch_shape, 1))
    return X1, X2, y


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), 4, patch_shape, patch_shape, 1))
    return X, y


#  save the generator models to file
def save_models(step, g_model_AtoB, g_model_BtoA):
    # save the first generator model
    filename1 = 'g_model_AtoB_%06d.h5' % (step+1)
    g_model_AtoB.save(filename1)
    # save the second generator model
    filename2 = 'g_model_BtoA_%06d.h5' % (step+1)
    g_model_BtoA.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))


# update image pool for fake images
def update_image_pool(pool, images, max_size=50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            # stock the pool
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
            # use image, but don't add it to the pool
            selected.append(image)
        else:
            # replace an existing image and use replaced image
            ix = randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    return asarray(selected)


# train cyclegan models
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, n_patches, train_dir, image_shape, nbepochs, batchsize):
    # define properties of the training run
    n_epochs, n_batch, = nbepochs, batchsize
    # determine the output square shape of the discriminator
    n_patch = d_model_A.output_shape[2]
    # calculate the number of batches per training epoch
    bat_per_epo = int(n_patches / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    k=0  #number of steps

    #save the losses in a pandas df
    losses_df = pd.DataFrame(columns=["dAlosses1", "dAlosses2", "dBlosses1", "dBlosses2", "glosses1", "glosses2"])

    # manually enumerate epochs
    for i in range(n_epochs):
        # select a batch of real samples
        array_samples = np.arange(0,n_patches)
        np.random.shuffle(array_samples)
        for sample_ in array_samples:

            X_realA, _, y_realA = generate_real_samples2([sample_], n_batch, n_patch, train_dir, image_shape)
            _, X_realB, y_realB = generate_real_samples2([sample_], n_batch, n_patch, train_dir, image_shape)
            # generate a batch of fake samples
            X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
            X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
            # update fakes from pool
            #X_fakeA = update_image_pool(poolA, X_fakeA)
            #X_fakeB = update_image_pool(poolB, X_fakeB)
            # update generator B->A via adversarial and cycle loss
            g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
            # update discriminator for A -> [real/fake]
            dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
            dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
            # update generator A->B via adversarial and cycle loss
            g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
            # update discriminator for B -> [real/fake]
            dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
            dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
            # summarize performance
            print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))

            res = {"dAlosses1": dA_loss1, "dAlosses2": dA_loss2, "dBlosses1": dB_loss1, "dBlosses2": dB_loss2, "glosses1": g_loss1, "glosses2": g_loss2}
			row = len(losses_df)
			losses_df.loc[row] = res

            if (k+1) % (bat_per_epo * 1) == 0:
                # save the models
                save_models(i, g_model_AtoB, g_model_BtoA)
                losses_df.to_csv('lossescycleGAN.csv', sep=';', index=False)

            k+=1

    print('Number of steps: {}'.format(n_steps))
    print('k: {}'.format(k))