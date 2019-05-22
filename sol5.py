import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave

from scipy.ndimage.filters import convolve
from . import sol5_utils
# import sol5_utils
from skimage.color import rgb2gray
from scipy.misc import imread
from tensorflow.keras.layers import Input , Dense , Conv2D , Activation , Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
# from tensorflow.python.keras.layers import Input , Dense , Conv2D , Activation , Add
# from tensorflow.python.keras.models import Model
# from tensorflow.python.keras.optimizers import Adam
import random


#'From ex.1 import read_image. read_image reads a file from the project folder and representation value of 1/2 from BW or colored image'
# From Ex.1 read_image
def read_image(filename,representation):
    if representation not in [1,2]:                  # In case of invalid representation entry
        raise ("representation index error")
    image = imread(filename)                    #  Loading RGB image.
    if image.ndim == 3:
        if representation == 1:  # Turns to gray, normalize and return image.
            image = rgb2gray(image)
            max_value = np.amax(np.amax(image))
            if max_value > 1:
                image = (image / 255).astype(np.float64)
            return image
        if representation == 2:                     #  Normalize intensity and return image
            max_value = np.amax([np.amax(image[:, :, 0]), np.amax(image[:, :, 1]), np.amax(image[:, :, 2])])
            if max_value > 1:
                image = (image / 255).astype(np.float64)
            return image
    if image.ndim == 2:
        max_value = np.amax(np.amax(image))
        if max_value > 1:
            image = (image / 255).astype(np.float64)
        return image




'Inputs are:'
'Filename is a list of filenames corresponds to clean images'
'batch_size is the size of each batch of an image for each iteration of the SGD'
'Corruption func recives an image as a singal argument, returns a randomly corrupted image'
'crop_size is a tuple (height,width) specifaying the crop size'
'Output is (source_batch,target_patch)'
'each of them in the shape of (batch_size,height,width,1) Target corresponds to clean batches wheres the source are the noisy ones'
'To improve taking the random number'


def load_dataset(filenames, batch_size,corruption_func,crop_size):
    # Prepare dictionaries,arrays and generate random indices for images.
    dictionary_images = {}
    dictionary_shapes = {}
    N_filenames = len(filenames)
    shape_list = []
    filenames_ind_rand = np.random.randint(0, N_filenames,batch_size)
    # filenames_ind_rand = np.random.randint(530,1000,batch_size)
    # filenames_ind_rand = np.random.randint(435,477,batch_size)

    source_batch = np.zeros([batch_size,crop_size[0],crop_size[1],1])
    target_batch = np.zeros([batch_size,crop_size[0],crop_size[1],1])
    # Load dictionaries: one for images the other for shapes
    # Reduces time to call read_image
    # Shapes dictionary is ['filename'] = max_row-3*crop_size_row ,  max_col-3*crop_size_col
    while True:
        for i in np.arange(0,batch_size):
            filename_ind_rand = filenames_ind_rand[i]
            if filenames[filename_ind_rand] not in dictionary_images.keys():
                dictionary_images[filenames[filename_ind_rand]] = read_image(filenames[filename_ind_rand],1)
                row_tot = dictionary_images[filenames[filename_ind_rand]].shape[0]
                col_tot = dictionary_images[filenames[filename_ind_rand]].shape[1]
                dictionary_shapes[filenames[filename_ind_rand]] =  (row_tot - 3*crop_size[0],col_tot - 3*crop_size[1])

        # Use the dictionary to generate a random patch in every image
        for i in np.arange(0,batch_size):
            # Load image and subtracted shape
            filename = filenames[filenames_ind_rand[i]]
            image = dictionary_images[filename]
            max_row = dictionary_shapes[filename][0]
            max_col = dictionary_shapes[filename][1]
            # Generate random number in a cropped image
            rand_row = np.random.randint(3 * crop_size[0], max_row - 3 * crop_size[0])
            rand_col = np.random.randint(3 * crop_size[1], max_col - 3 * crop_size[1])
            # Get patch for cropped image
            indices_r_i = rand_row - np.floor(3*crop_size[0]/2).astype(np.int32)
            indices_r_f = rand_row + np.floor(3*crop_size[0]/2).astype(np.int32)
            indices_c_i = rand_col - np.floor(3*crop_size[1]/2).astype(np.int32)
            indices_c_f = rand_col + np.floor(3*crop_size[1]/2).astype(np.int32)

            # shift the indexes to the whole picture
            rand_row_shift = np.random.randint(-np.floor(3 * crop_size[0])/2, np.floor(3 * crop_size[0])/2)
            rand_col_shift = np.random.randint(-np.floor(3 * crop_size[1])/2, np.floor(3 * crop_size[0])/2)
            img_cropped_3 = image[indices_r_i+rand_row_shift:indices_r_f+rand_row_shift,indices_c_i+rand_col_shift:indices_c_f+rand_col_shift]

            # plt.figure(3);
            # plt.imshow(img_cropped_3);
            # plt.show()

            # Corrupted image
            img_corrupted = corruption_func(img_cropped_3)
            # Generating random in sub part of the image
            rand_row_cropped = np.random.randint(crop_size[0], img_corrupted.shape[0] - crop_size[0])
            rand_col_cropped = np.random.randint(crop_size[1], img_corrupted.shape[1] - crop_size[1])
            indices_r_i = rand_row_cropped - np.floor(crop_size[0]/2).astype(np.int32)
            indices_r_f = rand_row_cropped + np.floor(crop_size[0]/2).astype(np.int32)
            indices_c_i = rand_col_cropped - np.floor(crop_size[1]/2).astype(np.int32)
            indices_c_f = rand_col_cropped + np.floor(crop_size[1]/2).astype(np.int32)

            # Target batch generator
            rand_row_shift_1 = np.random.randint(-np.floor(1 * crop_size[0])/2, np.floor(1 * crop_size[0])/2)
            rand_col_shift_1 = np.random.randint(-np.floor(1 * crop_size[1])/2, np.floor(1 * crop_size[0])/2)
            img_cropped_1 = img_cropped_3[indices_r_i+rand_row_shift_1:indices_r_f+rand_row_shift_1,indices_c_i+rand_col_shift_1:indices_c_f+rand_col_shift_1]- 0.5
            # img_cropped_1 = img_cropped_3[indices_r_i:indices_r_f,indices_c_i:indices_c_f] - 0.5
            img_cropped_sub_res = np.reshape(img_cropped_1,[img_cropped_1.shape[0],img_cropped_1.shape[1],1])
            target_batch[i, :, :, :] = img_cropped_sub_res

            # plt.figure(1);
            # plt.imshow(img_cropped_1);

            # Source batch generator
            # img_corrupted_cropped_sub = img_corrupted[indices_r_i:indices_r_f,indices_c_i:indices_c_f] - 0.5
            img_corrupted_cropped_sub = img_corrupted[indices_r_i+rand_row_shift_1:indices_r_f+rand_row_shift_1,indices_c_i+rand_col_shift_1:indices_c_f+rand_col_shift_1]- 0.5
            img_corrupted_cropped_sub_res = np.reshape(img_corrupted_cropped_sub,[img_corrupted_cropped_sub.shape[0],img_corrupted_cropped_sub.shape[1],1])
            source_batch[i,:,:,:] = img_corrupted_cropped_sub_res

            # plt.figure(2);
            # plt.imshow(img_corrupted_cropped_sub);
            # plt.show()

        # plt.figure;
        # plt.imshow(target_batch[i,:,:,:]);
        # plt.show()
        # plt.figure;
        # plt.imshow(img_corrupted_cropped_sub);
        # plt.show()
        # print(dictionary_shapes)
        yield (source_batch , target_batch)

'Each resblock is conv, relu conv add (input,conv) and relu on (input+conv)'
def resblock(input_tensor, num_channels):
    conv = Conv2D (num_channels,(3,3),padding='same')(input_tensor)
    relu = Activation('relu')(conv)
    conv2 = Conv2D (num_channels,(3,3),padding='same')(relu)
    add = Add()([input_tensor,conv2])
    output_tensor = Activation('relu')(add)
    return output_tensor

'Build model as specified in the HW '
def build_nn_model(height,width,num_channels,num_res_blocks):
    inp = Input(shape=(height,width,1))
    conv = Conv2D (num_channels,(3,3),padding='same')(inp)
    block_out = Activation('relu')(conv)
    for i in np.arange(0,num_res_blocks):
        block_out = resblock(block_out,num_channels)
    conv_m2 = Conv2D (1,(3,3),padding='same')(block_out)
    add_m1 = Add()([inp,conv_m2])
    # add_m1 = Dense(1)(add_m1)
    model = Model(inputs =inp,outputs = add_m1)
    return model

def train_model(model,images,corruption_func,batch_size,steps_per_epoch,num_epochs,num_valid_samples):
# Split images to train and valid:
# Get valid and train size, 80% train 20% valid
    filenames_nump = np.array(images)
    images_length = len(images)
    train_size = np.floor(images_length * 0.8).astype(np.int64)
    valid_size = images_length - train_size
    list1 = [True] * train_size
    list2 = [False] * valid_size
    listcomb = list1 + list2
    rand_indices = np.random.choice(listcomb, images_length, replace=False)
    train_indices = np.argwhere(rand_indices == True)
    valid_indices = np.argwhere(rand_indices == False)
    train_samples = filenames_nump[train_indices][:,0]
    valid_samples = filenames_nump[valid_indices][:,0]
    train_samples_list = (train_samples)
    valid_samples_list = (valid_samples)
# Define generators
    train_generator = load_dataset(train_samples_list, batch_size, corruption_func, model.input_shape[1:3])
    valid_generator = load_dataset(valid_samples_list, batch_size, corruption_func, model.input_shape[1:3])
# Train model
    model.compile(optimizer= Adam(beta_2 = 0.9),loss='mean_squared_error')
    model.fit_generator(train_generator,steps_per_epoch,num_epochs,validation_data=valid_generator,validation_steps=num_valid_samples)
    # hist = model.fit_generator(train_generator, steps_per_epoch, num_epochs, validation_data=valid_generator,validation_steps=num_valid_samples // batch_size)
    # return hist


'restore image using predict'
def restore_image(corrupted_image,base_model):
    inp = Input(shape=(corrupted_image.shape[0],corrupted_image.shape[1],1))
    base_mod = base_model(inp)
    new_model = Model(inputs=inp,outputs= base_mod)
    image_shifted = np.reshape(corrupted_image-0.5,[corrupted_image.shape[0],corrupted_image.shape[1],1])
    image_predicted = new_model.predict(image_shifted[np.newaxis,...])[0].astype(np.float64)
    image_restored = image_predicted + 0.5
    return np.reshape(image_restored.clip(0,1),[corrupted_image.shape[0],corrupted_image.shape[1]])



'Adds gaussian noise to original image, randomly choosen between min_sigma to sigma'
def add_gaussian_noise(image,min_sigma,max_sigma):
    rand_sigma = np.random.uniform(min_sigma,max_sigma)
    gauss_noise = np.random.normal(0.0 , rand_sigma,image.shape)
    image_corrupted_rounded = np.round((gauss_noise + image)*255)
    image_corrupted_01 = np.clip(image_corrupted_rounded / 255 , 0,1)
    return image_corrupted_01

def learn_denoising_model(num_res_blocks =5,quick_mode = False):
    filenames = sol5_utils.images_for_denoising() ;
    height = 24 ; width = 24
    channels_num = 48
    model = build_nn_model(height,width,channels_num,num_res_blocks)
    corruption_func =  lambda x: add_gaussian_noise(x,0,0.2)
    batch_size = 100
    steps_per_epoch = 100
    epoch_num = 5
    num_valid_samples = 1000
    if quick_mode == True:
        batch_size = 10
        steps_per_epoch = 3
        epoch_num = 2
        num_valid_samples = 30

    # hist = train_model(model,filenames,corruption_func,batch_size,steps_per_epoch,epoch_num,num_valid_samples)
    # return model , hist

    train_model(model,filenames,corruption_func,batch_size,steps_per_epoch,epoch_num,num_valid_samples)
    return model




def add_motion_blur(image,kernel_size,angle):
    mask = sol5_utils.motion_blur_kernel(kernel_size,angle)
    img_corrupted = convolve(image,mask,mode='reflect',cval=0)
    return img_corrupted

def random_motion_blur(image,list_of_kernel_sizes):
    angle = random.uniform(0,np.pi)
    kernel_size = np.random.choice(list_of_kernel_sizes)
    img_corrupted = add_motion_blur(image,kernel_size,angle)
    img_corrupted_255 = np.round(img_corrupted * 255)
    img_corrupted_01 = np.clip(img_corrupted_255 / 255,0,1)
    # plt.figure(1) ; plt.imshow(image) ; plt.show()
    return img_corrupted_01.astype(np.float64)

def learn_deblurring_model(num_res_blocks=5, quick_mode = False):
    filenames = sol5_utils.images_for_deblurring()
    # filenames.remove(filenames[434])
    # filenames.remove(filenames[529 - 1])
    height = 16 ; width = 16
    channels_num = 32
    # num_res_blocks = 5
    model = build_nn_model(height,width,channels_num,num_res_blocks)
    corruption_func =  lambda x: random_motion_blur(x,[7])
    batch_size = 100
    steps_per_epoch = 100
    epoch_num = 10
    num_valid_samples = 1000
    if quick_mode == True:
        batch_size = 10
        steps_per_epoch = 3
        epoch_num = 2
        num_valid_samples = 30
    # hist = train_model(model,filenames,corruption_func,batch_size,steps_per_epoch,epoch_num,num_valid_samples)
    # return  model , hist

    train_model(model,filenames,corruption_func,batch_size,steps_per_epoch,epoch_num,num_valid_samples)
    return  model


