"""
This code builds a Inception-v3 model using Keras, trains and validates the model for a few epochs
and saves the best model according to validation accuracy.

If you want to run this code, please change all paths in the main.py and misc_fun.py according to your folder structure.

"""
import sys
sys.path.insert(0, '/home/mshaik')
# sys.path.insert(0, '/myNN')
from misc_fun import FLAGS
FLAGS.DEFAULT_IN = '/data/cephfs/punim0811/Datasets/iNaturalist/tfrecords_299/'
FLAGS.DEFAULT_OUT = '/home/mshaik/update2/'
FLAGS.IMAGE_FORMAT = 'channels_last'
FLAGS.IMAGE_FORMAT_ALIAS = 'NHWC'
import os
if FLAGS.MIXED_PRECISION:  # This line currently has no effects and it's safe to delete
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
from inaturalist_func import read_inaturalist
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, GlobalAveragePooling2D
from tensorflow.keras import optimizers, applications
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from tensorflow.keras.backend import set_session
#from multiplier import LRMultiplier

allow_growth = False
target_size = 299
num_classes = 1010
num_epoch = 100
buffer_size = 256
data_size = {'train': 265213, 'val': 3030, 'test': 35350}
do_save_and_load = False
init_epoch = 0
target_epoch = 100

"""
# prepare folder and ckpt files to load
inception_v3_path = FLAGS.DEFAULT_OUT + 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
ckpt_folder = FLAGS.DEFAULT_OUT + 'trial_{}/'.format(target_size)
ckpt_name_prefix = 'full_GAPDropD_Adam_LRM_0.50'
ckpt_list = [file for file in sorted(os.listdir(ckpt_folder)) if ckpt_name_prefix in file]
last_ckpt = ckpt_folder + (ckpt_list[-1] if ckpt_list else '')
ckpt_path_template = ckpt_folder + ckpt_name_prefix + '_{epoch:02d}_{val_acc:.3f}.h5'
"""

# read the dataset
# 299x299, 128, 256
# 800x800, 16, 64

dataset_tr, steps_per_tr_epoch = read_inaturalist(
    'train', batch_size=64, target_size=target_size, do_augment=True)
dataset_va, steps_per_va_epoch = read_inaturalist(
    'val', batch_size=64, target_size=target_size, do_augment=True)

if allow_growth:  # allow gpu memory to grow, for debugging purpose, safe to delete
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    set_session(sess)

# load the model
"""
if do_save_and_load and os.path.isfile(last_ckpt):
    mdl = load_model(
        last_ckpt,
        custom_objects={'softmax_cross_entropy': tf.losses.softmax_cross_entropy})
    print('Model loaded from {}'.format(last_ckpt))
else:
    base_model = applications.InceptionV3(
        weights=None,
        include_top=False,
        input_shape=(target_size, target_size, 3))
    base_model.load_weights(inception_v3_path)
    print('Model weights loaded from {}'.format(inception_v3_path))
    # Freeze some layers  # no need to freeze layers, we fine-tune all layers
    # for layer in base_model.layers:
    #     layer.trainable = False
"""

#my model													  
base_model = applications.InceptionV3(
    weights=None,
    include_top=True,
    input_shape=(target_size, target_size, 3))
base_model.load_weights('/home/mshaik/update2/inception_v3_weights_ai.h5')
																   
# Freeze some layers  # no need to freeze layers, we fine-tune all layers
# for layer in base_model.layers[:-20]:
#     layer.trainable = False
bottleneck_input = base_model.get_layer(index=0).input
bottleneck_output = base_model.get_layer(index=-2).output
mdl = Model(inputs=bottleneck_input, outputs=bottleneck_output)
mdl = Sequential([mdl, Dense(num_classes, activation='linear')])
														 
"""
    # Adding custom layers
    mdl = Sequential([
        base_model, GlobalAveragePooling2D('channels_last'), Dropout(0.50),
        Dense(num_classes, activation='linear')])
    mdl.compile(
        tf.keras.optimizers.Adam(lr=0.001),
        loss=tf.losses.softmax_cross_entropy, metrics=['accuracy'])
    # mdl.compile(
    #     LRMultiplier('adam', {'inception_v3': 0.1, 'dense': 1.0}),
    #     loss=tf.losses.softmax_cross_entropy, metrics=['accuracy'])
"""
mdl.compile(tf.keras.optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,decay=1.97e-5),loss=tf.losses.softmax_cross_entropy, metrics=['accuracy'])
mdl.summary()

# check point
checkpoint = ModelCheckpoint(
    FLAGS.DEFAULT_OUT+'trial_{}_GAPDDropD_ADAM.h5'.format(target_size), monitor='val_acc', verbose=1,
    save_best_only=True, save_weights_only=False, mode='auto', period=1)
	
ES = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')																		

# do the training
start_time = time.time()
history = mdl.fit(
    dataset_tr.dataset, epochs=num_epoch, callbacks=[checkpoint, ES], validation_data=dataset_va.dataset,
    steps_per_epoch=512, validation_steps=10, verbose=1)
duration = time.time() - start_time
print('\n The training process took {:.1f} seconds'.format(duration))
