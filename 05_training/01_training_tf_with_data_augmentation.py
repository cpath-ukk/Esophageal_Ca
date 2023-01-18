'''
Training script for InceptionResNet networks
for 11 tissue classes.
Data augmentation.
Tensorflow 1.11
'''



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 19:33:09 2018

@author: dr_pusher
"""
#Found 148782 images belonging to 3 classes.


import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras import models, layers, optimizers
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from random import randint

###
###AUGMENTATION PIPELINE
###
m_p_s = 220
b_s = 100 #Batch size global
THRES_MAIN = 4 #Global threshold for data augmentation (0...9). 9 means no augmentation.
THRES = 5 #Threshold for single data augmentation techniques (0...9)
n_classes = 11

train_dir = ''

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
	train_dir,
	image_size=(m_p_s,m_p_s),
	labels = "inferred",
	seed = 123,
	label_mode = "categorical",
	batch_size=1)

test_dir = ''

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
	test_dir,
	image_size=(m_p_s,m_p_s),
	labels = "inferred",
	seed = 123,
	label_mode = "categorical",
	batch_size=1)

##Parameters augmentation
p_br = 0.25 #0.3 as single was good
cntr_min = 0.7 #0.5 as single was good
cntr_max = 1.5
p_hue = 0.1 #0.05 and 0.1 is possible as well
sat_min = 0.3
sat_max= 1.3 #2.0 as single was good

def rescale (image):
	image = tf.cast(image, tf.float32)
	image = (image / 255.0)
	return image
	
rng = tf.random.Generator.from_seed(123, alg='philox')

def augment(image, label):
	
    image = rescale (image)
    
    i = randint(0,9)
    if i > THRES_MAIN:
        i = randint(0,9)
        if i > THRES: #threshold to apply advanced transformations, e.g. 50%
            image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=1, maxval=4, dtype=tf.int32))
            
        i = randint(0,9)
        if i > THRES: #threshold to apply advanced transformations, e.g. 50%
            image = tf.image.random_flip_left_right(image)
        
        i = randint(0,9)
        if i > THRES: #threshold to apply advanced transformations, e.g. 50%
            image = tf.image.random_flip_up_down(image)
        
        i = randint(0,9)
        if i > THRES: #threshold to apply advanced transformations, e.g. 50%
            seed=rng.make_seeds(2)[0]
            image = tf.image.stateless_random_brightness(image, p_br, seed)
        
        i = randint(0,9)
        if i > THRES: #threshold to apply advanced transformations, e.g. 50%        
            seed=rng.make_seeds(2)[0]
            image = tf.image.stateless_random_contrast(image, cntr_min, cntr_max, seed)
            
        i = randint(0,9)
        if i > THRES: #threshold to apply advanced transformations, e.g. 50%   
            seed=rng.make_seeds(2)[0]
            image = tf.image.stateless_random_hue(image, p_hue, seed)
        
        i = randint(0,9)
        if i > THRES: #threshold to apply advanced transformations, e.g. 50%   
            seed=rng.make_seeds(2)[0]
            image = tf.image.stateless_random_saturation(image, sat_min, sat_max, seed)
    
    image = tf.clip_by_value(image, 0, 1)
    return image, label


def augment_test(image, label):	
    image = rescale (image)
    return image, label
    

AUTOTUNE = tf.data.AUTOTUNE

train_gen = (
    train_ds
    .map(augment, num_parallel_calls=AUTOTUNE)
	.shuffle(10000)
	.batch(b_s)
)

train_gen = train_gen.map(lambda x, y: (tf.reshape(x, [b_s,m_p_s,m_p_s,3]), tf.reshape(y, [b_s, n_classes])))
train_gen = train_gen.prefetch(AUTOTUNE)

test_gen = (
    test_ds
    .map(augment_test, num_parallel_calls=AUTOTUNE)
	.shuffle(1500)
	.batch(b_s)
)

test_gen = test_gen.map(lambda x, y: (tf.reshape(x, [b_s,m_p_s,m_p_s,3]), tf.reshape(y, [b_s, n_classes])))
test_gen = test_gen.prefetch(AUTOTUNE)


###
###TRAINING PIPELINE
###

#Verteilung des Trainings zwischen mehreren GPU Karten
my_strategy = tf.distribute.MirroredStrategy()
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


with my_strategy.scope():
    #Import von Architektur
    conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(m_p_s,m_p_s,3))
    #Type of model, should be "Sequential"
    model = models.Sequential()
    
    #Construction of the model 
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation = 'relu'))
    model.add(layers.Dense(11, dtype='float32', activation = 'softmax'))
    
    conv_base.trainable = False
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-3), metrics=['acc'])
    

#conv_base.summary()
model.summary()

#Get model architecture
#stringlist = []
#conv_base.summary(print_fn=lambda x: stringlist.append(x))
#short_model_summary = "\n".join(stringlist)
#print(short_model_summary)

history = model.fit(train_gen, epochs=1, steps_per_epoch = 18061, validation_data = test_gen, validation_steps = 1330)
history = model.fit(train_gen, epochs=1, steps_per_epoch = 18061, validation_data = test_gen, validation_steps = 1330)
history = model.fit(train_gen, epochs=1, steps_per_epoch = 18061, validation_data = test_gen, validation_steps = 1330)
model.load_weights('OES_IRNV2_V0E3.weights')
#model.save('OES_IRNV2_V0E5.h5')



#v1, up to block8_1_conv
conv_base.trainable = True
set_trainable = False

with my_strategy.scope():

    for layer in conv_base.layers:
        if layer.name == 'block8_1_conv':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-3), metrics=['acc'])

model.summary()
history = model.fit(train_gen, epochs=1, steps_per_epoch = 18061, validation_data = test_gen, validation_steps = 1330)
history = model.fit(train_gen, epochs=1, steps_per_epoch = 18061, validation_data = test_gen, validation_steps = 1330)
history = model.fit(train_gen, epochs=1, steps_per_epoch = 18061, validation_data = test_gen, validation_steps = 1330)
model.load_weights('OES_IRNV2_V1E3.weights')
#model.save('OES_IRNV2_V1E5.h5')


#v2, up to block17_10_conv
conv_base.trainable = True
set_trainable = False

with my_strategy.scope():

    for layer in conv_base.layers:
        if layer.name == 'block17_10_conv':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-3), metrics=['acc'])
    
model.summary()

history = model.fit(train_gen, epochs=1, steps_per_epoch = 18061, validation_data = test_gen, validation_steps = 1330)
history = model.fit(train_gen, epochs=1, steps_per_epoch = 18061, validation_data = test_gen, validation_steps = 1330)

model.save_weights('OES_IRNV2_V2E2.weights')
#model.save('OES_IRNV2_V2E3.h5')


#v3, up to conv2d_78
conv_base.trainable = True
set_trainable = False

with my_strategy.scope():

    for layer in conv_base.layers:
        if layer.name == 'conv2d_78':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-3), metrics=['acc'])
    
model.summary()
history = model.fit(train_gen, epochs=1, steps_per_epoch = 18061, validation_data = test_gen, validation_steps = 1330)
history = model.fit(train_gen, epochs=1, steps_per_epoch = 18061, validation_data = test_gen, validation_steps = 1330)

model.load_weights('OES_IRNV2_V3E2.weights')
#model.save('OES_IRNV2_V3E3.h5')



#v4, up to block35_1_conv
conv_base.trainable = True
set_trainable = False

with my_strategy.scope():

    for layer in conv_base.layers:
        if layer.name == 'block35_1_conv':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-3), metrics=['acc'])
model.summary()
history = model.fit(train_gen, epochs=1, steps_per_epoch = 18061)#, validation_data = test_gen, validation_steps = 1330)
history = model.fit(train_gen, epochs=1, steps_per_epoch = 18061)#, validation_data = test_gen, validation_steps = 1330)

model.save_weights('OES_IRNV2_V4E2.weights')
#model.save('OES_IRNV2_V4E3.h5')


#v5, all
with my_strategy.scope():

    model.trainable = True
    conv_base.trainable = True    
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-3), metrics=['acc'])
    
model.summary()

history = model.fit(train_gen, epochs=1, steps_per_epoch = 18061)#, validation_data = test_gen, validation_steps = 1330)
history = model.fit(train_gen, epochs=1, steps_per_epoch = 18061)#, validation_data = test_gen, validation_steps = 1330)

model.save_weights('OES_IRNV2_V5E2.weights')
model.save('OES_IRNV2_V5E2.h5')

with my_strategy.scope():

    model.trainable = True
    conv_base.trainable = True    
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4), metrics=['acc'])

history = model.fit(train_gen, epochs=1, steps_per_epoch = 18061)#, validation_data = test_gen, validation_steps = 1330)
model.save_weights('OES_IRNV2_V5E3.weights')
model.save('OES_IRNV2_V5E3.h5')
history = model.fit(train_gen, epochs=1, steps_per_epoch = 18061)#, validation_data = test_gen, validation_steps = 1330)
model.save_weights('OES_IRNV2_V5E4.weights')
model.save('OES_IRNV2_V5E4.h5')


with my_strategy.scope():

    model.trainable = True
    conv_base.trainable = True    
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-5), metrics=['acc'])


history = model.fit(train_gen, epochs=1, steps_per_epoch = 18061)#, validation_data = test_gen, validation_steps = 1330)
model.save_weights('OES_IRNV2_V5E5.weights')
model.save('OES_IRNV2_V5E5.h5')
history = model.fit(train_gen, epochs=1, steps_per_epoch = 18061)#, validation_data = test_gen, validation_steps = 1330)
model.save_weights('OES_IRNV2_V5E6.weights')
model.save('OES_IRNV2_V5E6.h5')

with my_strategy.scope():

    model.trainable = True
    conv_base.trainable = True    
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-6), metrics=['acc'])


history = model.fit(train_gen, epochs=1, steps_per_epoch = 18061)#, validation_data = test_gen, validation_steps = 1330)
model.save_weights('OES_IRNV2_V5E7.weights')
model.save('OES_IRNV2_V5E7.h5')


with my_strategy.scope():

    model.trainable = True
    conv_base.trainable = True    
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-7), metrics=['acc'])


history = model.fit(train_gen, epochs=1, steps_per_epoch = 18061)#, validation_data = test_gen, validation_steps = 1330)
model.save_weights('OES_IRNV2_V5E8.weights')
model.save('OES_IRNV2_V5E8.h5')
