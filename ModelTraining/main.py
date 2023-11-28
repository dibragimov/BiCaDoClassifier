# This is a sample Python script.
import os
import shutil
import random
import numpy as np
import logging
import tensorflow as tf
import argparse
import helper.file_utility
import helper.plot_utility
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
def train_image_classifier(base_dir):
    logging.info(f'Training, using {base_dir}')
    # Check if one file exists. If it does not exist - download files
    need_preparation = False
    if not os.path.exists(os.path.join(base_dir, 'kagglecatsanddogs_3367a.zip')):
        need_preparation = True
        logging.info('File 1 is downloading')
        os.system('wget -c https://dmmlcnndata.s3.amazonaws.com/kagglecatsanddogs_3367a.zip -P ' + base_dir)
    # second check - after this if both files exist it will not prepare files.
    # if either one is missing - system is not ready - shall prepare files
    if not os.path.exists(os.path.join(base_dir, 'kagglecatsanddogs_3367a.zip')):
        need_preparation = True
        logging.info('File 2 is downloading')
        os.system('wget -c https://dmmlcnndata.s3.amazonaws.com/CUB_200_2011.tar -P ' + base_dir)

    # run filehelper to unzip and create train/test
    if need_preparation or not os.path.exists(os.path.join(base_dir, 'PetImages')):
        logging.info('Unzipping files')
        helper.file_utility.prepare_files(os.path.join(base_dir, 'kagglecatsanddogs_3367a.zip'),
                                          os.path.join(base_dir, 'CUB_200_2011.tar'),
                                          base_dir)

    # creating image generators
    train_idg = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255, rotation_range=35,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                                                fill_mode='nearest', horizontal_flip=True)
    training_generator = train_idg.flow_from_directory(os.path.join(base_dir, 'train'), class_mode='categorical',
                                                       batch_size=64, target_size=(150, 150))

    valid_idg = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    valid_generator = valid_idg.flow_from_directory(os.path.join(base_dir, 'valid'), class_mode='categorical',
                                                    batch_size=64, target_size=(150, 150))
    # defining a model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(96, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(96, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    # compiling a model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # define callbacks
    loss_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=1e-3)
    # start training
    history = model.fit(training_generator,  epochs=50,
                        validation_data=valid_generator,
                        callbacks=[loss_callback])

    # save model. Make sure folders exist
    model_dir = os.path.join(base_dir, 'custom_model')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    version = 1  # version of the model
    export_path = os.path.join(model_dir, str(version))
    if not os.path.exists(export_path):
        os.mkdir(export_path)
    logging.info(f'export_path = {export_path}\n')
    # Save the model
    tf.keras.models.save_model(
        model,
        export_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )
    # draw and save graphs
    helper.plot_utility.plot_and_save(history, save_dir=base_dir,
                                      experiment_name='custom_training',
                                      params=['accuracy', 'loss'])


# training transfer learning
def train_transfer_learning_image_classifier(base_dir):
    logging.info(f'Training Inception_V3 model, using {base_dir}')
    # Check if one file exists. If it does not exist - download files
    inception_v3_weights_file = helper.file_utility.load_inception_weights(base_dir)
    # inception model
    pretrained_model = tf.keras.applications.inception_v3.InceptionV3(input_shape=(150, 150, 3),
                                                                      include_top=False,
                                                                      weights=None)
    pretrained_model.load_weights(inception_v3_weights_file)
    # make all layers static
    for layer in pretrained_model.layers:
        layer.trainable = False
    logging.debug('Loaded Inception Model:')
    # logger = logging.getLogger(__name__)
    pretrained_model.summary(print_fn=logging.info)
    # Choose `mixed_7` as the last layer of your base model
    last_layer = pretrained_model.get_layer('mixed7')
    logging.info('last layer output shape: ', last_layer.output_shape)
    last_layer_output = last_layer.output
    # define a model's layers
    dnn_vect = tf.keras.layers.Flatten()(last_layer_output)
    drp_vect = tf.keras.layers.Dropout(0.3)(dnn_vect)
    hidden_vect = tf.keras.layers.Dense(128, activation='relu')(drp_vect)
    class_vect = tf.keras.layers.Dense(3, activation='softmax')(hidden_vect)
    # define a model
    inception_model = tf.keras.models.Model(pretrained_model.input, outputs=class_vect)
    logging.debug('New Inception Model:')
    inception_model.summary(print_fn=logging.info)

    # creating image generators
    train_idg = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255, rotation_range=35,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                                                fill_mode='nearest', horizontal_flip=True)
    training_generator = train_idg.flow_from_directory(os.path.join(base_dir, 'train'), class_mode='categorical',
                                                       batch_size=64, target_size=(150, 150))

    valid_idg = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    valid_generator = valid_idg.flow_from_directory(os.path.join(base_dir, 'valid'), class_mode='categorical',
                                                    batch_size=64, target_size=(150, 150))
    # compiling a model
    inception_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # define callbacks
    loss_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=1e-3)
    # start training
    history = inception_model.fit(training_generator,  epochs=8,
                        validation_data=valid_generator,
                        callbacks=[loss_callback])

    # save model. Make sure folders exist
    model_dir = os.path.join(base_dir, 'inception_model')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    version = 1  # version of the model
    export_path = os.path.join(model_dir, str(version))
    if not os.path.exists(export_path):
        os.mkdir(export_path)
    logging.info(f'export_path = {export_path}\n')
    # Save the model
    tf.keras.models.save_model(
        inception_model,
        export_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )
    # draw and save graphs
    helper.plot_utility.plot_and_save(history, save_dir=base_dir,
                                      experiment_name='inception_model_training',
                                      params=['accuracy', 'loss'])


if __name__ == '__main__':
    logging.basicConfig(filename='training.log', encoding='utf-8', level=logging.INFO)
    parser = argparse.ArgumentParser('BiCaDo Model Training')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Name of the folder to store image data')
    parser.add_argument('--training_type', type=str, required=True,
                        help='Type of training: inception_v3 or custom')
    args, unknown_args = parser.parse_known_args()
    training_type = args.training_type.lower()
    if args.data_dir:
        logging.info(f'Data Folder: {args.data_dir}')
    # start training
    with tf.device('/GPU:0'):
        if training_type == 'custom':
            train_image_classifier(args.data_dir)
        elif training_type == 'inception_v3':
            train_transfer_learning_image_classifier(args.data_dir)
        else:
            logging.error(f'Wrong training type: {training_type}. Exiting')
            print('Wrong training type. Exiting')
            exit(1)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
