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


if __name__ == '__main__':
    logging.basicConfig(filename='training.log', encoding='utf-8', level=logging.INFO)
    parser = argparse.ArgumentParser('BiCaDo Model Training')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Name of the folder to store image data')
    args, unknown_args = parser.parse_known_args()
    if args.data_dir:
        logging.info(f'Data Folder: {args.data_dir}')
    # start training
    with tf.device('/GPU:0'):
        train_image_classifier(args.data_dir)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
