import os
import zipfile
import tarfile
import shutil
import random
import logging


def move_to_destination(origin, destination, percentage_split):
    rand_init = 40  # for reproducibility
    random.seed(rand_init)
    num_images = int(len(os.listdir(origin))*percentage_split)
    image_names = random.sample(os.listdir(origin), num_images)
    for image_name in image_names:
        shutil.move(os.path.join(origin, image_name), destination)


def prepare_files(cat_dog_source, bird_source, base_dir, percent_to_move=0.8):
    with zipfile.ZipFile(cat_dog_source, mode='r') as zfl:
        zfl.extractall(path=base_dir)

    with tarfile.TarFile(bird_source, mode='r') as trfl:
        trfl.extractall(path=base_dir)

    base_dogs_dir = os.path.join(base_dir, 'PetImages/Dog')
    base_cats_dir = os.path.join(base_dir, 'PetImages/Cat')

    # moving all bird images to a single directory (PetImages/Bird will be used for consistency)
    raw_birds_dir = os.path.join(base_dir, 'CUB_200_2011/images')
    base_birds_dir = os.path.join(base_dir, 'PetImages/Bird')
    if not os.path.exists(base_birds_dir):
        os.mkdir(base_birds_dir)

    for subdir in os.listdir(raw_birds_dir):
        subdir_path = os.path.join(raw_birds_dir, subdir)
        for image in os.listdir(subdir_path):
            shutil.move(os.path.join(subdir_path, image), os.path.join(base_birds_dir))

    logging.info(f"There are {len(os.listdir(base_birds_dir))} images of birds")
    # remove corrupt images

    os.system('find ' + base_dir + ' -size 0 - exec rm {} +')
    os.system('find ' + base_dir + ' -type f ! -name "*.jpg" - exec rm {} +')

    train_eval_dirs = ['train/cats', 'train/dogs', 'train/birds',
                       'valid/cats', 'valid/dogs', 'valid/birds']

    for direc in train_eval_dirs:
        if not os.path.exists(os.path.join(base_dir, direc)):
            os.makedirs(os.path.join(base_dir, direc))

    # Move % of the images to the train dir
    move_to_destination(base_cats_dir, os.path.join(base_dir, 'train/cats'), percent_to_move)
    move_to_destination(base_dogs_dir, os.path.join(base_dir, 'train/dogs'), percent_to_move)
    move_to_destination(base_birds_dir, os.path.join(base_dir, 'train/birds'), percent_to_move)

    # Move the remaining images to the eval dir
    move_to_destination(base_cats_dir, os.path.join(base_dir, 'valid/cats'), 1)
    move_to_destination(base_dogs_dir, os.path.join(base_dir, 'valid/dogs'), 1)
    move_to_destination(base_birds_dir, os.path.join(base_dir, 'valid/birds'), 1)

    # remove corrupt images
    for fldr in train_eval_dirs:
        for fl in os.listdir(os.path.join(base_dir, fldr)):
            if os.path.getsize(os.path.join(os.path.join(base_dir, fldr), fl)) == 0:
                os.remove(os.path.join(os.path.join(base_dir, fldr), fl))
                logging.info(f'deleting file {fl} - empty')
            if not fl.endswith('.jpg'):
                os.remove(os.path.join(os.path.join(base_dir, fldr), fl))
                logging.info(f'deleting file {fl} - not picture')


def load_inception_weights(base_dir):
    if not os.path.exists(os.path.join(base_dir, 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')):
        need_preparation = True
        logging.info('File with weights is downloading')
        os.system('wget --no-check-certificate -c https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 -P ' + base_dir)
    return os.path.join(base_dir, 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
