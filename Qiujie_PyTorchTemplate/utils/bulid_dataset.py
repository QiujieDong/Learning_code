"""Split the signal dataset into train/val and resize images to SIZE x SIZE.

Example:
    The SIGNS dataset comes into the following format:
        train_signs/
            0_IMG_5864.jpg
            ...
        test_signs/
            0_IMG_5942.jpg
            ...

    Original images have size (3024, 3024).
    Resizing to (SIZE, SIZE) reduces the dataset size from 1.16 GB to 4.7 MB, and loading smaller images
    makes training faster.

To split dataset into train and val sets. We should obtain test from elsewhere.
Because The data with label is precious.

train_data : val_data = 8 : 2

"""


import argparse
import random
import os

from PIL import Image
from tqdm import tqdm

SIZE = 64

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='/data/SIGNS_data/SIGNS',
                    help="Directory with the SIGNS dataset")
parser.add_argument('--output_dir', default='/data/SIGNS_data/64x64_SIGNS',
                    help="Where to write the new data")


def resize_and_save(filename, output_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(output_dir, filename.split('/')[-1]))


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(
        args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # TODO: Rewriting based on different data sets

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir, 'train_signs')
    test_data_dir = os.path.join(args.data_dir, 'test_signs')

    # Get the filenames in each directory (train and test)
    filenames = os.listdir(train_data_dir)
    filenames = [os.path.join(train_data_dir, f)
                 for f in filenames if f.endswith('.jpg')]

    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, f)
                      for f in test_filenames if f.endswith('.jpg')]

    # Split the images in 'train_signs' into 80% train and 20% val
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(230)
    filenames.sort()  # 排序
    random.shuffle(filenames)

    split = int(0.8 * len(filenames))
    train_filenames = filenames[:split]
    val_filenames = filenames[split:]

    filenames = {'train': train_filenames,
                 'val': val_filenames,
                 'test': test_filenames}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, val and test
    for split in ['train', 'val', 'test']:
        output_dir_split = os.path.join(
            args.output_dir, '{}_signs'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(
            split, output_dir_split))
        for filename in tqdm(filenames[split]):
            resize_and_save(filename, output_dir_split, size=SIZE)

    print("Done building dataset")
