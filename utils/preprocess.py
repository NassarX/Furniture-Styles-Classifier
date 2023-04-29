import os
import shutil

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle

from setup import downloader, unziper, extractor


def setup_dataset(dataset_url, downloads_path):
    # Download dataset
    print("[1] Downloading dataset...")
    downloader.process([dataset_url], downloads_path)

    # Unzip dataset
    print("[2] Unzipping dataset...")
    unziper.process(downloads_path)

    print("Dataset downloaded and unzipped successfully.")


def extract_contour(image):
    return extractor.extract_contour(image, True)


def extract_images(source_path, dataset_path):
    # Extract images
    print("Extracting images...")
    extractor.process(source_path, dataset_path)

    print("Images cropped successfully.")


def preprocess_data(source_path, destination_path, img_size=256):

    # Extract images
    extract_images(source_path, destination_path)

    extractor.clear_screen()

    # if os.path.exists(destination_path):
    #     shutil.rmtree(destination_path)

    # extractor.make_folder(destination_path)

    for root, dirs, files in os.walk(destination_path):
        sub_dir = os.path.relpath(root, destination_path)
        extractor.make_folder(os.path.join(destination_path, sub_dir))

        progress_bar = tqdm(total=len(files), desc=f'Processing {sub_dir} images', dynamic_ncols=True)
        # extract each file and update the progress bar
        if files:

            for file in files:
                if not file.endswith('.jpg'):
                    continue

                # Read the original image
                original_path = os.path.join(root, file)
                img = cv2.imread(original_path)
                processed_img = preprocess_image(img, img_size)

                # Save the optimized image
                processed_path = os.path.join(destination_path, sub_dir, file)
                cv2.imwrite(processed_path, processed_img)

            progress_bar.update(len(files))
        progress_bar.close()


def preprocess_image(image, size=256):
    # Resize image
    image = cv2.resize(image, dsize=(size, size), interpolation=cv2.INTER_CUBIC)

    # Convert the image to grayscale
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize the image
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # remove images noise.
    image = cv2.bilateralFilter(image, 2, 50, 50)

    image = (image * 255).astype(np.uint8)

    return image


def save_augmented_images(data, destination_path, labels):
    extractor.clear_screen()

    if os.path.exists(destination_path):
        shutil.rmtree(destination_path)

    extractor.make_folder(destination_path)

    num_batches = len(data[0])
    progress_bar = tqdm(total=num_batches, desc=f'Saving augmented images', dynamic_ncols=True)

    # Iterate over the generated batches
    for i, (batch_x, batch_y) in enumerate(zip(*data)):
        # Get the label of the image
        label = batch_y

        label_name = labels[label]
        label_dir = os.path.join(destination_path, label_name)

        # Create a directory for the label if it doesn't exist
        extractor.make_folder(label_dir)

        # batch_x = batch_x.reshape((batch_x.shape[0], 256, 256))

        # Iterate over the images in the batch
        for j in range(batch_x.shape[0]):
            # img = np.expand_dims(batch_x[j], axis=-1)
            # img = Image.fromarray(img.astype('uint8'), 'L')

            image_path = os.path.join(label_dir, f'Tr-{label_name[:2]}_{i * batch_x.shape[0] + j}.jpg')
            cv2.imwrite(image_path, batch_x[j])

        progress_bar.update(1)

        # Exit the loop if all batches have been processed
        # if i == num_batches - 1:
        #   break

    progress_bar.close()


def load_data(dataset_path, labels):
    """ Load each label dataset into list.
    Parameters:
        dataset_path(str): Name of the path for dataset.
        labels(dict): Dictionary of labels.
    Returns: 2 lists of data & labels
    """

    labels_lv1 = labels['lv1']
    labels_lv2 = labels['lv2']

    # Create an empty DataFrame
    df = pd.DataFrame(columns=['file'] + labels_lv1 + labels_lv2)

    # Loop through each label level 1 directory
    for label_lv1 in labels_lv1:
        # Define the path to the label level 1 directory
        label_lv1_path = os.path.join(dataset_path, label_lv1)

        # Loop through each label level 2 directory
        for label_lv2 in labels_lv2:
            # Define the path to the label level 2 directory
            label_lv2_path = os.path.join(label_lv1_path, label_lv2)

            # Get a list of image files in the label level 2 directory
            image_files = os.listdir(label_lv2_path)

            # Add each image file to the DataFrame with the appropriate label
            for image_file in image_files:
                row = [os.path.join(label_lv2_path, image_file)]
                row += [int(lv1 == label_lv1) for lv1 in labels_lv1]
                row += [int(lv2 == label_lv2) for lv2 in labels_lv2]
                df.loc[len(df)] = row
        print(f'Loaded {label_lv1} dataset successfully {len(df)} .')
    # Shuffle the data
    df = shuffle(df)

    return df
