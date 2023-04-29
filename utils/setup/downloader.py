import os
import urllib

import requests
import wget
from tqdm import tqdm
from urllib import request
import subprocess



def check_if_file_exits(file):
    """ Checks if the file specified is downloaded or not.
    Parameters:
        file(str): Name of the file to be checked.

    Returns: None
    """
    return True if os.path.isfile(file) else False


def download_file(url, path, source='kaggle'):
    """ Download the file in url to the path specified.
    Parameters:
        url(str): URL of the file to be downloaded.
        path(str): Destination where the downloaded file will be saved.
        source(str): Source of the file to be downloaded.

    Returns: None
    """
    # Check if file already exists.
    if check_if_file_exits(path):
        print(f'Already existing file {path}')
        return

    # Deleting the partial downloaded file.
    if os.path.isfile(path):
        print(f'Deleted existing partial file {path}')
        os.remove(path)

    if source == 'kaggle':
        # Downloading the dataset from Kaggle
        os.system(f'kaggle datasets download -d {url} -p {path}')
    else:
        # Downloading the dataset over HTTP
        command = ['wget', url, '-P', path]
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
                              universal_newlines=True) as process:
            progress = tqdm(unit='B', unit_scale=True, unit_divisor=1024, total=None)
            for line in process.stdout:
                progress.write(line.rstrip())
                progress.update(len(line))

        progress.close()


def make_folder(target_folder):
    """Creates folder if there is no folder in the specified path.
    Parameters:
        target_folder(str): path of the folder which needs to be created.

    Returns: None
    """
    if not (os.path.isdir(target_folder)):
        print(f'Creating {target_folder} folder')
        os.mkdir(target_folder)


def clear_screen():
    """Clears the console screen irrespective of os used"""
    import platform
    if platform.system() == 'Windows':
        os.system('cls')
        return
    os.system('clear')


def process(dataset_urls, downloads_path):
    # Clears the screen.
    clear_screen()

    # Make downloads dir if not exists
    make_folder(downloads_path)

    print('\tStarting download process')
    for url in dataset_urls:
        try:
            print(f'\t\tDownloading :  {url}')
            download_file(url, downloads_path, 'HTTP')
        except KeyboardInterrupt:
            print('\t\t\nDownload stopped')
            break
