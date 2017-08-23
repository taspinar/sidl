from mnist import MNIST
from utils import *
from six.moves.urllib.request import urlopen
import gzip, tarfile
from shutil import move

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

SOURCE_URL_MNIST = 'http://yann.lecun.com/exdb/mnist/'
SOURCE_URL_CIFAR10 = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
SOURCE_URL_OXFLOWER17 = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz'

MNIST_FILES = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

CIFAR10_TRAIN_DATASETS = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', ]
CIFAR10_TEST_DATASETS = ['test_batch']
CIFAR_10_GZ_FILE  = 'cifar-10-python.tar.gz'
CIFAR_10_FOLDER = 'cifar-10-batches-py/'

def unzip_download(download_response):
    compressedFile = StringIO()
    compressedFile.write(download_response.read())
    compressedFile.seek(0)
    decompressedFile = gzip.GzipFile(fileobj=compressedFile, mode='rb')
    return decompressedFile

def mnist(input_folder, image_width, image_height, image_depth):
    if not os.path.exists(input_folder):
        os.mkdir(input_folder)
    
    for filename in MNIST_FILES:
        unzipped_filename = filename.split('.')[0]
        if unzipped_filename not in os.listdir(input_folder):
            print('Downloading MNIST file ', filename)
            response = urlopen(SOURCE_URL_MNIST + filename)
            with open(input_folder + unzipped_filename, 'wb') as outfile:
                outfile.write(gzip.decompress(response.read()))
            print('Succesfully downloaded and unzipped', filename)
    print("Loading MNIST dataset...")
    mndata = MNIST(input_folder)
    train_dataset_, train_labels_ = mndata.load_training()
    test_dataset_, test_labels_ = mndata.load_testing()
    train_dataset, train_labels = reformat_data(train_dataset_, train_labels_, image_width, image_height, image_depth)
    test_dataset, test_labels = reformat_data(test_dataset_, test_labels_, image_width, image_height, image_depth)
    print("The MNIST training dataset contains {} images, each of size {}".format(train_dataset[:,:,:,:].shape[0], train_dataset[:,:,:,:].shape[1:]))
    print("The MNIST test dataset contains {} images, each of size {}".format(test_dataset[:,:,:,:].shape[0], test_dataset[:,:,:,:].shape[1:]))
    print("There are {} number of labels.".format(len(np.unique(train_labels_))))
    return train_dataset, train_labels, test_dataset, test_labels

def cifar10(input_folder, image_width, image_height, image_depth):
    if not os.path.exists(input_folder):
        os.mkdir(input_folder)
    
    download_flag = False
    for file in [CIFAR_10_GZ_FILE] + CIFAR10_TRAIN_DATASETS + CIFAR10_TEST_DATASETS:
        if file not in os.listdir(input_folder):
            download_flag = True
            
    if download_flag:
        print("Downloading CIFAR10 dataset")
        response = urlopen(SOURCE_URL_CIFAR10)
        with open(input_folder + CIFAR_10_GZ_FILE, 'wb') as outfile:
            outfile.write(response.read())
        print('Succesfully downloaded and unzipped', CIFAR_10_GZ_FILE)
        print("extracting files...")
        tar = tarfile.open(input_folder + CIFAR_10_GZ_FILE)
        tar.extractall(input_folder)
        files = os.listdir(input_folder + CIFAR_10_FOLDER)
        for file in files:
            move(input_folder + CIFAR_10_FOLDER + file, input_folder + file)
        os.rmdir(input_folder + CIFAR_10_FOLDER)
    print("Loading Cifar-10 dataset")
    with open(input_folder + CIFAR10_TEST_DATASETS[0], 'rb') as f0:
        c10_test_dict = pickle.load(f0, encoding='bytes')

    c10_test_dataset, c10_test_labels = c10_test_dict[b'data'], c10_test_dict[b'labels']
    
    c10_train_dataset, c10_train_labels = [], []
    for train_dataset in CIFAR10_TRAIN_DATASETS:
        with open(input_folder + train_dataset, 'rb') as f0:
            c10_train_dict = pickle.load(f0, encoding='bytes')
            c10_train_dataset_, c10_train_labels_ = c10_train_dict[b'data'], c10_train_dict[b'labels']
            
            c10_train_dataset.append(c10_train_dataset_)
            c10_train_labels += c10_train_labels_

    c10_train_dataset = np.concatenate(c10_train_dataset, axis=0)
    test_dataset, test_labels = reformat_data(c10_test_dataset, c10_test_labels, image_width, image_height, image_depth)
    train_dataset, train_labels = reformat_data(c10_train_dataset, c10_train_labels, image_width, image_height, image_depth)
    print("The CIFAR-10 training dataset contains {} images, each of size {}".format(train_dataset[:,:,:,:].shape[0], train_dataset[:,:,:,:].shape[1:]))
    print("The CIFAR-10 test dataset contains {} images, each of size {}".format(test_dataset[:,:,:,:].shape[0], test_dataset[:,:,:,:].shape[1:]))
    print("There are {} number of labels.".format(len(np.unique(c10_train_labels))))
    return train_dataset, train_labels, test_dataset, test_labels