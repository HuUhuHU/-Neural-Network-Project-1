"""Download MNIST dataset if not present."""
import os
import urllib.request

MNIST_DIR = os.path.join(os.path.dirname(__file__), 'dataset', 'MNIST')
FILES = {
    'train-images-idx3-ubyte.gz': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
}


def download_mnist(target_dir=MNIST_DIR):
    os.makedirs(target_dir, exist_ok=True)
    for filename, url in FILES.items():
        path = os.path.join(target_dir, filename)
        if os.path.exists(path):
            continue
        print(f'Downloading {filename} ...')
        urllib.request.urlretrieve(url, path)
        print(f'Saved to {path}')


if __name__ == '__main__':
    download_mnist()
