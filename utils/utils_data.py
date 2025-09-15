import numpy as np
import math
import skimage
import skimage.transform

import torch
import torch.utils
import torch.utils.data

def normalize(image, mean=[149.91262888, 125.60738922, 155.41683322], std=[64.47969016, 65.98274776, 53.71798459],
              inverse=False):
    """Global normalization
    
    Args:
        image: [h, w, c]
        inverse: if True, reconstruct uint8 image from normalized image
    """
    image_n = np.zeros_like(image).astype(float)
    n_channels = image.shape[2]
    
    if not inverse:
        for ch in range(n_channels):
            image_n[..., ch] = (image[..., ch] - mean[ch])/std[ch]
    else:
        for ch in range(n_channels):
            image_n[..., ch] = image[..., ch] * std[ch] + mean[ch]
        image_n = np.clip(image_n, 0, 255).astype(np.uint8)
    return image_n


def augmentor(image, label=None):
    """Do image shape and color augmentation.
    
    Args:
        image: [h, w, c], float
    """
    n_channels = image.shape[2]
    
    # Shape augmentation
    if np.random.uniform(0, 1) < 0.8:
        # Projective transform
        scale = np.random.uniform(0.9, 1.1)
        aspect_ratio = np.random.uniform(0.9, 1.1)
        rotation = np.random.uniform(-0.1, 0.1)
        translationX = np.random.uniform(-10, 10)
        translationY = np.random.uniform(-10, 10)
        g = np.random.uniform(-0.001, 0.001)
        h = np.random.uniform(-0.001, 0.001)

        matrix = np.array([[math.cos(rotation) * scale * aspect_ratio, -math.sin(rotation), translationX],
                          [math.sin(rotation), math.cos(rotation) * scale / aspect_ratio, translationY],
                          [g, h, 1]])
        tform = skimage.transform.ProjectiveTransform(matrix=matrix)
        image_aug = np.zeros_like(image, dtype=np.float)
        for ch in range(n_channels):
            image_aug[..., ch] = skimage.transform.warp(image[..., ch], tform, preserve_range=True)
        
        image = image_aug
        
        if label is not None:
            label_aug = np.zeros_like(label, dtype=np.int)
            label_aug = skimage.transform.warp(label, tform, preserve_range=True, order=0)
            label = label_aug
    if np.random.uniform(0, 1) < 0.5:
        # Do 50% vertical flipping
        image = image[::-1, :, :]
        
        if label is not None:
            label = label[::-1, :]
    if np.random.uniform(0, 1) < 0.5:
        # Do 50% horizontal flipping
        image = image[:, ::-1, :]
        
        if label is not None:
            label = label[:, ::-1]

    # Color augmentation
    # 1) add a global shifting for all channels
    image = image + np.random.randn(1)[0] * 0.1

    # 2) add a shifting & variance for each channel
    for ch in range(n_channels):
        image[:, :, ch] = image[:, :, ch] * np.clip(np.random.randn(1)[0] * 0.2 + 1, 0.8, 1.2) + np.random.randn(1)[0] * 0.3
        
    if label is not None:
        return image, label
    else:
        return image


class Dataset(torch.utils.data.Dataset):
    __initialized = False
    def __init__(self, indexes, image_dict, count_df, augmentation=False):
        """
        Args:
            indexes: index used for both image_dict and count_df
        """
        self.indexes = indexes
        self.indexes_dict = dict()
        for i, _ in enumerate(indexes):
            self.indexes_dict[_] = i
        
        self.image_dict = image_dict
        self.count_df = count_df
        self.augmentation = augmentation
        self.__initialized = True

    def __len__(self):
        """Denotes the number of samples"""
        return len(self.indexes)
    
    def __getitem__(self, index):
        """Generate one batch of data.
        
        Returns:
            idx: indexes of samples (long)
        """
        # Generate indexes of the batch
        data_index = self.indexes[index]
        
        # Generate torch.long indexes of the batch samples
        idx = self.indexes_dict[data_index]

        # Generate data
        data = self.__data_generation(data_index)

        return data, idx
    
    def __data_generation(self, indexes):
        """Generates data containing batch_size samples.
        
        Returns:
            data: a dictionary with data.image in [b, ch, h, w]; data.gene in [b, n_genes]
        """
        image = self.image_dict[indexes]
        if self.augmentation:
            image = augmentor(normalize(image))
        else:
            image = normalize(image)
        
        # data = IGCData()
        # data.image = torch.tensor([np.transpose(image, (2, 0, 1)).astype(float)])
        # data.gene = torch.tensor(self.count_df.loc[indexes, :].values)
        
        data = dict()
        data['image'] = torch.tensor(np.transpose(image, (2, 0, 1)).astype(float))
        data['gene'] = torch.tensor(self.count_df.loc[indexes, :].values)
        
        return data


class DataLoader(object):
    """Data loader for image & genetic data
    
    TODO: adapt to torch.utils.data.DataLoader
    
    Example:
    test = DataLoader(indexes, None, None, batch_size=100, shuffle=True)
    for epoch in range(2):
        test.on_epoch_start()
        for i, j in enumerate(test):
            print(i)
            print(j)
    """
    __initialized = False
    def __init__(self, indexes, image_dict, count_df, batch_size=1, shuffle=False, drop_last=False):
        """
        Args:
            indexes: index used for both image_dict and count_df
        """
        self.indexes = indexes
        self.indexes_dict = dict()
        for i, _ in enumerate(indexes):
            self.indexes_dict[_] = i
        
        self.image_dict = image_dict
        self.count_df = count_df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.__initialized = True
        
    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data.
        
        Returns:
            idx: indexes of samples (long)
        """
        if drop_last:
            if index >= len(self) - 1:
                raise StopIteration()
        else:
            if index >= len(self):
                raise StopIteration()
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Generate torch.long indexes of the batch samples
        idx = []
        for _ in indexes:
            idx.append(self.indexes_dict[_])

        # Generate data
        data = self.__data_generation(indexes)

        return data, idx

    def on_epoch_start(self):
        """Updates indexes before each epoch."""
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples.
        
        Returns:
            data: a dictionary with data.image in [b, ch, h, w]; data.gene in [b, n_genes]
        """
        return data