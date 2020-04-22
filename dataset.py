import PIL.Image as Image
import os
import glob
import torch
from torch.utils.data import DataLoader
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import numpy as np


class EM_DATA(torch.utils.data.Dataset):
    """
    Data class to load in and tranform the data given on campusnet
    """
    
    def __init__(self, train, size, _transform, data_path="EM_ISBI_Challenge"):
        """
        This assumes the same folder structure as the campusnet data i.e
        EM_ISBI_Challenge
            - test_images
            - train_images
            - train_labels
        :param train: Boolean to select if the test or train data should be loaded
        :param size: Image size to return
        :param data_path: Path to the EM_ISBI_Challenge folder
        :param _transform: A torchvision transform object containing only non random transformations!
        """

        self._size = size
        self._transform = _transform
        self._train = train
        self._root_dir = data_path
        self.data_path = os.path.join(self._root_dir, 'train_images' if self._train else 'test_images')
        self.image_paths = sorted(glob.glob(os.path.join(self.data_path, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(self._root_dir, "train_labels", "*.png"))) if self._train else None
        self._test_mask = torch.zeros(self._size)
        
    def __len__(self):
        """
        Returns the total number of samples
        :return: # of images
        """
        return len(self.image_paths)
    
    def transform(self, image, mask):
        """
        The idea of this function is that you can add transformations. This can be done in multiple ways
        but here it is important that the exact same transformation is done to the training label and image
        :param image:
        :param mask:
        :return: Transformed images and masks
        """
        # Resize
        resize = transforms.Resize(size=(self._size, self._size))
        image = resize(image)
        mask = resize(mask) if self._train else self._test_mask

        """ Example on how you could at a transformation you define yourself
        # Random horizontal flipping
        if self._train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
        """

        # Apply the transformations defined in the input transform parameter. Remember to end it with 'to_tensor'
        image = self._transform(image)
        mask = self._transform(mask) if self._train else self._test_mask
        return image, mask

    def _load_mask(self, idx):
        """
        Helper function to load in masks
        :param idx: index to return
        :return: mask with the given index
        """
        mask = Image.open(self.mask_paths[idx])
        mask = np.asarray(mask).copy()
        mask[mask>0] = 255
        return Image.fromarray(mask)

    def __getitem__(self, idx):
        """
        This is the entire idea of this function and makes it a python generator function which is what Pytorch
         assume for the dataloader
        :param idx: Image index to return
        :return: Return a X, Y pair of image and mask
        """
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path).convert("RGB")
        mask = self._load_mask(idx) if self._train else self._test_mask

        X, y = self.transform(image, mask)

        return X, y