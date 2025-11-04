import os
import random
import h5py
import numpy as np
import torch
import cv2
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, noise='none'):
        self.output_size = output_size
        self.noise = noise

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # noise injection
        if self.noise in ['0.1', '0.2']:
            sigma = 0.1 if self.noise == '0.1' else 0.2
            orig_min, orig_max = image.min(), image.max()
            noise = np.random.normal(0.0, sigma, size=image.shape)
            image = image + noise
            image = np.clip(image, orig_min, orig_max)
        elif self.noise == 'strong':
            # strong noise: Gaussian (15% range), salt&pepper (5% min/max), multiplicative noise (std 0.1)
            img_min = float(image.min())
            img_max = float(image.max())
            img_range = img_max - img_min if img_max > img_min else 1.0
            gaussian_noise = np.random.normal(0.0, 0.15 * img_range, size=image.shape).astype(np.float32)
            multiplicative_noise = np.random.normal(1.0, 0.1, size=image.shape).astype(np.float32)
            noisy = image * multiplicative_noise + gaussian_noise
            sp_mask = np.random.rand(*image.shape)
            noisy[sp_mask < 0.05] = img_max
            noisy[sp_mask > 0.95] = img_min
            image = np.clip(noisy.astype(np.float32), img_min, img_max)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, nclass=9, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.nclass = nclass

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            #print(image.shape)
            #image = np.reshape(image, (512, 512))
            #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            #label = np.reshape(label, (512, 512))
            
            
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
            #image = np.reshape(image, (image.shape[2], 512, 512))
            #label = np.reshape(label, (label.shape[2], 512, 512))
            #label[label==5]= 0
            #label[label==9]= 0
            #label[label==10]= 0
            #label[label==12]= 0
            #label[label==13]= 0
            #label[label==11]= 5

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
