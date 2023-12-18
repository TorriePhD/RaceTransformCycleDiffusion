import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
from pathlib import Path

# from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir,excludeList=None):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    user = fname.split("_")[:-1]
                    user = "_".join(user)
                    if excludeList is not None and user in excludeList:
                        continue
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)


class RaceTransformDataset(data.Dataset):
   def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[160, 160], loader=pil_loader,races=["Caucasian", "African"]):#,'Caucasian','Indian']):
        imgs = []
        testSet = np.load(Path(data_root).parent / "test.npy", allow_pickle=True).item()
        valSet = np.load(Path(data_root).parent / "val.npy", allow_pickle=True).item()
        trainSet = np.load(Path(data_root).parent / "train.npy", allow_pickle=True).item()
        exxcludeList = []
        oldDataRoot = data_root
        for i in races:
            testSetRace = testSet[i]
            valSetRace = valSet[i]
            trainSetRace = trainSet[i]
            excludeSet = valSetRace
            if oldDataRoot == "/home/st392/fsl_groups/grp_nlp/compute/RFW/CroppedImages":
                excludeSet = excludeSet
            elif oldDataRoot == "/home/st392/fsl_groups/grp_nlp/compute/RFW/CroppedImagesTest":
                excludeSet = trainSet +testSetRace
                data_root = "/home/st392/fsl_groups/grp_nlp/compute/RFW/CroppedImages"

            imgs += make_dataset(os.path.join(data_root,i),excludeSet)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.images = [self.tfs(self.loader(i)) for i in self.imgs]
        self.image_size = image_size
        self.races = races

   def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.images[index]

        ret['gt_image'] = img
        ret["cond_image"] = img.clone()
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        race = path.rsplit("/")[-2].rsplit("\\")[-1]
        ret['direction'] = "A" if race ==self.races[0] else "B"
        return ret

   def __len__(self):
        return len(self.imgs)

from torch.utils.data import Sampler
import random

class CustomBatchSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        # Organize indices by direction
        self.A_indices = [i for i, _ in enumerate(data_source) if data_source[i]['direction'] == 'A']
        self.B_indices = [i for i, _ in enumerate(data_source) if data_source[i]['direction'] == 'B']

    def _generate_batches(self, indices):
        # Generate batches for a given set of indices
        return [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]

    def __iter__(self):
        # Shuffle the indices
        random.shuffle(self.A_indices)
        random.shuffle(self.B_indices)

        # Create batches for each direction
        A_batches = self._generate_batches(self.A_indices)
        B_batches = self._generate_batches(self.B_indices)

        # Combine and shuffle the batches
        combined_batches = A_batches + B_batches
        random.shuffle(combined_batches)

        # Yield interleaved batches
        for batch in combined_batches:
            yield batch

    def __len__(self):
        # Calculate the number of batches
        return (len(self.A_indices) + len(self.B_indices)) // self.batch_size




if __name__=="__main__":
    raceTransform = RaceTransformDataset("/home/st392/fsl_groups/grp_nlp/compute/RFW/CroppedImagesTest")
    print(len(raceTransform))