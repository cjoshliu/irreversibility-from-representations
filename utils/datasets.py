import abc
import glob
import logging
import os
from skimage.io import imread
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


DIR = os.path.abspath(os.path.dirname(__file__))
COLOR_BLACK = 0
COLOR_WHITE = 1
DATASETS_DICT = {"cgle64":"CGLE64"}
DATASETS = list(DATASETS_DICT.keys())


def get_dataset(dataset):
    """Return the correct dataset."""
    dataset = dataset.lower()
    try:
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError("Unknown dataset: {}".format(dataset))


def get_img_size(dataset):
    """Return the correct image size."""
    return get_dataset(dataset).img_size


def get_background(dataset):
    """Return the image background color."""
    return get_dataset(dataset).background_color


def get_dataloaders(dataset, root=None, shuffle=True, pin_memory=True,
                    batch_size=128, logger=logging.getLogger(__name__), **kwargs):
    """
    Return PyTorch DataLoader for a given dataset.

    Parameters
    ----------
    dataset : {"cgle64", some DIY dataset}
        Name of the dataset to load

    root : str
        Root directory of dataset. If `None` uses the default one.

    kwargs :
        Additional arguments to `DataLoader`. Default values are modified.
    """
    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available
    Dataset = get_dataset(dataset)
    dataset = Dataset(logger=logger) if root is None else Dataset(root=root, logger=logger)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      pin_memory=pin_memory,
                      **kwargs)


class DisentangledDataset(Dataset, abc.ABC):
    """
    Base class for disentangled VAE datasets.

    Parameters
    ----------
    root : str
        Root directory of dataset.

    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.
    """

    def __init__(self, root, transforms_list=[], logger=logging.getLogger(__name__)):
        self.root = root
        self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger

    def __len__(self):
        return len(self.imgs)

    @abc.abstractmethod
    def __getitem__(self, idx):
        """
        Return the image indexed `idx`.

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.) of shape `img_size`.
        """
        pass


class CGLE64(DisentangledDataset):
    """
    2 x 64 x 64 simulated CGL or experimental Rho data
    """
    files = {"train": "img_align_cgle64"}
    img_size = (2, 64, 64) # first channel is cosine, second is sine
    background_color = COLOR_BLACK

    def __init__(self, root=os.path.join(DIR, '../data/cgle64'), **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)
        self.imgs = sorted(glob.glob(self.train_data + '/*'))

    def __getitem__(self, idx):
        """
        Return the image indexed `idx`

        Return
        ------
        img : torch.Tensor
            Tensor in [0.,1.) of shape `img_size`.

        placeholder : 0
            Placeholder value, as their are no targets.
        """
        img_path = self.imgs[idx]
        # img values are between 0 and 255 because we use TIFF
        img = imread(img_path)

        # rescale each pixel to [0.,1.) and reshape img to (C x H x W)
        img = self.transforms(img)
        if img.shape[0] == 64:
            img = torch.transpose(img, 0, 1)
            img = torch.transpose(img, 1, 2)

        # no label so return 0 (required by dataloaders)
        return img, 0