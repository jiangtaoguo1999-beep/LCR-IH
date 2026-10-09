import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from natsort import natsorted
from PIL import Image
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

# Public-safe default dataset paths; replace with local paths as needed.
DIV2K_train_dir = "./data/DIV2K_train"
DIV2K_valid_dir = "./data/DIV2K_valid"
COCO_dir = "./data/COCO"


# albumentations transforms
transform_A = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.RandomRotate90(),
    A.HorizontalFlip(),
    A.augmentations.transforms.ChannelShuffle(0.3),
    ToTensorV2()
])

transform_A_valid = A.Compose([
    A.CenterCrop(width=256, height=256),
    ToTensorV2()
])

transform_A_test = A.Compose([
    A.CenterCrop(width=256, height=256),
    ToTensorV2()
])

transform_A_test_256 = A.Compose([
    A.PadIfNeeded(min_width=256, min_height=256),
    A.CenterCrop(width=256, height=256),
    ToTensorV2()
])


class DIV2K_Split_Dataset(Dataset):
    def __init__(self, transforms_=None, image_type='cover'):
        self.transform = transforms_
        self.image_type = image_type
        all_files = natsorted(sorted(glob.glob(f"{DIV2K_valid_dir}/*.png")))
        if image_type == 'cover':
            self.files = all_files[:50]
        elif image_type == 'secret':
            self.files = all_files[50:100]
        else:
            raise ValueError("image_type must be 'cover' or 'secret'")
        print(f"Loaded {len(self.files)} images for {image_type}")

    def __getitem__(self, index):
        img = cv2.imread(self.files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        trans_img = self.transform(image=img)
        item = trans_img['image']
        item = item / 255.0
        return item

    def __len__(self):
        return len(self.files)


class DIV2K_Dataset(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        self.mode = mode
        if mode == 'train':
            self.files = natsorted(sorted(glob.glob(f"{DIV2K_train_dir}/*.png")))
        else:
            self.files = natsorted(sorted(glob.glob(f"{DIV2K_valid_dir}/*.png")))

    def __getitem__(self, index):
        img = cv2.imread(self.files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        trans_img = self.transform(image=img)
        item = trans_img['image']
        item = item / 255.0
        return item

    def __len__(self):
        return len(self.files)


class COCO_Test_Dataset(Dataset):
    def __init__(self, transforms_=None):
        self.transform = transforms_
        self.files = natsorted(sorted(glob.glob(f"{COCO_dir}/*.jpg")))

    def __getitem__(self, index):
        img = cv2.imread(self.files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        trans_img = self.transform(image=img)
        item = trans_img['image']
        item = item / 255.0
        return item

    def __len__(self):
        return len(self.files)
