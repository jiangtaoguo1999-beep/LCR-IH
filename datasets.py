import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from natsort import natsorted
from PIL import Image
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import config
args = config.Args()

# albumentations 变换
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

DIV2K_path = '/seu_nvme/home/230240036/projects/DIV2K_train_HR_20250605152066/DIV2K_train_HR'
DIV2K_path_valid = '/seu_nvme/home/230240036/projects/DIV2K_valid_HR_20250605152066/DIV2K_valid_HR'
batchsize = 12


# ========== 新增：分离载体和秘密图像的数据集类 ==========
class DIV2K_Split_Dataset(Dataset):
    """
    从验证集中分离前50张作为载体图像，后50张作为秘密图像
    """
    def __init__(self, transforms_=None, image_type='cover'):
        """
        Args:
            transforms_: 数据增强变换
            image_type: 'cover' (载体图像，前50张) 或 'secret' (秘密图像，后50张)
        """
        self.transform = transforms_
        self.image_type = image_type

        all_files = natsorted(sorted(glob.glob(DIV2K_path_valid + "/*.png")))

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


# 原有的数据集类保持不变
class DIV2K_Dataset(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        self.mode = mode
        if mode == 'train':
            self.files = natsorted(sorted(glob.glob(DIV2K_path + "/*.png")))
        else:
            self.files = natsorted(sorted(glob.glob(DIV2K_path_valid + "/*.png")))

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
        self.files = natsorted(
            sorted(glob.glob("/seu_nvme/ogai/datasets/coco2017/test2017" + "/*." + "jpg")))

    def __getitem__(self, index):
        img = cv2.imread(self.files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        trans_img = self.transform(image=img)
        item = trans_img['image']
        item = item / 255.0
        return item

    def __len__(self):
        return len(self.files)


# ========== 训练数据加载器（保持不变）==========
DIV2K_train_cover_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A, mode="train"),
    batch_size=args.single_batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    drop_last=True
)

DIV2K_train_secret_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A, mode="train"),
    batch_size=args.single_batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    drop_last=True
)

DIV2K_val_cover_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A_valid, mode="val"),
    batch_size=args.single_batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=2,
    drop_last=True
)

DIV2K_val_secret_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A_valid, mode="val"),
    batch_size=args.single_batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=2,
    drop_last=True
)

DIV2K_multi_train_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A, mode="train"),
    batch_size=args.multi_batch_iteration,
    shuffle=True,
    pin_memory=True,
    num_workers=16,
    drop_last=True
)

DIV2K_multi_val_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A_valid, mode="val"),
    batch_size=args.multi_batch_iteration,
    shuffle=True,
    pin_memory=True,
    num_workers=16,
    drop_last=True
)


# ========== 新的测试数据加载器：使用分离的数据集 ==========
DIV2K_test_cover_loader = DataLoader(
    DIV2K_Split_Dataset(transforms_=transform_A_test, image_type='cover'),
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    num_workers=1,
    drop_last=False
)

DIV2K_test_secret_loader = DataLoader(
    DIV2K_Split_Dataset(transforms_=transform_A_test, image_type='secret'),
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    num_workers=1,
    drop_last=False
)

DIV2K_multi_test_loader = DataLoader(
    DIV2K_Split_Dataset(transforms_=transform_A_test, image_type='cover'),
    batch_size=args.test_multi_batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=1,
    drop_last=False
)

COCO_test_multi_loader = DataLoader(
    COCO_Test_Dataset(transforms_=transform_A_test_256),
    batch_size=args.test_multi_batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)

COCO_test_cover_loader = DataLoader(
    COCO_Test_Dataset(transforms_=transform_A_test_256),
    batch_size=1,
    shuffle=True,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)

COCO_test_secret_loader = DataLoader(
    COCO_Test_Dataset(transforms_=transform_A_test_256),
    batch_size=1,
    shuffle=True,
    pin_memory=False,
    num_workers=1,
    drop_last=True
)


if __name__ == "__main__":
    print("=" * 50)
    print("测试数据加载器信息:")
    print(f"载体图像数量: {len(DIV2K_test_cover_loader.dataset)}")
    print(f"秘密图像数量: {len(DIV2K_test_secret_loader.dataset)}")
    print("=" * 50)

    for i, (cover, secret) in enumerate(zip(DIV2K_test_cover_loader, DIV2K_test_secret_loader)):
        print(f"Pair {i+1}: Cover shape: {cover.shape}, Secret shape: {secret.shape}")
        if i >= 2:
            break
