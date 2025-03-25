import os
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from PIL import Image
import albumentations as A
import random
import torch
import numpy as np
import warnings

# 忽略特定类型的警告
warnings.filterwarnings("ignore")
def strong_aug(p=0.5):

    return A.Compose([
        A.HorizontalFlip(),  # 水平翻转
        A.OneOf([
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=1, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=30, p=0.2),
        A.OneOf([
            # CLAHE(clip_limit=2),
            # IAASharpen(),
            # IAAEmboss(),
            A.RandomBrightnessContrast(0.1, 0.1),  # 放射变换
        ], p=0.3),
        # HueSaturationValue(p=0.3),
    ], p=p)
class TripletDataset(Dataset):
    def __init__(self, root=r"F:\dlproj\face anti-spoofing\archive\LCC_FASD\LCC_FASD_training",
                 sub_dirs=["real", 'spoof']):
        self.root = root
        self.sub_dirs = sub_dirs

        # 检查目录是否存在
        pos_dir = os.path.join(self.root, self.sub_dirs[0])
        neg_dir = os.path.join(self.root, self.sub_dirs[1])
        if not os.path.isdir(pos_dir):
            raise FileNotFoundError(f"Positive directory not found: {pos_dir}")
        if not os.path.isdir(neg_dir):
            raise FileNotFoundError(f"Negative directory not found: {neg_dir}")

        # 生成正样本和负样本文件列表，并检查文件是否有效
        self.pos_filelist = [os.path.join(pos_dir, item) for item in os.listdir(pos_dir) if os.path.isfile(os.path.join(pos_dir, item))]
        self.neg_filelist = [os.path.join(neg_dir, item) for item in os.listdir(neg_dir) if os.path.isfile(os.path.join(neg_dir, item))]

        # 定义图像变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 定义数据增强
        self.aug = strong_aug(0.5)
    def __getitem__(self, idx):
        imgs = []
        labels = None  # 规定 0 -> 正正负    1 -> 负负正
        if idx % 2 == 0:  # 正正负的情况
            labels = 0
            for k in range(3):
                if k == 0:
                    t = random.randint(0, len(self.pos_filelist) - 1)
                    l = self.pos_filelist[t]  # 取一个正样本
                elif k == 1:
                    t = random.randint(0, len(self.pos_filelist) - 1)
                    l = self.pos_filelist[t]
                else:
                    t = random.randint(0, len(self.neg_filelist) - 1)
                    l = self.neg_filelist[t]  # 从所有类型的负样本中随机选取一个
                img_path = l
                img = Image.open(img_path).convert("RGB")

                # img_w, img_h = img.size

                # ymin, ymax, xmin, xmax = 92, 188, 42, 138  # crop 整张脸

                # img = img.crop([xmin, ymin, xmax, ymax])

                img = self.aug(image=np.array(img))["image"]  # self.transform(img)

                img = self.transform(Image.fromarray(img))

                imgs.append(img)
        else:  # 负负正的情况

            labels = 1

            for k in range(3):

                if k == 0:
                    t = random.randint(0, len(self.neg_filelist) - 1)
                    l = self.neg_filelist[t]
                elif k == 1:
                    t = random.randint(0, len(self.neg_filelist) - 1)
                    l = self.neg_filelist[t]

                else:
                    t = random.randint(0, len(self.pos_filelist) - 1)
                    l = self.pos_filelist[t]
                img_path = l
                img = Image.open(img_path).convert("RGB")

                img_w, img_h = img.size

                #    ymin, ymax, xmin, xmax = 92, 188, 42, 138  # crop 整张脸

                #   img = img.crop([xmin, ymin, xmax, ymax])

                img = self.aug(image=np.array(img))["image"]  # self.transform(img)

                img = self.transform(Image.fromarray(img))

                imgs.append(img)

        return imgs[0], imgs[1], imgs[2], torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.neg_filelist)
if __name__ == "__main__":
    train_data = DataLoader(TripletDataset(),batch_size=32,shuffle=True,num_workers=2)
    print(len(TripletDataset()))
    print(len(train_data))