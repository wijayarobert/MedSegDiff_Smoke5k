import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import torchvision.utils as vutils
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
image_root='D:\\Courses\\Semester 1 2023\\COMP8604\\Experiment\\MedSegDiff\\dataset\\SMOKE5K\\SMOKE5K\\train\\img\\'
gt_root='D:\\Courses\\Semester 1 2023\\COMP8604\\Experiment\\MedSegDiff\\dataset\\SMOKE5K\\SMOKE5K\\train\\gt\\'

root_dir = 'D:\\Courses\\Semester 1 2023\\COMP8604\\Experiment\\MedSegDiff\\dataset\\SMOKE5K\\SMOKE5K\\train\\'

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transforms=None, mode='Training'):
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_dir = os.path.join(root_dir, 'img')
        self.mask_dir = os.path.join(root_dir, 'gt')
        self.image_filenames = os.listdir(self.image_dir)
        self.mode = mode

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx])
        # img_path = '/content/MedSegDiff_Smoke5k/dataset/SMOKE5K/SMOKE5K/test/img/1530902341_+00660.jpg'


        # Replace file extension from .jpg to .png
        # img_path = img_path.replace('.jpg', '.png')
        mask_path = mask_path.replace('.jpg', '.png')
        
        with Image.open(img_path) as img:
            img = Image.open(img_path).convert('RGB')
            # img.show()
            # Resize image
            # print(img.shape)
            # image = transforms.Resize((256, 256))(img)
            # Convert to tensor
            # image = transforms.ToTensor()(img)
        
        with Image.open(mask_path) as mask:
            mask = mask.convert('L')
            # Resize mask
            # mask = transforms.Resize((256, 256))(mask)
            # Convert to tensor
            # mask = transforms.ToTensor()(mask)
        
        # if self.transforms is not None:
        #     image, mask = self.transforms(image, mask)

        if self.transforms:
            state = torch.get_rng_state()
            image = self.transforms(img)
            torch.set_rng_state(state)
            mask = self.transforms(mask)

        if self.mode == 'Training':
            return (image, mask)
        else:
            return (image, mask, self.image_filenames[idx]) 
        
        # return image, mask



# class SegmentationDataset(Dataset):
#     def __init__(self, image_paths, mask_paths, transform=None):
#         self.image_paths = image_paths
#         self.mask_paths = mask_paths
#         self.transform = transform
        
#     def __getitem__(self, index):
#         image = Image.open(self.image_paths[index])
#         mask = Image.open(self.mask_paths[index])
#         if self.transform:
#             image = self.transform(image)
#             mask = self.transform(mask)
#         return image, mask
        
#     def __len__(self):
#         return len(self.image_paths)

# image_paths = 'D:\\Courses\\Semester 1 2023\\COMP8604\\Experiment\\MedSegDiff\\dataset\\SMOKE5K\\SMOKE5K\\train\\img\\'
# mask_paths = 'D:\\Courses\\Semester 1 2023\\COMP8604\\Experiment\\MedSegDiff\\dataset\\SMOKE5K\\SMOKE5K\\train\\gt\\'

# transform = Compose([
#     ToTensor(),
#     Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# dataset = SegmentationDataset(image_paths, mask_paths, transform=transform)

# dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)


# class SMOKE5kDataset(torch.utils.data.Dataset):
#     def __init__(self, image_root, gt_root, batch_size, trainsize):
#         self.trainsize = trainsize
#         self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
#         self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
#                     or f.endswith('.png')]
#         #self.trans= [trans_map_root + f for f in os.listdir(trans_map_root) if f.endswith('.jpg')]
#         self.images = sorted(self.images)
#         self.gts = sorted(self.gts)
#         #self.trans=sorted(self.trans)
#         self.filter_files()
#         self.size = len(self.images)
#         self.img_transform = transforms.Compose([
#             transforms.Resize((self.trainsize, self.trainsize)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#         self.gt_transform = transforms.Compose([
#             transforms.Resize((self.trainsize, self.trainsize)),
#             transforms.ToTensor()])
#         # self.trans_transform = transforms.Compose([
#         #     transforms.Resize((self.trainsize, self.trainsize)),
#         #     transforms.ToTensor()])

#     def __getitem__(self, index):
#         image = self.rgb_loader(self.images[index])
#         gt = self.binary_loader(self.gts[index])
#         # tran = self.binary_loader(self.trans[index])
#         image = self.img_transform(image)
#         gt = self.gt_transform(gt)
#         # tran = self.trans_transform(tran)
#         return image, gt

#     def filter_files(self):
#         assert len(self.images) == len(self.gts)
#         # assert len(self.images) == len(self.trans)
#         images = []
#         gts = []
#         trans=[]
#         for img_path, gt_path in zip(self.images, self.gts):
#             img = Image.open(img_path)
#             gt = Image.open(gt_path)
#             if img.size == gt.size:
#                 images.append(img_path)
#                 gts.append(gt_path)
#                 # trans.append(tran_path)
#         self.images = images
#         self.gts = gts
#         # self.trans=trans

#     def rgb_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('RGB')

#     def binary_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             # return img.convert('1')
#             return img.convert('L')

#     def resize(self, img, gt):
#         assert img.size == gt.size
#         w, h = img.size
#         if h < self.trainsize or w < self.trainsize:
#             h = max(h, self.trainsize)
#             w = max(w, self.trainsize)
#             return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
#         else:
#             return img, gt

#     def __len__(self):
#         return self.size


# def get_loader(image_root, gt_root,batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):

#     dataset = SMOKE5kDataset(image_root, gt_root, batchsize, trainsize)
#     data_loader = data.DataLoader(dataset=dataset,
#                                   batch_size=batchsize,
#                                   shuffle=shuffle,
#                                   num_workers=num_workers,
#                                   pin_memory=pin_memory)
#     return data_loader


# class test_dataset:
#     def __init__(self, image_root, testsize):
#         self.testsize = testsize
#         self.images = [image_root + f for f in os.listdir(image_root)  if f.endswith('.jpg')
#                     or f.endswith('.png')]
#         self.images = sorted(self.images)
#         self.transform = transforms.Compose([
#             transforms.Resize((self.testsize, self.testsize)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#         self.size = len(self.images)
#         self.index = 0

#     def load_data(self):
#         image = self.rgb_loader(self.images[self.index])
#         HH = image.size[0]
#         WW = image.size[1]
#         image = self.transform(image).unsqueeze(0)
#         name = self.images[self.index].split('/')[-1]
#         if name.endswith('.jpg'):
#             name = name.split('.jpg')[0] + '.png'
#         self.index += 1
#         return image, HH, WW, name

#     def rgb_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('RGB')

#     def binary_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('L')

#     def resize(self, img, gt):
#         assert img.size == gt.size
#         w, h = img.size
#         if h < self.trainsize or w < self.trainsize:
#             h = max(h, self.trainsize)
#             w = max(w, self.trainsize)
#             return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
#         else:
#             return img, gt

#     def __len__(self):
#         return self.size

# def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):

#     dataset = SMOKE5kDataset(image_root, gt_root, trainsize)
#     data_loader = data.Data


