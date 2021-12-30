import os
from PIL import Image
import cv2
import torch.utils.data as data
import torchvision.transforms as transforms


class GEDDDataset(data.Dataset):
    """
    dataloader for GEDD segmentation tasks
    """
    def __init__(self, image_root, gt_root, ed_root, edp_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.eds = [ed_root + f for f in os.listdir(ed_root) if f.endswith('.jpg')]
        self.edps = [edp_root + f for f in os.listdir(edp_root) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.eds = sorted(self.eds)
        self.edps = sorted(self.edps)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        image = Image.fromarray(image)
        gt = self.binary_loader(self.gts[index])
        ed = self.binary_loader(self.eds[index])
        edp = self.binary_loader(self.edps[index])
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        ed = self.gt_transform(ed)
        edp = self.gt_transform(edp)
        return image, gt, ed, edp

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        eds = []
        edps = []
        for img_path, gt_path, ed_path, edp_path in zip(self.images, self.gts, self.eds, self.edps):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            ed = Image.open(ed_path)
            edp = Image.open(edp_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                eds.append(ed_path)
                edps.append(edp_path)
        self.images = images
        self.gts = gts
        self.eds = eds
        self.edps = edps


    def rgb_loader(self, path):
        #with open(path, 'rb') as f:
            #img = Image.open(f)
            #return img.convert('RGB')
        img = cv2.imread(path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, ed_root, edp_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True):

    dataset = GEDDDataset(image_root, gt_root, ed_root, edp_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = Image.fromarray(image)
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        #with open(path, 'rb') as f:
            #f = f.decode('ascii')
        img = cv2.imread(path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
