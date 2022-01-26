import os, glob
import os.path as osp
import paddle
# import torchvision.transforms as tf
import cv2
import random
import numpy as np

def augment(img_list, hflip=True, rot=True):
    """horizontal flip OR rotate (0, 90, 180, 270 degrees)"""
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]

class TrainData(paddle.io.Dataset):

    def __init__(self, data_path, patch_size=128, file_type='*', hr_dir='hr', lr_dir='lr', scale = 2, start=0, end=999999):
        super(TrainData, self).__init__()
        self.hr_list = sorted(glob.glob(osp.join(data_path, hr_dir, '*.' + file_type)))[start:end]
        self.lr_list = sorted(glob.glob(osp.join(data_path, lr_dir, '*.' + file_type)))[start:end]
        self.patchsize = patch_size
        self.scale = scale


    def __getitem__(self, index):
        if os.path.basename(self.hr_list[index]).split('.')[0] != os.path.basename(self.lr_list[index]).split('.')[0]:
            print('error!', os.path.basename(self.hr_list[index]), os.path.basename(self.lr_list[index]))
        hr_img = cv2.imread(self.hr_list[index], cv2.IMREAD_UNCHANGED) #Image.open(self.hr_list[index])

        lr_img = cv2.imread(self.lr_list[index], cv2.IMREAD_UNCHANGED) #Image.open(self.lr_list[index])
        if hr_img is None or lr_img is None:
            print(self.hr_img[index], self.lr_img[index])
        H, W , _ = lr_img.shape
        lr_size = self.patchsize
        hr_size = self.patchsize * self.scale

        rnd_h = random.randint(0, max(0, H - lr_size))
        rnd_w = random.randint(0, max(0, W - lr_size))
        img_lr = lr_img[rnd_h:rnd_h + lr_size, rnd_w:rnd_w + lr_size, :] / 255.0
        # cv2.imwrite('lr.jpg', lr_img[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :])
        rnd_h_hr, rnd_w_hr = int(rnd_h * self.scale), int(rnd_w * self.scale)
        img_hr = hr_img[rnd_h_hr:rnd_h_hr + hr_size, rnd_w_hr:rnd_w_hr + hr_size, :] / 255.0
        # cv2.imwrite('hr.jpg', hr_img[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :])

        # augmentation - flip, rotate
        img_lr, img_hr = augment([img_lr, img_hr])
        # cv2.imwrite('{}lr.jpg'.format(index), (img_LQ + 0.5)*255)
        # cv2.imwrite('{}hr.jpg'.format(index), (img_GT + 0.5)*255)
        img_hr = img_hr[:, :, [2, 1, 0]]
        img_lr = img_lr[:, :, [2, 1, 0]]

        img_hr, img_lr = img_hr.astype(np.float32), img_lr.astype(np.float32)
        img_hr = paddle.to_tensor(np.ascontiguousarray(np.transpose(img_hr, (2, 0, 1))))
        img_lr = paddle.to_tensor(np.ascontiguousarray(np.transpose(img_lr, (2, 0, 1))))

        return img_lr, img_hr, self.lr_list[index], self.hr_list[index]

    def __len__(self):
        return len(self.hr_list)

class ValData(paddle.io.Dataset):

    def __init__(self, data_path, file_type='png', hr_dir='hr', lr_dir='lr', start=0, end=999999):
        super(ValData, self).__init__()
        self.lr_list =  sorted(glob.glob(osp.join(data_path, lr_dir, '*.' + file_type)))[start:end]
        self.hr_list = sorted(glob.glob(osp.join(data_path, hr_dir, '*.' + file_type)))[start:end]
        # print(self.lr_list)

    def __getitem__(self, index):
        lr_img = cv2.imread(self.lr_list[index], cv2.IMREAD_UNCHANGED)  # Image.open(self.lr_list[index])
        img_lr = lr_img / 255.0
        img_lr = img_lr[:, :, [2, 1, 0]]
        img_lr = img_lr.astype(np.float32)
        img_lr = paddle.to_tensor(np.ascontiguousarray(np.transpose(img_lr, (2, 0, 1))))

        hr_img = cv2.imread(self.hr_list[index], cv2.IMREAD_UNCHANGED)
        img_hr = hr_img / 255.0
        img_hr = img_hr[:, :, [2, 1, 0]]

        img_hr = img_hr.astype(np.float32)
        img_hr = paddle.to_tensor(np.ascontiguousarray(np.transpose(img_hr, (2, 0, 1))))

        return img_lr, img_hr, self.lr_list[index], self.hr_list[index]

    def __len__(self):
        return len(self.lr_list)

