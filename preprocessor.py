import random
from typing import Tuple

import cv2
import numpy as np
# import torchvision
# import torch



class Preprocessor:
    def __init__(self, image: np.ndarray, transform=None, augmentation=False, vocab=""):
        self.image = image
        self.transform = transform
        # self.image_size = (256, 32) 
        self.image_size = (512, 64)
        # self.image_size = (1408, 96) ## an alternative for the image size(we'll see based on training results)
        self.augment = augmentation
        self.vocab = vocab
    
    def __call__(self, img, label:str, max_len : int = 32):
        # img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img, label = self.preprocess_img(img, label)
        label = self._truncate_label(label, max_text_len=max_len)
        label = self.label_indexer(self.vocab, label)
        #label = self.label_padding(0, 32, label)
        label = self.label_padding(len(self.vocab), max_len, label)

        if self.augment:
            # kernel = np.ones((5,5),np.float32)/25
            # dst = cv.filter2D(img,-1,kernel)
            # kernel = np.ones((2,2),np.float32)/20
            # img = cv2.Laplacian(img, cv2.CV_16S, ksize=3)

            # here we should apply randomsharpening, (randomnoise), randomblur, randombrightness
            if np.random.rand() < 0.25:
                random_odd = np.random.randint(1,3) * 2 + 1
                img = cv2.GaussianBlur(img, (random_odd, random_odd), 0)
            if np.random.rand() < 0.25:
                brightness = np.random.randint(0,50)
                img = cv2.add(img, brightness)
            # if np.random.rand() < 0.25:
            #     random_odd = np.random.randint(1,3) * 2 + 1 # a random odd number between 3 and 5
            #     kernel = np.ones((random_odd,random_odd),np.uint8)
            #     img = cv2.erode(img, kernel, iterations = 1)
            # if np.random.rand() < 0.25:
            #     sharpen_factor = 1 + np.random.rand()
            #     kernel = np.array([[0, -1, 0], 
            #                [-1, sharpen_factor,-1], 
            #                [0, -1, 0]], dtype='float32')
            #     img = cv2.filter2D(img, -1, kernel)
            if np.random.rand() < 0.25:            
                img = cv2.convertScaleAbs(img, alpha=2.2, beta=np.random.randint(0,35))
        return img, label

    @staticmethod
    def _truncate_label(text: str, max_text_len: int) -> str:
        """
        Function ctc_loss can't compute loss if it cannot find a mapping between text label and input
        labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        If a too-long label is provided, ctc_loss returns an infinite gradient.
        """
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > max_text_len:
                return text[:i]
        return text

    def preprocess_img(self, img: np.ndarray, text: str) -> Tuple[np.ndarray, str]:
        """
        Resize image and truncate text label if it is too long
        """
        # # Resize image
        # img = cv2.resize(img, self.image_size)

        # # Truncate text label
        # text = self._truncate_label(text, max_text_len=32)

        # wt, ht = self.image_size
        # h, w = img.shape
        # f = min(wt / w, ht / h)
        # tx = (wt - w * f) / 2
        # ty = (ht - h * f) / 2

        # # map image into target image
        # M = np.float32([[f, 0, tx], [0, f, ty]])
        # target = np.ones([ht, wt]) * 255
        # img = cv2.warpAffine(img, M, dsize=(wt, ht), dst=target, borderMode=cv2.BORDER_TRANSPARENT)
         
        target_width, target_height = self.image_size
        height, width = img.shape[:2]
        ratio = min(target_width / width, target_height / height)
        new_w, new_h = int(width * ratio), int(height * ratio)

        resized_image = cv2.resize(img, (new_w, new_h))
        delta_w = target_width - new_w
        delta_h =  target_height - new_h
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        padding_color = 0 # (0,0,0) if we are using rgb images

        img = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)


        # img = torch.from_numpy(img)
        # print(f'before resizing img.size(): {img.size()}')
        # img = torchvision.transforms.functional.resize(img, (32,356))
        # print('torchvision transform is working')
        #  img = img.numpy()
        
        return img, text

    @staticmethod
    def label_indexer(vocab: str, label: np.ndarray):
        """Convert label to index by vocab
        """
        # def __call__(self, data: np.ndarray, label: np.ndarray):
        return np.array([vocab.index(l) for l in label if l in vocab])

    @staticmethod
    def label_padding(padding_value: int, max_len: int, label: np.ndarray):
        label = label[:max_len]
        # put 0 padding values on the left, max_len - len(labels) padding values on the right
        return np.pad(label, (0,max_len - len(label)), mode='constant', constant_values=padding_value)
    #Preprocessing of single image
    def single_image_preprocessing(self, img):
        target_width, target_height = self.image_size
        height, width = img.shape[:2]
        ratio = min(target_width / width, target_height / height)
        new_w, new_h = int(width * ratio), int(height * ratio)
        resized_image = cv2.resize(img, (new_w, new_h))
        delta_w = target_width - new_w
        delta_h =  target_height - new_h
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        padding_color = 0
        img = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
        return img


