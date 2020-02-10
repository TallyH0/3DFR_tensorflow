import tensorflow as tf
import numpy as np 
import cv2
import random
import os

class Data_loader_sequence():
    def __init__(self, dir_data, h, w, num_seq=50, gray = True):
        self.n_seq = num_seq
        self.img_h = h
        self.img_w = w

        self.frame_list = []
        # self.index = 0
        self.epoch = 0
        self.cnt = 0

        with open(os.path.join(dir_data, 'temporalROI.txt')) as f:
            dir_gt = os.path.join(dir_data, 'groundtruth')
            dir_input = os.path.join(dir_data, 'input')
            line = f.read()
            start, end = line.split(' ')
            if gray:
                frame_roi = np.expand_dims(cv2.imread(os.path.join(dir_data, 'ROI.bmp'), 0), 2) / 255
            else:
                frame_roi = cv2.imread(os.path.join(dir_data, 'ROI.bmp')) / 255

            for i in range(int(start) - 49, int(end) + 1):
                if gray:
                    fname_input = np.expand_dims(cv2.imread(os.path.join(dir_input, 'in%06d.jpg' % i), 0), 2)
                else:
                    fname_input = cv2.imread(os.path.join(dir_input, 'in%06d.jpg' % i))

                fname_input = fname_input * frame_roi
                fname_input = cv2.resize(fname_input, (self.img_w, self.img_h))
                if gray:
                    fname_input = np.expand_dims(fname_input, 2)
                fname_gt = cv2.imread(os.path.join(dir_gt, 'gt%06d.png' % i), 0)
                fname_gt = np.expand_dims(fname_gt, 2)
                fname_gt = cv2.resize(fname_gt, (self.img_w, self.img_h))
                ret, fname_gt = cv2.threshold(fname_gt, 200, 255, cv2.THRESH_BINARY)
                fname_gt = np.expand_dims(fname_gt, 2)

                self.frame_list.append([fname_input, fname_gt])

        self.size = len(self.frame_list) - self.n_seq
    
    def batch(self, n, index=None):
        b_image, b_label = [], []
        b_current, b_median = [], []

        for i in range(n):
            seq_image, seq_label = [], []
            if index is None:
                index = random.randint(0, len(self.frame_list) - self.n_seq)
            for j in range(index, index + self.n_seq, 1):
                seq_image.append(self.frame_list[j][0])
                seq_label.append(self.frame_list[j][1])
            
            seq_median = np.median(seq_image, axis=0)

            b_image.append(seq_image)
            b_current.append(seq_image[-1])
            b_label.append(seq_label[-1])
            b_median.append(seq_median)

            self.cnt += 1
            if self.cnt > self.size:
                self.epoch += 1
                self.cnt = 0

        return b_image, b_current, b_label, b_median
            


if __name__ == '__main__':
    # test = Data_loader_Unsupervised('image', 100, 100, True)
    # test_iter = test.get_data_iterator(3, 100)
    # print(test_iter)
    dir_data = 'F:/Data/CDNet2014/dataset/dynamicBackground/overpass'
    test = Data_loader_sequence(dir_data, 240, 320, in_memory = True, gray=False)
    # bimg, bcurrent, blabel, bmedian = test.batch(1, len(test.frame_list) - test.n_seq - 10)
    bimg, bcurrent, blabel, bmedian = test.batch(1, 120)
    print(bcurrent[0].shape)
    print(bmedian[0].shape)
    print(blabel[0].shape)

    cv2.imshow('label', blabel[0])
    cv2.imshow('median', np.array(bmedian[0], np.uint8))
    cv2.imshow('img', np.array(bcurrent[0], np.uint8))
    cv2.waitKey(0)