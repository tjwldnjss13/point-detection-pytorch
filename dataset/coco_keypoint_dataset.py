import os
import cv2 as cv
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image
from pycocotools.coco import COCO

from dataset.augment import rotate2d, horizontal_flip


class COCOKeypointDataset(data.Dataset):
    def __init__(self, root, image_size, for_train=True, year='2017', is_categorical=False, augmentation=None):
        super().__init__()
        self.root = root
        if for_train:
            self.images_dir = os.path.join(self.root, 'images', 'train' + year)
            self.coco = COCO(os.path.join(self.root, 'annotations', 'person_keypoints_train' + year + '.json'))
        else:
            self.images_dir = os.path.join(self.root, 'images', 'val' + year)
            self.coco = COCO(os.path.join(self.root, 'annotations', 'person_keypoints_val' + year + '.json'))
        # self.images_dir = images_dir
        # self.coco = COCO(annotation_path)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.imgs_pth, self.anns = self.load_person_data()
        del self.images_dir, self.coco, self.ids

        self.is_categorical = is_categorical
        self.augmentation = augmentation

        self.num_classes = 2  # Background include
        if isinstance(image_size, tuple):
            self.image_size = image_size
        elif isinstance(image_size, int):
            self.image_size = (image_size, image_size)

        self.heatmap_size = 3

    def load_person_data(self):
        imgs = []
        anns_custom = []

        for ann in self.coco.anns.values():
            ann_custom = {}

            keypoints = torch.Tensor(ann['keypoints']).view(-1, 3)[1:7, :2]
            if 0 not in keypoints[..., -1]:
                img_id = str(ann['image_id'])
                img_pth = os.path.join(self.images_dir, '0' * (12 - len(img_id)) + img_id + '.jpg')

                bbox = torch.Tensor(ann['bbox']).type(torch.int)
                y1 = bbox[1]
                x1 = bbox[0]
                y2 = y1 + bbox[3]
                x2 = x1 + bbox[2]

                if torch.sum(keypoints[..., 0] < x1) > 0:
                    x1 = torch.min(keypoints[..., 0]).type(torch.int).item()
                if torch.sum(keypoints[..., 0] > x2) > 0:
                    x2 = torch.ceil(torch.max(keypoints[..., 0])).type(torch.int).item()
                if torch.sum(keypoints[..., 1] < y1) > 0:
                    y1 = torch.min(keypoints[..., 1]).type(torch.int).item()
                if torch.sum(keypoints[..., 1] > y2) > 0:
                    y2 = torch.ceil(torch.max(keypoints[..., 1])).type(torch.int).item()

                keypoints[..., 0] -= x1
                keypoints[..., 1] -= y1

                bbox = torch.Tensor([x1, y1, x2, y2]).type(torch.int)

                imgs.append(img_pth)
                ann_custom['keypoint'] = keypoints
                ann_custom['bbox'] = bbox
                anns_custom.append(ann_custom)

                # if 0 in ann['keypoints']:
                #     img_np = np.array(Image.open(img_pth))
                #     img_np = img_np[y1:y2+1, x1:x2+1]
                #     # keypoints = torch.Tensor(ann['keypoints']).view(-1, 3)[1:7]
                #     for i, keypoint in enumerate(keypoints):
                #         pnt = keypoint[:2]
                #         label = keypoint[-1]
                #         if label != 0:
                #             img_np = cv.circle(img_np.copy(), (pnt[0], pnt[1]), 2, (0, 255, 0), -1)
                #             img_np = cv.putText(img_np.copy(), str(i), (pnt[0], pnt[1]), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2)
                #
                #     plt.imshow(img_np)
                #     plt.show()

        return imgs, anns_custom

    def __getitem__(self, idx):
        # img = transforms.ToTensor()(Image.open(self.imgs_pth[idx]))
        img = Image.open(self.imgs_pth[idx]).convert('L')
        ann = self.anns[idx]
        pnt = ann['keypoint'].clone()
        x1, y1, x2, y2 = ann['bbox']

        img = transforms.ToTensor()(img)
        img = img[..., y1:y2 + 1, x1:x2 + 1]
        img = transforms.Resize(self.image_size)(img)
        # img = transforms.Compose([transforms.Resize(self.image_size), transforms.ToTensor()])(img)

        pnt[..., 0] *= self.image_size[0] / (x2 - x1)
        pnt[..., 1] *= self.image_size[1] / (y2 - y1)

        if self.augmentation is not None:
            for aug in self.augmentation:
                img, pnt = aug(img, point=pnt)

        ###### For normalization 0-1 ######
        # pnt[..., 0] /= self.image_size[0]
        # pnt[..., 1] /= self.image_size[1]
        ###################################

        ###### For normalization 0-[heatmap size] ######
        # pnt[..., 0] *= self.heatmap_size / self.image_size[0]
        # pnt[..., 1] *= self.heatmap_size / self.image_size[1]
        ###################################

        # img_np = img.permute(1, 2, 0).numpy()
        # for p in pnt:
        #     img_np = cv.circle(img_np.copy(), (p[0], p[1]), 2, (0, 255, 0), -1)
        # plt.imshow(img_np)
        # plt.show()

        pnt = pnt.reshape(-1)

        return img, pnt

    def __len__(self):
        return len(self.imgs_pth)

    @staticmethod
    def to_categorical(label, num_classes):
        label_list = []
        if isinstance(label, list):
            for l in label:
                label_base = [0 for _ in range(num_classes)]
                label_base[l] = 1
                label_list.append(label_base)
        else:
            label_base = [0 for _ in range(num_classes)]
            label_base[label] = 1
            label_list.append(label_base)

        return label_list

    @staticmethod
    def to_categorical_multi_label(label, num_classes):
        label_result = [0 for _ in range(num_classes)]
        if isinstance(label, list):
            label_result
            for l in label:
                label_result[l] = 1
        else:
            label_result[label] = 1

        return label_result


def custom_collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    # print('data: ', data)
    # print('target: ', target)
    return [data, target]


if __name__ == '__main__':
    import numpy as np
    import cv2 as cv
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    from utils.pytorch_util import *
    from dataset.augment import *
    root = 'D://DeepLearningData/COCO/'
    augmentation = [shift_augmentation]
    dset = COCOKeypointDataset(root=root, image_size=224, for_train=False, year='2017', augmentation=augmentation)

    img, pnt = dset[111]
    img_np = img.permute(1, 2, 0).numpy()
    for p in pnt:
        img_np = cv.circle(img_np.copy(), (p[1], p[0]), 1, (0, 255, 0), -1)

    # plt.imshow(img_np)
    # plt.show()

    # for i in range(len(dset)):
    #     img, ann = dset[i]
    #
    #     exit()





























