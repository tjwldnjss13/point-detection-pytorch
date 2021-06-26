import numpy as np
import cv2 as cv
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

from PIL import Image

from models.face_shoulder_net import *


def inference(model, image_path, device):
    img = Image.open(image_path).convert('L')
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    x = T.Compose([T.Resize((224, 224)), T.ToTensor()])(img).unsqueeze(0).to(device)

    out = model(x)
    pnts = out.view(-1, 2)
    pnts[..., 0] *= w / 224
    pnts[..., 1] *= h / 224
    pnts = torch.abs(pnts).type(torch.int)

    for p in pnts:
        print(p)
        img_np = cv.circle(img_np.copy(), (p[0], p[1]), 5, (0, 255, 0), -1)

    plt.imshow(img_np)
    plt.show()


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FSNet().to(device)
    model.load_state_dict(torch.load('./pretrain/50epoch_0.00010lr_2.67666loss(train)_2.79753loss(val).pth'))
    model.eval()

    img_pth = './sample/person1.jpg'

    inference(model, img_pth, device)


if __name__ == '__main__':
    main()





































