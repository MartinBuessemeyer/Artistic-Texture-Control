#!/usr/bin/python3
# coding=utf-8

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import saliency_estimation.LDF.train_fine.dataset as dataset
from saliency_estimation.LDF.train_fine.net import LDF


def predict_saliency(image: torch.Tensor) -> torch.Tensor:
    device = image.device
    with torch.no_grad():
        image = image.detach().clone()
        cfg = dataset.Config(snapshot=Path(__file__).parent / 'out' / 'model-40.pt', mode='test')
        transform_options = dataset.Data(cfg, no_load=True)
        net = LDF(cfg)
        net.train(False)
        net.eval()
        net.to(device)

        image = image.squeeze().permute(1, 2, 0).detach().clone().cpu().numpy() * 255
        shape = image.shape[:2]
        image = transform_options.normalize(image)
        image = transform_options.resize(image)
        image = transform_options.totensor(image).unsqueeze(0).to(device)

        outb1, outd1, out1, outb2, outd2, out2 = net(image, shape)
        out = out2
        pred = torch.sigmoid(out)
        return pred


class Test(object):
    def __init__(self, Dataset, Network, Path):
        ## dataset
        self.cfg = Dataset.Config(datapath=Path, snapshot='./out/model-40', mode='test')
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def save(self):
        with torch.no_grad():
            for image, (H, W), name in self.loader:
                image, shape = image.cuda().float(), (H, W)
                outb1, outd1, out1, outb2, outd2, out2 = self.net(image, shape)
                out = out2
                pred = torch.sigmoid(out[0, 0]).cpu().numpy() * 255
                head = '../eval/maps/LDF/' + self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0] + '.png', np.round(pred))


if __name__ == '__main__':
    for path in ['../data/ECSSD', '../data/PASCAL-S', '../data/DUTS', '../data/HKU-IS', '../data/DUT-OMRON',
                 '../data/THUR15K']:
        t = Test(dataset, LDF, path)
        t.save()
