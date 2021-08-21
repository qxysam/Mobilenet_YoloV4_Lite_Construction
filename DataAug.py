# 练数据的可变行，增强泛化能力-----------------------------------------------------------------------------------
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import random
ia.seed(1)
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.02)), # random crops
    iaa.Cutout(nb_iterations=1,size=0.1,squared=False),
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 1.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.65, 1.2)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.3),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (1.0, 1.5), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.1)},
        rotate=(-15, 10),
        shear=(-0.4, 0.4)
    ),
    iaa.WithChannels(0, iaa.Add((10, 100)))
], random_order=True) # apply augmenters in random order

imglist = []
img = cv2.imread('kobe.jpg')
img2 = cv2.imread('kobe2.jpg')
imglist.append(img)
imglist.append(img2)
images_aug = seq.augment_images(imglist)
cv2.imwrite("im_after.jpg", images_aug[0])
cv2.imwrite("im_after2.jpg", images_aug[1])