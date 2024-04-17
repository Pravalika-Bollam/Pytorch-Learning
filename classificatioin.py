import cv2
import  albumentations as A
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt




image = Image.open("DeepLearning/marina.jpg")
transform = A.Compose(
    [
        A.Resize(width = 1920, height = 1080),
        A.RandomCrop(width = 1280, height = 720),
        A.Rotate(limit = 40, p = 0.9),
        A.HorizontalFlip(p = 0.46),
        A.VerticalFlip(p = 0.1),
        A.RGBShift(r_shift_limit =25, g_shift_limit = 25, b_shift_limit = 25, p = 0.8),
        A.OneOf(
            [
                A.Blur(p = 0.7),
                A.ColorJitter(p = 0.8 ),
                
            ], p= 0.9,
        ),
    ]
    
)

images_list = [image]
image = np.array(image)
for i in range(15):
    augmentations = transform(image = image) # returns a dictionary
    aug_image = augmentations["image"]
    images_list.append(aug_image)

fig, axs = plt.subplots(4, 4, figsize=(12, 4))
for i, image_array in enumerate(images_list):
    row_index = i // 4
    col_index = i % 4
    axs[row_index, col_index].imshow(image_array)
    axs[row_index, col_index].axis('off')  
plt.tight_layout()
plt.show()

    
    