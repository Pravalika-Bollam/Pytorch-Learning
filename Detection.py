import cv2
import  albumentations as A
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os 




image = Image.open("Dog and Cat .png/Cat/109.png")
image2 = cv2.imread("Dog and Cat .png/Cat/109.png")
gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
_, tresh= cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(tresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

x_min, y_min, w, h = cv2.boundingRect(contours[0])
x_max = x_min+w
y_max = y_min + h
bboxes = [[81, 26, 280, 340]]
#the above code is in the form of pascal_voc = (x_min, y_min, x_max, y_max)


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
    ], 
    bbox_params = A.BboxParams(format = "pascal_voc", min_area= 80,
                                min_visibility=0.3, label_fields=[])
)
image = np.array(image)
images_list = [image]
saved_bboxes = [bboxes[0]] # saving the og

output_dir = "DeepLearning/DetectedImages"
os.makedirs(output_dir, exist_ok=True)

for i in range(15):
    augmentations = transform(image = image, bboxes = bboxes) # returns a dictionary
    aug_image = augmentations["image"]
    if len(augmentations["bboxes"]) == 0:
        continue
    aug_bbox = augmentations["bboxes"][0]
    images_list.append(aug_image)
    saved_bboxes.append(aug_bbo)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(aug_image)
    xmin, ymin, xmax, ymax = aug_bbox
    width = xmax - xmin
    height = ymax - ymin
    rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.axis('off')
    output_path = os.path.join(output_dir, f"augmented_image_{i+1}.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


