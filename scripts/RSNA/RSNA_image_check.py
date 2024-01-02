import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import cv2

image_dir = '/home/santosh.sanjeev/rsna_18/train/'
image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

selected_images = random.sample(image_files, 6)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for i, image_path in enumerate(selected_images):
    # img = Image.open(image_path).convert('RGB')
    img = cv2.imread(image_path)
    print(img.shape)
    row = i // 3
    col = i % 3
    axes[row, col].imshow(img)
    axes[row, col].axis('off')

plt.subplots_adjust(wspace=0.05, hspace=0.1)
plt.savefig('selected_images_subplot.png', bbox_inches='tight', pad_inches=0.1)
plt.show()
