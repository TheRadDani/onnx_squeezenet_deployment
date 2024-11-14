import numpy as np
from PIL import Image

image = Image.open("images/cat.jpg")
image_array = np.array(image)

shape = image_array.shape
if len(shape) == 3:
    if shape[2] == 3:  # Assuming RGB images
        print("Image format is HWC")
    else:
        print("Image format is likely CHW")
else:
    print("Image format is not recognized")

print(image_array.shape)