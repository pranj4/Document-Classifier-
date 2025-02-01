import cv2
import matplotlib.pyplot as plt

image_path = "/config\\data\\docs-sm\\invoice\\0000137486.jpg"  # Change as needed
image = cv2.imread(image_path)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
