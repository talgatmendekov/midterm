import cv2
import matplotlib.pyplot as plt


image = cv2.imread("test.jpg") 

if image is None:
    print("Error: Image not found!")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

original_size = image.shape
print("Original Size (H, W, C):", original_size)

half_image = cv2.resize(image_rgb, (image_rgb.shape[1]//2, image_rgb.shape[0]//2))

equalized_gray = cv2.equalizeHist(gray)


plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Color Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis("off")


plt.subplot(2, 2, 3)
plt.imshow(half_image)
plt.title("Resized to Half")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(equalized_gray, cmap='gray')
plt.title("Brightness Improved (Histogram Equalization)")
plt.axis("off")

plt.show()