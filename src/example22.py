import cv2

image = cv2.imread("./data/one_pounch_man.jpg")
print("image shape: {}".format(image.shape))

cv2.imshow("Image", image)
cv2.waitKey(0)

bgr_pixel_1 = image[20, 100]    # accesses pixel at x=100, y=20
bgr_pixel_2 = image[75, 25]     # accesses pixel at x=25, y=75
bgr_pixel_3 = image[90, 85]     # accesses pixel at x=85, y=90

print('bgr_pixel_1: {}'.format(bgr_pixel_1))
print('bgr_pixel_2: {}'.format(bgr_pixel_2))
print('bgr_pixel_3: {}'.format(bgr_pixel_3))

