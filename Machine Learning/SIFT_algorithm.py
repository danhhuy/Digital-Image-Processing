# Xử dụng bộ dò SIFT để tìm các điểm và đặc trưng giống nhau giữa 2 ảnh, Ứng dụng có thể là mix 2 ảnh vào làm 1
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_sift_features(img):
    sift_initialize = cv2.xfeatures2d.SIFT_create()
    key_points, descriptors = sift_initialize.detectAndCompute(img, None)
    return key_points, descriptors
def showing_sift_features(img1, img2, key_points):
    return plt.imshow(cv2.drawKeypoints(img1, key_points,img2.copy()))

print("'Make Sure that both the images are in the same folder '")
x = input("Enter First Image Name: ")
Image1 = cv2.imread(x)
y = input("Enter Second Image Name: ")
Image2 = cv2.imread(y)
Image1_gray = cv2.cvtColor(Image1, cv2.COLOR_BGR2GRAY)
Image2_gray = cv2.cvtColor(Image2, cv2.COLOR_BGR2GRAY)
Image1_key_points, Image1_descriptors = extract_sift_features(Image1_gray)
Image2_key_points, Image2_descriptors = extract_sift_features(Image2_gray)
print( 'Displaying SIFT Features')
showing_sift_features(Image1_gray, Image1, Image1_key_points);
norm = cv2.NORM_L2
bruteForce = cv2.BFMatcher(norm)
matches = bruteForce.match(Image1_descriptors, Image2_descriptors)
matches = sorted(matches, key = lambda match:match.distance)
matched_img = cv2.drawMatches(Image1, Image1_key_points,Image2, Image2_key_points,matches[:100], Image2.copy())

plt.figure(figsize=(100,300))
plt.imshow(matched_img)