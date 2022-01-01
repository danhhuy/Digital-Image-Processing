import cv2
# Sử dụng thư vien openCV2
img = cv2.imread('./Ass2/Lena.png')
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("using function OpenCV2", imgHSV)
# Sử dụng hàm
def rgb_to_hsv(pixel):
    b = pixel[0]
    g = pixel[1]
    r = pixel[2]
    r /= 255
    g /= 255
    b /= 255
    minc = min(r, g, b)
    v = max(r, g, b)
    if v == 0:
        s = 0
    else:
        s = (v-minc)/v
    if v == r:
        h = 60*(g-b)/(v-minc)
    elif v == g:
        h = 120+60*(b-r)/(v-minc)
    elif v == b:
        h = 240+60*(b-r)/(v-minc)
    else:
        h = 0
    if h < 0:
        h = h + 360
    return [round(h/2), round(s * 255), round(v * 255)]
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img[i][j] = rgb_to_hsv(img[i][j])
img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
cv2.imshow("sefl-made",  img)
cv2.waitKey(0)