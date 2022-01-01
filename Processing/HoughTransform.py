import cv2
import numpy as np

# Tim duong thang
# bieu dien duong thang bang p = x.cos(theta) + y.sin(theta)
# 1 duong thang sang khong gian Hough la 1 diem, 1 diem sang khong gian Hough la mot hinh sin, noi giao nhau cua nhieu hinh sin
# la cac diem nam cung 1 hang, loc nguong se duoc duong thang.
# read image
img = cv2.imread('geometry.jpg')
# convert to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # color -> gray
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)    
cv2.imwrite('geo_hough.jpg',img)

#tim duong tron
# Mỗi một đường tròn sẽ có 1 tâm va 1 ban kính, áp dụng bài toán phát hiện cạnh, ta dược các đường bao của hình tròn.
# Tại mỗi điểm của hình bao đường tròn đấy ta vẽ 1 đường tròn khác có tâm tại điểm đấy và cùng bán kính với hình tròn gốc.
# các đường tròn sẽ cắt nhau nhiều nhất tại 1 điểm, dùng ngưỡng lọc bỏ cá điểm không cần thiết đi ta được tâm hình tròn.
import sys
import cv2 as cv
import numpy as np
def main(argv):
    
    default_file = 'smarties.png'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    
    
    gray = cv.medianBlur(gray, 5)
    
    
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                            param1=100, param2=30,
                            minRadius=1, maxRadius=30)
    
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)
    
    
    cv.imshow("detected circles", src)
    cv.waitKey(0)
    
    return 0
if __name__ == "__main__":
    main(sys.argv[1:])