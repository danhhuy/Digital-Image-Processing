import cv2
import numpy as np
from skimage import io
import os 
from math import sqrt

file_imggt = "./traonguocdoA/GT/a"
file_imggoc = "./traonguocdoA/Orig/a"

string = ""

for h in range(227):
    
    path_imggt = file_imggt + str(h) + ".jpg"
    path_imggoc = file_imggoc + str(h) + ".jpg"
    imCheck = cv2.imread("./traonguocdoA/Orig/a0.jpg")
    img_gt = io.imread(path_imggt)
    img_goc = io.imread(path_imggoc)
    cv2.imshow('0',img_goc )
    img_gt = cv2.resize(img_gt, (1280, 960))
    img_goc = cv2.resize(img_goc, (1280, 960))
    kernel = np.ones((20, 20), np.uint8)
    img3 = cv2.erode(img_gt, kernel, iterations=1)
    img4 = cv2.dilate(img_gt, kernel, iterations=1)
    img2_gray = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
    img3_gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    img4_gray = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    ret1, img2_binary = cv2.threshold(img2_gray, 127, 255, cv2.THRESH_BINARY)
    ret2, img3_binary = cv2.threshold(img3_gray, 127, 255, cv2.THRESH_BINARY)
    ret3, img4_binary = cv2.threshold(img4_gray, 127, 255, cv2.THRESH_BINARY)
    contour1 , hierarchy1 = cv2.findContours(img2_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contour2 , hierarchy2 = cv2.findContours(img3_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contour3 , hierarchy3 = cv2.findContours(img4_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('1',img_goc )
#hien duong bao
    cv2.drawContours(img_goc,contour1,-1,(0,255,0),1)
    cv2.drawContours(img_goc,contour2,-1,(255,0,0),1)
    cv2.drawContours(img_goc,contour3,-1,(0,0,255),1)
    cv2.imshow('2',img_goc )
# duong inner outer nhi phan
    inner = cv2.subtract(img2_binary, img3_binary)
    outer = cv2.subtract(img4_binary, img2_binary)

    outer = outer != 0
    outer = outer.astype(np.uint8)
    
    cv2.imshow('3',img_goc )
# duong inner outer RGB
    blue0, green0, red0 = cv2.split(img_goc)
    r0 = cv2.multiply(outer,red0)
    g0 = cv2.multiply(outer,green0)
    b0 = cv2.multiply(outer,blue0)
    outer_RGB = cv2.merge([b0, g0, r0])
    
    inner = inner != 0
    inner = inner.astype(np.uint8)

    blue1, green1, red1 = cv2.split(img_goc)
    r1 = cv2.multiply(inner,red1)
    g1 = cv2.multiply(inner,green1)
    b1 = cv2.multiply(inner,blue1)
    inner_RGB = cv2.merge([b1, g1, r1])

    image_RGB = cv2.add(inner_RGB,outer_RGB)
    cv2.imshow('4',img_goc )
#cv2.imshow('outer',outer_RGB)
#cv2.imshow('inner',inner_RGB)
#cv2.imshow('image',image_RGB)
#cv2.imshow('img',img_goc)
#cv2.waitKey(0)
    count1 = 0
    sum_red1 = 0
    sum_green1 = 0
    sum_blue1 = 0
    count2 = 0
    sum_red2 = 0
    sum_green2 = 0
    sum_blue2 = 0
#tính mean
    for i in range(960):
        for j in range(1280):
            if inner_RGB[i,j,0] > 0 or inner_RGB[i,j,1] > 0 or inner_RGB[i,j,2] > 0:
                count1+=1
                sum_red1+= img_goc [i,j,2]
                sum_green1+= img_goc [i,j,1]
                sum_blue1+= img_goc [i,j,0]
                print(img_goc [i,j,2])
            if outer_RGB[i,j,0] > 0 or outer_RGB[i,j,1] > 0 or outer_RGB[i,j,2] > 0:
                count2+=1
                sum_red2+= img_goc [i,j,2]
                sum_green2+= img_goc [i,j,1]
                sum_blue2+= img_goc [i,j,0]
    if count1 == 0 and count2 == 0:
        mean_red1 = " "
        mean_green1 = " "
        mean_blue1 = " "
        mean_red2 = " "
        mean_green2 = " "
        mean_blue2 = " "
    else:   
        mean_red1 = round((sum_red1 / count1),2)
        mean_green1 = round((sum_green1 / count1),2)
        mean_blue1 = round((sum_blue1 / count1),2)
        
        mean_red2 = round((sum_red2 / count2),2)
        mean_green2 = round((sum_green2 / count2),2)
        mean_blue2 = round((sum_blue2 / count2),2)
        
    mean1 = (mean_red1, mean_green1, mean_blue1)
    mean2 = (mean_red2, mean_green2, mean_blue2)

    cv2.imshow('anhgoc', img_goc)
    print(mean_red1)
    print(mean_green1)
    print(mean_blue1)
    print(mean_red2)
    print(mean_green2)
    print(mean_blue2)
    print(count1)
    print(count2)
    print("////////////////")
    count1 = 0
    sum_red1 = 0
    sum_green1 = 0
    sum_blue1 = 0
    count2 = 0
    sum_red2 = 0
    sum_green2 = 0
    sum_blue2 = 0
    for i in range(960):
        for j in range(1280):
            if inner_RGB[i,j,0] > 0 or inner_RGB[i,j,1] > 0 or inner_RGB[i,j,2] > 0:
                count1+=1
                sum_red1+= imCheck [i,j,2]
                sum_green1+= imCheck [i,j,1]
                sum_blue1+= imCheck [i,j,0]
                print(imCheck [i,j,2])
            if outer_RGB[i,j,0] > 0 or outer_RGB[i,j,1] > 0 or outer_RGB[i,j,2] > 0:
                count2+=1
                sum_red2+= imCheck [i,j,2]
                sum_green2+= imCheck [i,j,1]
                sum_blue2+= imCheck [i,j,0]
    if count1 == 0 and count2 == 0:
        mean_red1 = " "
        mean_green1 = " "
        mean_blue1 = " "
        mean_red2 = " "
        mean_green2 = " "
        mean_blue2 = " "
    else:   
        mean_red1 = round((sum_red1 / count1),2)
        mean_green1 = round((sum_green1 / count1),2)
        mean_blue1 = round((sum_blue1 / count1),2)
        
        mean_red2 = round((sum_red2 / count2),2)
        mean_green2 = round((sum_green2 / count2),2)
        mean_blue2 = round((sum_blue2 / count2),2)
    cv2.imshow('anhgoc',imCheck )
    print(mean_red1)
    print(mean_green1)
    print(mean_blue1)
    print(mean_red2)
    print(mean_green2)
    print(mean_blue2)
    print(count1)
    print(count2)
    cv2.waitKey(0)
    #print(mean1)
    #print (file, ": ", mean1, '\t')
    #string = string + file + ":" + str(mean1) + "\t"
    
    #image2 = io.imread('b0.jpg')
    img_gt = img_gt.astype(np.uint64)
#image1 = io.imread('b0gt.jpg')
    img_goc = img_goc.astype(np.uint64)

    square_red1 = 0
    square_green1 = 0
    square_blue1 = 0
    square_red2 = 0
    square_green2 = 0
    square_blue2 = 0
#tính variance
    for i in range(960):
        for j in range(1280):
            if inner_RGB[i,j,0] > 0 or inner_RGB[i,j,1] > 0 or inner_RGB[i,j,2] > 0:
                square_red1 += (img_goc[i,j,2] - mean_red1)**2
                square_green1+= (img_goc[i,j,1] - mean_green1)**2
                square_blue1+= (img_goc[i,j,0] - mean_blue1)**2
            if outer_RGB[i,j,0] > 0 or outer_RGB[i,j,1] > 0 or outer_RGB[i,j,2] > 0:
                square_red2 += (img_goc[i,j,2] - mean_red2)**2
                square_green2+= (img_goc[i,j,1] - mean_green2)**2
                square_blue2+= (img_goc[i,j,0] - mean_blue2)**2
    if count1 == 0 and count2 ==0:
        var_red1 = " "
        var_green1 = " "
        var_blue1 = " "
        var_red2 = " "
        var_green2 = " "
        var_blue2 = " "
    else:
        var_red1 = round((square_red1 / count1),2)
        var_green1 = round((square_green1 / count1),2)
        var_blue1 = round((square_blue1 / count1),2)
        var1 = (var_red1, var_green1, var_blue1)
        
        var_red2 = round((square_red2 / count2),2)
        var_green2 = round((square_green2 / count2),2)
        var_blue2 = round((square_blue2 / count2),2)
        var2 = (var_red2, var_green2, var_blue2)
        
        
    #print(var1)
    #print (file, ": ", var1, '\t')
    #string = string1 + file + ":" + str(var1) + "\t"


#tính m3############
    triple_red1 = 0
    triple_green1 = 0
    triple_blue1 = 0
    triple_red2 = 0
    triple_green2 = 0
    triple_blue2 = 0
    for i in range(960):
        for j in range(1280):
            if inner_RGB[i,j,0] > 0 or inner_RGB[i,j,1] > 0 or inner_RGB[i,j,2] > 0:
            
                triple_red1 += (img_goc[i,j,2] - mean_red1)**3
                triple_green1+= (img_goc[i,j,1] - mean_green1)**3
                triple_blue1+= (img_goc[i,j,0] - mean_blue1)**3
            if outer_RGB[i,j,0] > 0 or outer_RGB[i,j,1] > 0 or outer_RGB[i,j,2] > 0:
                triple_red2 += (img_goc[i,j,2] - mean_red2)**3
                triple_green2 += (img_goc[i,j,1] - mean_green2)**3
                triple_blue2 += (img_goc[i,j,0] - mean_blue2)**3
    if count1 == 0 and count2 == 0:
        m3_red1 = " "
        m3_green1 = " "
        m3_blue1 = " "
        m3_red2 = " "
        m3_green2 = " "
        m3_blue2 = " "
        
        skewness_red1 =  " "
        skewness_green1 =  " "
        skewness_blue1 =  " "
        skewness_red1 =  " "
        skewness_green1 =  " "
        skewness_blue1 =  " "
        
    else: 
        
        m3_red1 = round((triple_red1 / count1),2)
        m3_green1 = round((triple_green1 / count1),2)
        m3_blue1 = round((triple_blue1 / count1),2)
        #m3 = (m3_red1, m3_green1, m3_blue1)
        
        m3_red2 = round((triple_red2 / count2),2)
        m3_green2 = round((triple_green2 / count2),2)
        m3_blue2 = round((triple_blue2 / count2),2)
        #m3 = (m3_red2, m3_green2, m3_blue2)
    #print(m3)

#Skewness
        skewness_red1 = round(((sqrt(count1 * (count1 - 1)) / (count1 - 2)) * m3_red1 / (var_red1**1.5)), 2)
        skewness_green1 = round(((sqrt(count1 * (count1 - 1)) / (count1 - 2)) * m3_green1 / (var_green1**1.5)), 2)
        skewness_blue1 = round(((sqrt(count1 * (count1 - 1)) / (count1 - 2)) * m3_blue1 / (var_blue1**1.5)), 2)
        skewness1 = (skewness_red1, skewness_green1, skewness_blue1)
        
        skewness_red2 = round(((sqrt(count2 * (count2 - 1)) / (count2 - 2)) * m3_red2 / (var_red2**1.5)), 2)
        skewness_green2 = round(((sqrt(count2 * (count2 - 1)) / (count2 - 2)) * m3_green2 / (var_green2**1.5)), 2)
        skewness_blue2 = round(((sqrt(count2 * (count2 - 1)) / (count2 - 2)) * m3_blue2 / (var_blue2**1.5)), 2)
        skewness2 = (skewness_red2, skewness_green2, skewness_blue2)
    #print (skewness1)
    #print (file, ": ", skewness1, '\t')
    #string1 = string1 + file + ":" + str(skewness1) + "\t"





#tính m4
    tetra_red1 = 0
    tetra_green1 = 0
    tetra_blue1 = 0
    tetra_red2 = 0
    tetra_green2 = 0
    tetra_blue2 = 0
    for i in range(960):
        for j in range(1280):
            if inner_RGB[i,j,0] > 0 or inner_RGB[i,j,1] > 0 or inner_RGB[i,j,2] > 0:
            
                tetra_red1 += (img_goc[i,j,2] - mean_red1)**4
                tetra_green1+= (img_goc[i,j,1] - mean_green1)**4
                tetra_blue1+= (img_goc[i,j,0] - mean_blue1)**4
            if outer_RGB[i,j,0] > 0 or outer_RGB[i,j,1] > 0 or outer_RGB[i,j,2] > 0:
                tetra_red2 += (img_goc[i,j,2] - mean_red2)**4
                tetra_green2+= (img_goc[i,j,1] - mean_green2)**4
                tetra_blue2+= (img_goc[i,j,0] - mean_blue2)**4
                
    if count1 == 0 and count2 == 0:
        m4_red1 = " "
        m4_green1 = " "
        m4_blue1 = " "
        m4_red2 = " "
        m4_green2 = " "
        m4_blue2 = " "
        kurtosis_red1 = " "
        kurtosis_green1 = " "
        kurtosis_blue1 = " "
        kurtosis_red2 = " "
        kurtosis_green2 = " "
        kurtosis_blue2 = " "
    else:
        
        m4_red1 = round((tetra_red1 / count1),2)
        m4_green1 = round((tetra_green1 / count1),2)
        m4_blue1 = round((tetra_blue1 / count1),2)
        m4 = (m4_red1, m4_green1, m4_blue1)
        
        m4_red2 = round((tetra_red2 / count1),2)
        m4_green2 = round((tetra_green2 / count1),2)
        m4_blue2 = round((tetra_blue2 / count1),2)
        m4 = (m4_red1, m4_green1, m4_blue1)
#print(m4)


#Kurtosis

        kurtosis_red1 = round((count1 - 1) / ((count1 - 2) * (count1 - 3)) *((count1 + 1) * (m4_red1 / (var_red1**2) - 3) + 6), 2)
        kurtosis_green1 = round((count1 - 1) / ((count1 - 2) * (count1 - 3)) *((count1 + 1) * (m4_green1 / (var_green1**2) - 3) + 6), 2)
        kurtosis_blue1 = round((count1 - 1) / ((count1 - 2) * (count1 - 3)) *((count1 + 1) * (m4_blue1 / (var_blue1**2) - 3) + 6), 2)
        #kurtosis1 = (kurtosis_red1, kurtosis_green1, kurtosis_blue1)
        
        kurtosis_red2 = round((count2 - 1) / ((count2 - 2) * (count2 - 3)) *((count2 + 1) * (m4_red2 / (var_red2**2) - 3) + 6), 2)
        kurtosis_green2 = round((count2 - 1) / ((count2 - 2) * (count2 - 3)) *((count2 + 1) * (m4_green2 / (var_green2**2) - 3) + 6), 2)
        kurtosis_blue2 = round((count2 - 1) / ((count2 - 2) * (count2 - 3)) *((count2 + 1) * (m4_blue2 / (var_blue2**2) - 3) + 6), 2)
      
    print(str(h), mean_red1, mean_green1, mean_blue1, var_red1, var_green1, var_blue1, skewness_red1, skewness_green1, skewness_blue1, kurtosis_red1, kurtosis_green1, kurtosis_blue1, '\n', sep = "\t" )
    if count1 == 0 and count2 == 0:
        string = str(h) + " " + "1" + "\n" + str(h) + " " + "0" + "\n"
    else:
        string = str(h) + " " + "1" + " " + str(mean_red1) + " " + str(mean_green1) + " " + str(mean_blue1) + " " + str(var_red1) + " " + str(var_green1) + " " + str(var_blue1) + " " + str(skewness_red1) + " " + str(skewness_green1) + " " + str(skewness_blue1) + " " + str(kurtosis_red1) + " " + str(kurtosis_green1) + " " + str(kurtosis_blue1) + "\n" + str(h) + " " + "0" + " " + str(mean_red2) + " " + str(mean_green2) + " " + str(mean_blue2) + " " + str(var_red2) + " " + str(var_green2) + " " + str(var_blue2) + " " + str(skewness_red2) + " " + str(skewness_green2) + " " + str(skewness_blue2) + " " + str(kurtosis_red2) + " " + str(kurtosis_green2) + " " + str(kurtosis_blue2) + "\n"   
    #string2 = string2 + file + ":" + str(var1) + "\n"
    #string3 = string3 + file + ":" + str(skewness1) + "\n"
    #string4 = string4 + file + ":" + str(kurtosis1) + "\t"
    #print(kurtosis1)
    with open("momentStatistic_new.txt", "a") as f:
        f.write(string)
    
f.close()
    
print ("done")
    
