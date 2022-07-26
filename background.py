import cv2
import numpy as np

import os 

bg_img = cv2.imread("/home/labrobotica01/localCodePtython/backgroud01.png")
f_img = cv2.imread("/home/labrobotica01/localCodePtython/img/Tomato_left_1.png", cv2.IMREAD_COLOR)
#f_img = cv2.imread("/home/labrobotica01/localCodePtython/PingoDOuro_left_1.png", cv2.IMREAD_COLOR)
#f_img = cv2.imread("/home/labrobotica01/localCodePtython/CBSLaranja_left_1.png", cv2.IMREAD_COLOR)
#f_img = cv2.imread("/home/labrobotica01/localCodePtython/DellValleMaca_left_1.png", cv2.IMREAD_COLOR)
#f_img = cv2.imread("/home/labrobotica01/localCodePtython/Melita500g_left_1.png", cv2.IMREAD_COLOR)
#f_img = cv2.imread("/home/labrobotica01/localCodePtython/Stella_left_1.png", cv2.IMREAD_COLOR)
#f_img = cv2.imread("/home/labrobotica01/localCodePtython/DTone_left_1.png", cv2.IMREAD_COLOR)
#f_img = cv2.imread("/home/labrobotica01/localCodePtython/Moca_left_1.png", cv2.IMREAD_COLOR)
#f_img = cv2.imread("/home/labrobotica01/localCodePtython/Yakissoba_left_1.png", cv2.IMREAD_COLOR)


# deixar a imagem de fundo do mesmo tamanho da imagem capturada 
imS = cv2.resize(bg_img, (640, 480))
# converte para sistema HSV
hsv = cv2.cvtColor(f_img, cv2.COLOR_BGR2HSV)


# remove o fundo verde da imagem
l_green = np.array([30,10,10])
u_green = np.array([105,255,255])
mask = cv2.inRange(hsv, l_green, u_green)
cv2.imshow("Mask", mask)

# recorte da imagem 
img_crop = hsv[126:400, 218:400]
equalize = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
ret,thresh_image = cv2.threshold(equalize,100,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
equalize= cv2.equalizeHist(thresh_image)
cv2.imshow("Threshold", equalize)

kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(equalize, kernel, iterations=4)
kernel = np.ones((3,3), np.uint8)
erosion = cv2.dilate(erosion, kernel, iterations=10)
cv2.imshow('Dilate', erosion)

contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if cv2.contourArea(c) > 1:
        cv2.rectangle(erosion, (x, y), (x + w, y + h), (0, 0, 255), 2)
        print(cv2.contourArea(c))

cv2.imshow('First contour with bounding box', erosion)


# inverte a imagem
ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((1,1), np.uint8)
erode = cv2.erode(thresh, kernel)
cv2.imshow('Opened', erode)

#imask = erosion  == 255
#green = np.zeros_like(f_img, np.uint8)
#green[imask] = f_img[imask]

# Applying the Canny Edge filter
#edge = cv2.Canny(mask, 50, 100)
#kernel = np.ones((3,3), np.uint8),
#erosion = cv2.dilate(edge, kernel)
#cv2.imshow('Dilate', erosion)

inv = erode
contours, hierarchy = cv2.findContours(inv, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
#contour1 = cv2.drawContours(f_img, contours, 0,(255,0,255),3)
#x, y, w, h = cv2.boundingRect(contours[3])
#cv2.rectangle(contour1,(x,y), (x+w,y+h), (0,0,255), 5)
#cv2.imshow('First contour with bounding box', contour1)

# find largest area contour
max_area = -1
for c in range(len(contours)):
    area = cv2.contourArea(contours[c])
    if area>max_area:
        cnt = contours[c]
        max_area = area

#x, y, w, h = cv2.boundingRect(cnt)
#cv2.rectangle(f_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
#print(cv2.contourArea(cnt))
hull = cv2.convexHull(cnt)
cv2.fillPoly(mask, pts=[hull], color=(0, 0, 255))

res = cv2.bitwise_and(f_img, f_img, mask=mask)
dif = f_img-res

#for c in contours:
#    x, y, w, h = cv2.boundingRect(c)
 
#    dim = (640.0*480.0)/5.0; 
    # Make sure contour area is large enough
#    if cv2.contourArea(c) > 1000.0 and cv2.contourArea(c) < dim:
#        cv2.rectangle(f_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
#        print(cv2.contourArea(c))
 
#cv2.imshow('All contours with bounding box',f_img)

fundo = np.where(dif==0, imS, dif)
cv2.imshow("Dif", dif)
cv2.imshow("Original", f_img)
cv2.imshow("Result", fundo)

cv2.waitKey()
cv2.destroyAllWindows()
