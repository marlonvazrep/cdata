import cv2
from cv2 import grabCut
import numpy as np
import imutils


def grabcut(f_img):
    aux_img = f_img.copy()
    aux_img[121:373, 220:420] = 255 - aux_img[121:373, 220:420]

    bounding_box = [220, 121, 420-220, 373-121] 
    seg = np.zeros(f_img.shape[:2],np.uint8)
    x,y,width,height = [220, 121, 420-220, 373-121]
    seg[y:y+height, x:x+width] = 1
    background_mdl = np.zeros((1,65), np.float64)
    foreground_mdl = np.zeros((1,65), np.float64)

    cv2.grabCut(f_img, seg, bounding_box, background_mdl, foreground_mdl, 5,
        cv2.GC_INIT_WITH_RECT)

    # monta a mascara com o objeto
    mask_new = np.where((seg==2)|(seg==0),0,1).astype('uint8')
    #cv2.imshow('Mask', mask_new)

    img = f_img*mask_new[:,:,np.newaxis]
    #cv2.imshow('Output', img)

    # remonta a imagem invertida
    img_fundo = f_img.copy()
    img_fundo = 255
    new_img = np.where(img!=0, img_fundo, img)
    new_img = ~new_img
    #cv2.imshow("Teste", new_img)

    # converte a imagen para niveis de cinza
    normal = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    # dilata a imagem para remover os ruidos e ter a posição exata do objeto
    kernel = np.ones((3,3), np.uint8)
    dilate = cv2.dilate(normal, kernel, iterations=1)
    #cv2.imshow('Dilate', dilate)

    contours, hierarchy = cv2.findContours(~dilate, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)

    # find largest area contour
    max_area = -1
    for c in range(len(contours)):
        area = cv2.contourArea(contours[c])
        #print(area)
        if area>max_area:
            cnt = contours[c]
            max_area = area

    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(aux_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #cv2.imshow('Final', aux_img)

    # identifica a area de interesse  
    res = cv2.bitwise_and(f_img, f_img, mask=dilate)
    dif = f_img-res
    #cv2.imshow('Dif', dif)

    mask = dilate[y:y+h, x:x+w]
    img = dif[y:y+h, x:x+w]
    
    return mask, img

def add_obj(background, img, mask, x, y):
    '''
    Arguments:
    background - background image in CV2 RGB format
    img - image of object in CV2 RGB format
    mask - mask of object in CV2 RGB format
    x, y - coordinates of the center of the object image
    0 < x < width of background
    0 < y < height of background
    
    Function returns background with added object in CV2 RGB format
    
    CV2 RGB format is a numpy array with dimensions width x height x 3
    '''
    
    bg = background.copy()
    
    h_bg, w_bg = bg.shape[0], bg.shape[1]
    
    h, w = img.shape[0], img.shape[1]
    
    # Calculating coordinates of the top left corner of the object image
    x = x - int(w/2)
    y = y - int(h/2)    
    
    mask_boolean = mask[:,:,0] == 0
    mask_rgb_boolean = np.stack([mask_boolean, mask_boolean, mask_boolean], axis=2)
    
    if x >= 0 and y >= 0:
    
        h_part = h - max(0, y+h-h_bg) # h_part - part of the image which overlaps background along y-axis
        w_part = w - max(0, x+w-w_bg) # w_part - part of the image which overlaps background along x-axis

        bg[y:y+h_part, x:x+w_part, :] = bg[y:y+h_part, x:x+w_part, :] * ~mask_rgb_boolean[0:h_part, 0:w_part, :] + (img * mask_rgb_boolean)[0:h_part, 0:w_part, :]
        
    elif x < 0 and y < 0:
        
        h_part = h + y
        w_part = w + x
        
        bg[0:0+h_part, 0:0+w_part, :] = bg[0:0+h_part, 0:0+w_part, :] * ~mask_rgb_boolean[h-h_part:h, w-w_part:w, :] + (img * mask_rgb_boolean)[h-h_part:h, w-w_part:w, :]
        
    elif x < 0 and y >= 0:
        
        h_part = h - max(0, y+h-h_bg)
        w_part = w + x
        
        bg[y:y+h_part, 0:0+w_part, :] = bg[y:y+h_part, 0:0+w_part, :] * ~mask_rgb_boolean[0:h_part, w-w_part:w, :] + (img * mask_rgb_boolean)[0:h_part, w-w_part:w, :]
        
    elif x >= 0 and y < 0:
        
        h_part = h + y
        w_part = w - max(0, x+w-w_bg)
        
        bg[0:0+h_part, x:x+w_part, :] = bg[0:0+h_part, x:x+w_part, :] * ~mask_rgb_boolean[h-h_part:h, 0:w_part, :] + (img * mask_rgb_boolean)[h-h_part:h, 0:w_part, :]
    
    return bg


def join_img(img, bg, x, y):
    cut_mask, cut_img = grabcut(img)

    imS = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    cut_mask = cv2.cvtColor(cut_mask, cv2.COLOR_BGR2RGB)
    cut_img = cv2.cvtColor(cut_img, cv2.COLOR_BGR2RGB)
    imS = add_obj(imS, cut_img, cut_mask, x, y)
    
    return imS

def gaussian_noise(image, mean, var):
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out

def increase_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

bg_img = cv2.imread("/home/labrobotica01/localCodePtython/img_bkg/table01.JPG")
bg_img = cv2.cvtColor(bg_img, cv2.IMREAD_COLOR)
bg_img = cv2.resize(bg_img, (640, 480))

f_img01 = cv2.imread("/home/labrobotica01/localCodePtython/img/CBSLaranja_left_1.png")
f_img01 = cv2.cvtColor(f_img01, cv2.IMREAD_COLOR)

f_img02 = cv2.imread("/home/labrobotica01/localCodePtython/img/Moca_left_1.png")
f_img02 = cv2.cvtColor(f_img02, cv2.IMREAD_COLOR)

f_img03 = cv2.imread("/home/labrobotica01/localCodePtython/img/DellValleMaca_left_1.png")
f_img03 = cv2.cvtColor(f_img03, cv2.IMREAD_COLOR)

f_img04 = cv2.imread("/home/labrobotica01/localCodePtython/img/DTone_left_1.png")
f_img04 = cv2.cvtColor(f_img04, cv2.IMREAD_COLOR)

f_img05 = cv2.imread("/home/labrobotica01/localCodePtython/img/Sococo_left_1.png")
f_img05 = cv2.cvtColor(f_img05, cv2.IMREAD_COLOR)

rotated = imutils.rotate_bound(f_img03, 90)
rotated = cv2.resize(rotated, (640,480))
#cv2.imshow("Rotated (Correct)", rotated)


bg_img = join_img(f_img03, bg_img, 450, 250)
bg_img = join_img(f_img01, bg_img, 106, 210)
bg_img = join_img(f_img02, bg_img,  64, 280)
bg_img = join_img(rotated, bg_img, 573, 300)
bg_img = join_img(f_img04, bg_img, 192, 380)
bg_img = join_img(f_img05, bg_img, 333, 290)
cv2.imshow("Join", bg_img)

bg = bg_img.copy()

bg = gaussian_noise(bg, 0.001, 0.008)
cv2.imshow("Salt and pepper", bg)

bg =  bg_img.copy()
bg = increase_brightness(bg, 90)
cv2.imshow("Brightness", bg)

cv2.waitKey()
