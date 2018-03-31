# importování knihoven
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import numpy as np
import tensorflow as tf
import cv2
import math


# pomocné funkce pro přípravu snímků

# zmenší rozlišení snímku, ale zachová poměr stran
def resize_img(img, img_size):
    y = math.floor((img_size/img.shape[0]) * img.shape[1])
    resized = cv2.resize(img, (y, img_size))
    return resized
   
# ořízne obrázek horizontálně
def crop_img_hor(img, final_width):
    left_border = math.floor((img.shape[1] - final_width) / 2)
    right_border = final_width + left_border
    img_cropped = img[:, left_border:right_border]
    return img_cropped

# ořízne obrázek horizontálně
def crop_img_ver(img, final_width):
    top_border = math.floor((img.shape[0] - final_width) / 2)
    bottom_border = final_width + top_border
    img_cropped = img[top_border: bottom_border, :]
    return img_cropped



# proměnné určené pro konfiguracu nahrávacího procesu

# rozlišení ve kterém bude kamera nahrávat
res = (320, 240)

# velikost stran obrázku, který bude uložen
img_size = 64

# velikost stran po oříznutí (ovlivňuje zorné pole)
cropped_size = 190

# poměr zaznamenanných snímků = každý pátý snímek bude zaznamenán
capture_ratio = 5

# frekvence snímků za sekund při která kamera pracuje
fps = 30

# slozka kam se obrázky budou ukládat
PATH = 'data/dislike/train/like'

# jméno každého obrázku + číslo
img_name = 'like'

# první obrazek bude mít za jménem toto číslo
start_index = 150

# počet snímků pro zaznamenání
img_count = 200


# inicializace kamery
camera = PiCamera()
camera.resolution = res
camera.framerate = fps
rawCapture = PiRGBArray(camera, size=res)

# zastavení programu pro 0.1 sekundy, aby se kamera mohlo rozehřát
time.sleep(0.1)

# první loop
# slouží pouze k ukázání výstupu z kamery na displeji
for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):

    # aktuální snímek jako numpy pole
    img = frame.array

    # převede snímek z RGB do grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img_gray.shape)
    
    # ořízne obrázek tak aby jeho rozlišení byle cropped_size x _cropped_size
    img_cropped = crop_img_hor(img_gray, cropped_size)
    img_cropped = crop_img_ver(img_cropped, cropped_size) 

    # nastavení písma pro zobrazení textu na obrázku na displeji
    font = cv2.FONT_HERSHEY_SIMPLEX

    # přidá text 'not recording' na obrázek na displej
    cv2.putText(img_cropped, "NOT RECORDING",(40, 30),font, 0.4, (0,155.0,), 2, cv2.LINE_AA)
    
    # ukáže snímek na displeji
    cv2.imshow("cropped", img_cropped)
    

    # zaznamenává zmáčknutí klávesy
    key = cv2.waitKey(1) & 0xFF
    

	# smaže poslední snímek z paměti
    rawCapture.truncate(0)
    
    # pokud dojde ke stisknutí klávesy 'q' loop skončí a program pokraču
    if key == ord('q'):
        break



# první loop
# slouží k nahrávání obrázků do předem určené složky
i = 0
for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    
    # aktuální snímek jako numpy pole
    img = frame.array
    
    # převede snímek z RGB do grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ořízne obrázek tak aby jeho rozlišení byle cropped_size x _cropped_size
    img_cropped = crop_img_hor(img_gray, cropped_size)
    img_cropped = crop_img_ver(img_cropped, cropped_size)

    # přidá text 'recording' na obrázek na displej
    cv2.putText(img_cropped, "RECORDING",(40, 30),font, 0.4, (0,255,0), 2, cv2.LINE_AA)
    
	# ukáže snímek na displeji
    cv2.imshow("cropped", img_cropped)
    
    # zaznamenává stisknutí klávesy
    key = cv2.waitKey(1) & 0xFF

    # smaže poslední snímek z paměti
    rawCapture.truncate(0)

    # pokud je zbytek dělení pořadí tohoto snímku poměrem zachycení, tak dojde k uložení snímku
    scaled_index = 0
    if i % capture_ratio == 0:
    	# změní číslo snímku, aby bylo o jedna větší než to minulé
        scaled_index = int(i - (i / capture_ratio) * (capture_ratio-1) + start_index)
        # vypíše cestu k obrázku
        print(PATH + img_name + str(scaled_index) + ".jpg")

        # zmenší rozlišení obrázku na předem určené rozlišení
        img_resized = resize_img(img_cropped, img_size)

        # uloží obrázek to před určené složky pod před určeným názvem
        cv2.imwrite(PATH + img_name + str(scaled_index) +  ".jpg", img_resized)
    

    i += 1
    
    # pokud dojke ke stisknutí klávesy 'q' nebo se uložení před určené množství snímků tak se loop ukončí
    if key == ord('q') or scaled_index == (img_count + start_index):
        break
