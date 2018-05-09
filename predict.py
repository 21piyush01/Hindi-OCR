from __future__ import print_function
from sklearn.externals import joblib
from hog import HOG
import dataset
import argparse
import mahotas
import cv2
import imutils
import numpy as np

def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to where the model will be stored")
ap.add_argument("-i", "--image", required=True, help="path to the image file")
args = vars(ap.parse_args())

model = joblib.load(args["model"])

hog = HOG(orientations=18, pixelsPerCell=(10,10), cellsPerBlock=(1,1), transform=True)

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
lines = cv2.HoughLines(edges,1,np.pi/180, 200)
 
for r,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*r
    y0 = b*r
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(image,(x1,y1), (x2,y2), (255,255,255),3)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(blurred, 30, 150)
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts.sort(key=lambda x:get_contour_precedence(x, image.shape[1]))

f = open("output.txt", "w")

for c in cnts:
  (x,y,w,h) = cv2.boundingRect(c)
  roi = gray[y:y+h, x:x+w]
  thresh = roi.copy()
  T = mahotas.thresholding.otsu(roi)
  thresh[thresh > T] = 255
  thresh = cv2.bitwise_not(thresh)
  thresh = dataset.deskew(thresh, 20)
  thresh = dataset.center_extent(thresh, (20,20))
  cv2.imshow("thresh", thresh)
  hist = hog.describe(thresh)
  razor = model.predict([hist])[0]
  if razor == 0:
    goku = 'A'
  elif razor == 1:
    goku = '['
  elif razor == 2:
    goku = ']'    
  elif razor == 3:
    goku = '}'
  elif razor == 4:
    goku = 'e'
  elif razor == 5:
    goku = 'xa'
  elif razor == 6:
    goku = 'k'
  elif razor == 7:
    goku = 'K'
  elif razor == 8:
    goku = 'g'
  elif razor == 9:
    goku = 'Ga'
  elif razor == 10:
    goku = 'ca'
  elif razor == 11:
    goku = 'C'
  elif razor == 12:
    goku = 'ja'
  elif razor == 13:
    goku = 'Ja'
  elif razor == 14:
    goku = 'T'
  elif razor == 15:
    goku = 'z'
  elif razor == 16:
    goku = 'D'
  elif razor == 17:
    goku = 'Z'
  elif razor == 18:
    goku = 'N'
  elif razor == 19:
    goku = 't'
  elif razor == 20:
    goku = 'qa'
  elif razor == 21:
    goku = 'd'
  elif razor == 22:
    goku = 'Qa'
  elif razor == 23:
    goku = 'na'
  elif razor == 24:
    goku = 'p'
  elif razor == 25:
    goku = 'f'
  elif razor == 26:
    goku = 'ba'
  elif razor == 27:
    goku = 'Ba'
  elif razor == 28:
    goku = 'ma'
  elif razor == 29:
    goku = 'ya'
  elif razor == 30:
    goku = 'r'
  elif razor == 31:
    goku = 'la'                
  elif razor == 32:
    goku = 'L'
  elif razor == 33:
    goku = 'va'
  elif razor == 34:
    goku = 'S'
  elif razor == 35:
    goku = 'Ya'
  elif razor == 36:
    goku = 'sa'
  elif razor == 37:
    goku = 'h'
  elif razor == 38:
    goku = 'a'
  elif razor == 39:
    goku = '&'
  elif razor == 40:
    goku = 'o'
  elif razor == 41:
    goku = 'O'
  elif razor == 42:
    goku = '-'
  elif razor == 43:
    goku = '^'
  elif razor == 44:
    goku = 'u'
  elif razor == 45:
    goku = 'U'
  elif razor == 46:
    goku = '~'
  elif razor == 49:
    goku = 'I'
  else:
    goku = 'NONE'  
  f.write(goku + " ")
  print("I think that the character is : {}".format(razor))
  cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 1)
  cv2.putText(image, str(razor), (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
  cv2.imshow("image", image)
  cv2.waitKey(0)

f.close()  