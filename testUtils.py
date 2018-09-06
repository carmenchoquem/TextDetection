#coding: utf-8
#pip install scipy
from imutils import perspective
import numpy as np
import cv2


imagen = cv2.imread("notecard.png")
#clone = imagen.copy()

print("Mostrando imagen")
cv2.imshow("imagen",imagen)
print("Imagen mostrada")
#cv2.waitKey(0)
#print("Destruyendo ventana")
cv2.destroyAllWindows()

