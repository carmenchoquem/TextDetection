#coding: utf-8
# https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
#pip install scipy
#NOTA: Considerar que para ejecutar correctamente debe posicionarse en el directorio
# cd ./VisionArtificial
# Deiner Zapata Silva
from imutils import perspective
import numpy as np
import cv2
import time

namePaintWindow = "Lienzo canvas" #Nombre global del lienzo donde se va a dibujar y mostrar las imagenes
contCallbacks = 0
imagen = []
imagenWarped = []
xOld = -1
yOld = -1
pntsArray = []


#mouse callback function
def draw_circle(event,x,y,flags,param):
    global contCallbacks
    global xOld, yOld
    global pntsArray
    if(event==cv2.EVENT_LBUTTONDOWN and contCallbacks!=4):
        #cv2.circle(img, center, radius, color(rgb),)
        cv2.circle(imagen,(x,y),6,(255,255,0),-1)
        #cv2.circle(imagen,(x,y),4,(255,255,0),-1)

        contCallbacks=contCallbacks+1
        pntsArray.append( (x,y) )
        if(contCallbacks==4):
            pts = np.array(pntsArray)
            imagenWarped = perspective.four_point_transform( imagen, pts)
            cv2.namedWindow("Imagen corregida")
            cv2.imshow("Imagen corregida", imagenWarped)
        xOld = x
        yOld = y 



def showImagen(pathNameImg):
    global imagen
    print("showImagen("+pathNameImg+")")
    imagen = cv2.imread(pathNameImg) #clone = imagen.copy()
    
    #Creando la ventana anticipadamente
    cv2.namedWindow(namePaintWindow)
    cv2.setMouseCallback(namePaintWindow,draw_circle)

    while(1):
        cv2.imshow(namePaintWindow , imagen)
        if (cv2.waitKey(20) & 0xFF==27):#Detecta la tecla <ESC>
            break
    cv2.destroyAllWindows()
    

print("Iniciando . . . ")
showImagen("images/notecard.png")
print("continuando con el flujo")

