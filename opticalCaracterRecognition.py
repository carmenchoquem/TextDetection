# coding:utf-8
# https://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/
# pip install pillow
# pip install pytesseract
# pip install tesseract-ocr -> sigue sin funcionar | no encuentra <tesseract.exe>
# pip install tesseract -> sigue sin funcionar
# https://github.com/UB-Mannheim/tesseract/wiki -> instalador de teseract.exe en windows
# Don't forget set: cd ./VisionArtificial
#
# Developer: Deiner Zapata Silva


from PIL import Image
#from tesseract import image_to_string
import pytesseract
import cv2
import os
import time

# Argument
imagePath = "images/imgText_02.png"  # path to input to image to be OCR
preprocess= "thresh" # type of preprocessing to be done: thresh & blur

# load the example and convert it to grayscale
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# check to see if we should apply thresholding to preprocess the image
if (preprocess=="thresh"):
    # https://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# make a check to see if median blurring should be done to remove noise
elif (preprocess=="blur"):
    gray = cv2.medianBlur(gray,3)

# write the grayscale image to disk as a temporary file so we can apply OCR to it
####filename = r"C:/Users/LENOVO/Documents/PythonCode/VisionArtificial/21940.png" #
filename = "{}.png".format( os.getpid() )
cv2.imwrite(filename, gray) # create temporary file

# load the image as a PIL/Pillow iamge, apply OCR, and then delete the temporary file
try:
    img = Image.open(filename)
    pytesseract.pytesseract.tesseract_cmd = r"C:/Tesseract-OCR/tesseract.exe"
    text = pytesseract.image_to_string( img )
    print("-------------------------------------------------------------------")
    print("   *  *  T E X T    D E T E C T E D  *  *")
    print("-------------------------------------------------------------------")
    print(text)
    print("-------------------------------------------------------------------")
    #show the output images
    cv2.imshow("Image", image)
    cv2.imshow("Output", gray)
    print("Press <ESC> to close windows")
    while(1):
        if ( cv2.waitKey(20) & 0xFF==27):#Detecta la tecla <ESC>
            break
    cv2.destroyAllWindows()
except IOError as e:
    print("I/O ERROR : "+ str(e))
except:
    print("Oops! Error to read imagen.")
    
os.remove( filename )
