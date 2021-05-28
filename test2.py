import streamlit as st
from PIL import Image, ImageEnhance
import cv2 
import numpy as np
import scipy
import seaborn as sns
import copy
from scipy.interpolate import UnivariateSpline
from streamlit_lottie import st_lottie
import requests

st.set_page_config(layout="wide")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_book = load_lottieurl('https://assets6.lottiefiles.com/datafiles/9jPPC5ogUyD6oQq/data.json')
st_lottie(lottie_book, speed=1, height=200, key="initial")

sns.set_style('darkgrid')
row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.beta_columns(
    (.1, 2, .2, 1, .1))

row0_1.title('Image Filtering Using OpenCV')

with row0_2:
    st.write('')

row0_2.subheader(
    'A Web App by [Kritanjali Jain](https://www.linkedin.com/in/kritanjali-jain-009180191/)')

def test2():
    
    st.write('')
    row1_spacer1, row1_1, row1_spacer2 = st.beta_columns((.1, 3.2, .1))

    with row1_1:
        st.markdown("Hey there! Welcome to Image Filtering App. This app demonstrates OpenCV image preprocessing techniques. These are often used by machine learning developers to preprocess their images for Computer Vision models. This app never stores your image data.One last tip, if you're on a mobile device, switch over to landscape for viewing ease. Give it a go!")

    col1, col2 = st.beta_columns(2)

    with col1:
        st.subheader('Original Image')
        st.markdown("Select a filter from the dropdown box given alongside, adjust the value (if available) for that filter and click on 'Apply' to see the changes. You can apply emboss, blur, adjust the brightness, contrast, warmth, coolness, shrapness, threshold, detect edges, grayscale.")
        st.markdown("**Great! Now let's apply some cool filters to the image.**")
        image = Image.open('test.jpg')
        st.image(image, use_column_width=True)
 

    with col2:
            st.write('')
        
            filterchoice = st.selectbox('Select filter', ['Threshold', 'Brightness', 'Blur','Emboss','Warmth','Coolness','Sharpen','Grayscale','Detect Edges','Contrast'], key=1)
            
            image = cv2.imread('test.jpg') #for loading file
            
            def spreadLookupTable(x, y):
                spline = UnivariateSpline(x, y)
                return spline(range(256))
            
            
            if filterchoice == 'Threshold':             
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                x = st.slider('Adjust Threshold value',min_value = 50,max_value = 150)  
                ret,thresh1 = cv2.threshold(image,x,255,cv2.THRESH_BINARY)
                thresh1 = thresh1.astype(np.float64)
                st.image(thresh1, use_column_width=True,clamp = True)
            
            elif filterchoice == 'Brightness':
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    x = st.slider('Adjust Brightness',10, 150, value = 30)
                    def brightnessControl(image, level):
                        return cv2.convertScaleAbs(image, beta=level)
                    image_bright = brightnessControl(copy.deepcopy(image), x)
                    cv2.imwrite("results/bright.jpg", image_bright)
                    st.image(image_bright, use_column_width=True,clamp = True)
            
            elif filterchoice == 'Blur':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                x = st.slider('Adjust blur value', min_value = 15, max_value = 100, value = 51, step=2)
                
                image_blur = cv2.GaussianBlur(image, (x, x), 0)
                cv2.imwrite("results/blur.jpg", image_blur)
                st.image(image_blur, use_column_width=True,clamp = True)
                
            
            elif filterchoice == 'Contrast':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                def contrast(image):
                    kernel = np.array([[0.272, 0.534, 0.131],
                                [0.349, 0.686, 0.168],
                                [0.393, 0.769, 0.189]])
                    return cv2.filter2D(image, -1, kernel)
                image = contrast(copy.deepcopy(image))
                cv2.imwrite("results/contrast.jpg", image)
                st.markdown("Sorry! The values for this filter are not currently user-adjustable.")
                st.write("")
                st.image(image, use_column_width=True,clamp = True)

            elif filterchoice == 'Emboss':
                height, width = image.shape[:2]
                y = np.ones((height, width), np.uint8) * 128
                kernel = np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                output = cv2.add(cv2.filter2D(gray, -1, kernel), y)
                
                cv2.imwrite("results/emboss.jpg", output)
                st.markdown("Sorry! The values for this filter are not currently user-adjustable.")
                st.write("")
                st.image(output, use_column_width=True,clamp = True)
            
            elif filterchoice == 'Sharpen':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                image = cv2.filter2D(image, -1, kernel)
                cv2.imwrite("results/sharpen.jpg", image)
                st.markdown("Sorry! The values for this filter are not currently user-adjustable.")
                st.write("")
                st.image(image, use_column_width=True,clamp = True)

            elif filterchoice == 'Grayscale':
                image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                cv2.imwrite("results/gray.jpg", image)
                st.markdown("Sorry! The values for this filter are not currently user-adjustable.")
                st.write("")
                st.image(image, use_column_width=True,clamp = True)
            elif filterchoice == 'Warmth':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                def warmImage(image):
                    increaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 80, 160, 256])
                    decreaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 50, 100, 256])
                    red_channel, green_channel, blue_channel = cv2.split(image)
                    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
                    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
                    return cv2.merge((red_channel, green_channel, blue_channel))
                
                image = warmImage(copy.deepcopy(image))
                cv2.imwrite("results/warm.jpg", image)
                st.markdown("Sorry! The values for this filter are not currently user-adjustable.")
                st.write("")
                st.image(image, use_column_width=True,clamp = True)

            elif filterchoice == 'Coolness':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
                def coldImage(image):
                    increaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 80, 160, 256])
                    decreaseLookupTable = spreadLookupTable([0, 64, 128, 256], [0, 50, 100, 256])
                    red_channel, green_channel, blue_channel = cv2.split(image)
                    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
                    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
                    return cv2.merge((red_channel, green_channel, blue_channel))  

                image = coldImage(copy.deepcopy(image))       
                cv2.imwrite("results/cold.jpg", image)
                st.markdown("Sorry! The values for this filter are not currently user-adjustable.")
                st.write("")
                st.image(image, use_column_width=True,clamp = True)
           
            elif filterchoice == 'Detect Edges':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                edges = cv2.Canny(image,50,300)
                cv2.imwrite('results/edges.jpg',edges)
                st.markdown("Sorry! The values for this filter are not currently user-adjustable.")
                st.write("")
                st.image(edges,use_column_width=True,clamp=True)
    st.markdown('***')
    st.header("Let's sum up the filters applied and the resulting images!")
    row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.beta_columns((.1, 1, .1, 1, .1))
    
    with row3_1:
        st.subheader("Threshold")
        st.image('results/thresh.jpg', use_column_width=True,clamp = True)
        st.markdown("In thresholding, each pixel value is compared with the threshold value. If the pixel value is smaller than the threshold, it is set to 0, otherwise, it is set to a maximum value. The function cv.threshold was used to apply the thresholding. Basic thresholding was done by using the type cv.THRESH_BINARY")
    
    with row3_2:
        st.subheader("Brightness")
      
        st.image('results/bright1.jpg', use_column_width=True,clamp = True)
        
        st.markdown("In OpenCV, changing the brightness of an image is a very basic task to perform. By changing the image brightness, it is meant to change the value of each and every image pixel. This change can be done by either increasing or decreasing the pixel values of the image, by any constant.")
    

    st.write('')

    row4_space1, row4_1, row4_space2, row4_2, row4_space3 = st.beta_columns(
        (.1, 1, .1, 1, .1))

    with row4_1:
        st.subheader("Blur")
        st.image('results/blur1.jpg', use_column_width=True,clamp = True)
        st.markdown("To make an image blurry, the GaussianBlur() method of OpenCV is used. The GaussianBlur() uses the Gaussian kernel. The height and width of the kernel should be a positive and an odd number.")


    with row4_2:
        st.subheader("Sharpen")
        st.image('results/sharpen1.jpg', use_column_width=True,clamp = True)
        st.markdown("A predefined kernel is used to sharpen the details on the picture. The filter2D method from OpenCV library is used which will perform the convolution.")
    st.write('')
    
    row5_space1, row5_1, row5_space2, row5_2, row5_space3 = st.beta_columns(
        (.1, 1, .1, 1, .1))

    with row5_1:
        st.subheader("Increase temperature")
        st.image('results/warm1.jpg', use_column_width=True,clamp = True)
        st.markdown("For obtaining a warm image, the values of the red channel have been increased and the values of the blue channel have been decreased for all the pixels in the image.")


    with row5_2:
        st.subheader("Decrease temperature")
        st.image('results/cold1.jpg', use_column_width=True,clamp = True)
        st.markdown("For obtaining a cold image, the opposite has been done: increased the values for the blue channel and decreased the values for the red channel. The green channel remains untouched.")

    st.write('')
    row6_space1, row6_1, row6_space2, row6_2, row6_space3 = st.beta_columns(
        (.1, 1, .1, 1, .1))

    with row6_1:
        st.subheader("Emboss")
        st.image('results/emboss.jpg', use_column_width=True,clamp = True)
        

        st.markdown("Image embossing is a computer graphics technique in which each pixel of an image is replaced either by a highlight or a shadow, depending on light/dark boundaries on the original image. Low contrast areas are replaced by a gray background.A kernel was created and convolution was applied using cv2.conv2D and a value of 128 was added to all pixels.")

    with row6_2:
        st.subheader("Contrast")
        
        st.image('results/contrast1.jpg', use_column_width=True,clamp = True)

        st.markdown("Contrast is the difference in luminance or color that makes an object distinguishable from other objects within the same field of view. A kernel was created and convolution was applied using cv2.filter2D.")

    
    st.write('')
    row7_space1, row7_1, row7_space2, row7_2, row7_space3 = st.beta_columns(
        (.1, 1, .1, 1, .1))

    with row7_1:
        st.subheader('Grayscale')
    
        st.image('results/gray.jpg', use_column_width=True,clamp = True)  
        st.markdown("To convert a color image into a grayscale image, the cvtColor() method of the cv2 module is used which takes the original image and the COLOR_BGR2GRAY attribute as an argument.") 

    with row7_2:
        st.subheader("Edge Detection")
        
    
        st.image('results/edges.jpg',use_column_width=True,clamp=True)
       

        st.markdown("To detect the edges in an image, the Canny() method of cv2 is used which implements the Canny edge detector. The Canny edge detector is also known as the optimal detector.")

    st.write('')

    row8_spacer1, row8_1, row8_spacer2 = st.beta_columns((.1, 3.2, .1))

    with row8_1:
        st.header("And that was all! ")
        st.markdown("These are a few cool image processing techniques OpenCV offers. ")
        st.markdown('***')
        st.markdown(
            "Thanks for using this web-app! This web-app was made as a course end project for Stanford's Code in Place CS106A Spring 2021.")
            
if __name__ == "__main__":
    test2()