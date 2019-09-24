#usr/bin/python
# -*- coding: utf-8 -*-

#THIS version change order of layers in PHOTOROBOT

# Import the required modules
import cv2, os, sys
import numpy as np
from PIL import Image, ImageEnhance, ImageChops, ImageFilter
import PngImagePlugin

DEBUG = True

# constants for database
BEARD_Y = 1
EYES_Y = 1
EYES_X = 1
BROWS_Y = 1
BROWS_X = 1
MOUTH_Y = 1
EARS_Y = 1
NOSE_Y = 1
MOUSTACHE_Y = 1
IMAGE_XSIZE = 200
IMAGE_YSIZE = 200
IMAGE_SIZE = (IMAGE_XSIZE, IMAGE_YSIZE)
GREY_NUM = 8
grey_step = 256/GREY_NUM
RANGE_ZERO = ((0, -1), (0, 0))
RANGE_Y = ((0,-1), (0, 0), (0, 1))
#RANGE_XY = ((x,y) for x in range(-1, 2) for y in range(-1, 2))
#RANGE_XY = ((-1,-1), (-1, 0), (-1, 1), 
#            (0,-1), (0, 0), (0, 1), 
#            (1,-1), (1, 0), (1, 1))
RANGE_XY = ((-1,-1), (-1, 0), (-1, 1), 
            (0,-1), (0, 0), (0, 1), 
            (1,-1), (1, 0), (1, 1))
RANGE_YXclose = ((-1,-1), (-1, 0), (-1, 1))
RANGE_YXfar = ((1,-1), (1, 0), (1, 1))

ANGLE = 0 #rotate original file respect ortogonal
BLACK_COLOR = [0,0,0,]
DELTA_POINT_X = 0
DELTA_POINT_Y0 = 20
DELTA_POINT_Y = 20
subject = 'subject8'
gen_conf = 0
count_conf = 0

# Path to the Yale Dataset
ext = '.png' # 'gif'
path = './animatefaces_4'
dpath = path + '/faces-'+ ext.replace('.','') + '/'
image_path = os.path.join(dpath, subject + ext)

predict_image = Image.open(image_path)
#ifilter = ImageFilter.CONTOUR
#predict_image = predict_image.filter(ifilter)
predict_image = predict_image.rotate(ANGLE)
#xy_ratio = float(predict_image.size[0])/float(predict_image.size[1])
#if xy_ratio < 1:
#   predict_image = predict_image.resize((int(IMAGE_XSIZE*xy_ratio), IMAGE_YSIZE))
#else:
#   predict_image = predict_image.resize((IMAGE_XSIZE, int(IMAGE_YSIZE/xy_ratio)))

predict_image_pil = predict_image.convert('L')
predict_image_rgb = predict_image.convert('RGB')

predict_image_rgb.save('FACE.png', 'PNG')

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
# For face recognition we will the the LBPH Face Recognizer 
recognizer = cv2.createLBPHFaceRecognizer(1,8,5,5,200)    


# paths of phothorobots
#predict_image = np.array(predict_image_pil, 'uint8')
#predict_faces = faceCascade.detectMultiScale(predict_image, scaleFactor=1.2, minNeighbors=3, minSize=(70, 70), maxSize=(100,100))
print ('hello')
#print (predict_faces)
#predict_image_5plus = predict_image_pil.rotate(5)
image_show = np.array(predict_image_rgb, copy=True)
#cv2.imshow("Original", image_show)

AVERAGE_FACE = Image.new('L',IMAGE_SIZE)
# main table
# structure:
# usage in building | usage in composition | filename | monochrome matrix | X,Y-place
# monochrome matrix: [.,.,x,y] - use from x to y if 256 used | color of element
image_parts = {
    'hair': [None, True, 'hair1',[0,0,31,31],[0,0],BLACK_COLOR,],
#    'hface': [None, False, 'average',[0,0,0,0],[0,0],BLACK_COLOR,],        
    'beard': [None, True, 'beard1',[0,0,63,63],[0,BEARD_Y],BLACK_COLOR,],
#    'bface': [None, False, 'bface-1',[0,0,0,0],[0,0],BLACK_COLOR,],

    'eyes': ['F', True, 'eyes1', [0,0,31,31],[EYES_X,EYES_Y],BLACK_COLOR,], # black contours
    'eyebrows': ['XY', False, 'eyebrows1',[0,0,127,63],[BROWS_X,BROWS_X],BLACK_COLOR,], # black
    'mouth': ['F', True, 'mouth1',[0,0,95,63],[0,MOUTH_Y],BLACK_COLOR,],
#    'moustache':['Y', False, 'moustache0',[0,0,224,128],[0,MOUSTACHE_Y],BLACK_COLOR,], # black
    'ears': [None, False, 'ears3',[0,0,95,63],[0,EARS_Y],BLACK_COLOR,], # dont move ears
    'nose': ['Y', False, 'nose1',[0,0,95,63],[0,NOSE_Y],BLACK_COLOR,], #black contours
    }

############################################3

def shift_y(image, delta):
    xsize, ysize = image.size
    if delta == 0:
        return image
    elif delta > 0:
        crop_box = (0, 0, xsize, ysize-delta)
        paste_box = (0, delta)
    else:
        crop_box = (0, -delta, xsize, ysize)
        paste_box = (0, 0)
    region = image.crop(crop_box)    
    image.paste(region, paste_box)
    return image

def shift_x(image, delta):
    xsize, ysize = image.size
    xsize2 = xsize / 2
    if delta == 0:
        return image
    elif delta > 0: #far
        crop_box_left = (delta, 0, xsize2, ysize)
        paste_box_left = (0, 0)
    else: # close
        crop_box_left = (0, 0, xsize2+delta, ysize)
        paste_box_left = (-delta, 0)
    crop_box_right = (xsize2, 0, xsize-delta, ysize)
    paste_box_right = (xsize2+delta, 0)
    region = image.crop(crop_box_left)
    image.paste(region, paste_box_left)
    region = image.crop(crop_box_right)
    image.paste(region, paste_box_right)
    return image

######################################3

def shift (shift_image, mode, deltax, deltay):
    if deltax == 0 and deltay == 0:
        return shift_image
    if mode == 'Y':
        image_layer = shift_y(shift_image, deltay)
    elif mode == 'X':
        image_layer = shift_x(shift_image, deltax)
    else:                
        if deltax == 0:
            image_layer = shift_y(shift_image, deltay)
        elif deltay == 0:
            image_layer = shift_x(shift_image, deltax)
        else:                    
            image_layer = shift_y(shift_image, deltay)
            shift_image = shift_x(image_layer, deltax)
            image_layer = shift_image
    return image_layer
################

def get_image_rgb(x, y, w, h):
    #crop_box_center = (x+w//3, y+h//3, x+w*2//3, y+h*2//3)
    crop_box_center = (x, y, x+w, y+h)
    region = predict_image_rgb.crop(crop_box_center)
    #print(crop_box_center)
    image_new = Image.new('RGB', (w, h))    
    image_new.paste(region, (0,0))
    image_data = list(image_new.getdata())   
    #print (image_data)
    scale = []
    a = np.array(image_data)
    b = a.mean(axis=0)
    color = b.astype(int)
    dif = 255-color.max()
    scale = (color, color+dif/2, color+dif)
    #image_new.save('FACE'+str(dif+color.min()+color.max())+'.png', 'PNG')
    return scale

#############################################
def collect_layers():
    #im1 = Image.new('L',IMAGE_SIZE)
    #im1 = Image.open(dbfile('hair')).convert('L')
    im1 = Image.open(dpath + 'hair/empty' + ext).convert('L')

    for row in image_parts:
#        print(image_parts[row][1])
        if image_parts[row][1] == True:            
            im2 = Image.open(dbfile(row)).convert('L')
#            im2 = im2.point(lambda i: 256-(256-i)*(256-image_parts[row][3][3])/grey_step/GREY_NUM)
            im2 = im2.point(lambda i: 255-(255-i)*(255-image_parts[row][3][3])/grey_step/GREY_NUM)
            im1 = Image.composite(im1, im2, im2)
            im1.save('./results/rgb/'+row+ext, 'PNG')
    return im1

############################################
def get_face_part(table, layers):
    global faceCascade
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
#    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.startswith('subject16-') ]
    # images will contains face images

    #this is only for draft search
    key_name = layers['key_type']
    #epath = dpath + 'eyes'
    #range_list = [os.path.join(epath, f) for f in os.listdir(epath) if f.startswith('eyes-')]

    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    images_alt = []
    faces = []
    nbr = 0

    #this is int type division !
#    start_grey = (256-table[key_name][3][2]) // grey_step
#    end_grey = (256-table[key_name][3][3]) // grey_step
    start_grey = (255-table[key_name][3][2]) // grey_step
    end_grey = (255-table[key_name][3][3]) // grey_step

    print(start_grey, end_grey, grey_step)

#    print(table['faces'][2])
    input_image_sum = collect_layers()#AVERAGE_FACE
    #this layer should be disabled if 'F'
    if layers['axis'] == 'F':
        table[key_name][1] = False
    #Compose all layers
    #print(table.values())
    for layer in table:
    #for layer in table.values():
        #print(layer, table[layer])
        xp = table[layer][4][0]
        yp = table[layer][4][1]            
        # this layer must be X,Y movable and enabled for consideration
        if table[layer][1] == True:         
#            color = (256-table[layer][3][3]) / grey_step
            color = (255-table[layer][3][3]) / grey_step            
            image_layer = Image.open(dbfile(layer)).convert('L')
            if table[layer][0] != None:
                shift_image = image_layer
                image_layer = shift(shift_image, table[layer][0], table[layer][3][0]*xp, table[layer][3][1]*yp)

#            image2 = image_layer.point(lambda i: 256-(256-i)*color/GREY_NUM)
            image2 = image_layer.point(lambda i: 255-(255-i)*color/GREY_NUM)
            #image2 = image_layer
            input_image_sum = Image.composite(input_image_sum, image2, image2)
            
    if layers['range_type'] == None:
#        range_list = [os.path.join(dpath, f) for f in os.listdir(dpath) if f.startswith(key_name+'-')]     
        epath = dpath + key_name
        range_list = [os.path.join(epath, f) for f in os.listdir(epath) if f.startswith(key_name)]#+'-')]
        int_range = [int(f.replace(key_name,'').replace(ext,'')) for f in os.listdir(epath) if f.startswith(key_name)]#+'-')]        
        len_range = len(range_list)        
        max_len = max(int_range)        
        #print(int_range, max_len)
    else:
        range_list = layers['range_type']
        len_range = len(range_list)        
        max_len = len_range
    k = 0
    faces_num = 0
    image1 = input_image_sum

    #change some elements of layer and save training set  


    #if layers['axis'] == 'XY':
    #    len_range = 9
    #elif layers['axis'] == 'Y' or layers['axis'] == 'X':
    #    len_range = 3
    #else:
    #    len_range = len(range_list)
    
    # data of input
    data_n = table[key_name]
    xp = table[key_name][4][0]
    yp = table[key_name][4][1]

    for color in range(start_grey, end_grey+1):#GREY_NUM+1):
        for n in range_list:
            #print (n)
            # Read the image and convert to greyscale
            #print(image_path)
            # Combine faces and eye-mouths
            #range is files
            if layers['axis'] == None:                
                image2 = Image.open(n).convert('L')
            # Get the label of the image            
                nbr = int(os.path.split(n)[1].split(".")[0].replace(key_name, ""))#+'-', ""))
                #print(nbr)
                file_layer = (os.path.split(n)[1].split(".")[0])
            #this part is fixed
            elif layers['axis'] == 'F':
                #print ('FIXED LAYER!', data_n[3])
                shift_image = Image.open(n).convert('L')
                nbr = int(os.path.split(n)[1].split(".")[0].replace(key_name, ""))#+'-', ""))
                file_layer = (os.path.split(n)[1].split(".")[0])               
                image2 = shift(shift_image, layers['axis'], data_n[3][0]*xp, data_n[3][1]*yp)            
            #range is coordinates            
            else:                               
                image2 = Image.open(dbfile(key_name)).convert('L')                
                image2 = shift(image2, layers['axis'], n[0]*xp, n[1]*yp)

            # # Get the label of the image
                k += 1
                nbr = k
#            image_out = image2.point(lambda i: 256 - (256-i)* color/GREY_NUM)
            image_out = image2.point(lambda i: 255 - (255-i)* color/GREY_NUM)  

            image_pil = Image.composite(image_out, image1, image1)
            image = np.array(image_pil, 'uint8')
            cv2.imshow("Watermarked", image)
            cv2.waitKey(1)              

#            image = np.array(predict_image_in, 'uint8')
#            cv2.imshow("Input image", image)
#            cv2.waitKey(100)

            #face = faceCascade.detectMultiScale(image, scaleFactor=1.15, minNeighbors=3, minSize=(80,80), maxSize=(160,160))
            face = faceCascade.detectMultiScale(image, scaleFactor=1.15, minNeighbors=3, minSize=(70,70), maxSize=(160,160))

            #        print(nbr)           
            # If face is detected, append the face to images and the label to labels
            if layers['axis'] == None or layers['axis'] == 'F':
#                data_n = ['F', True, file_layer, [data_n[3][0], data_n[3][1], (256-color*grey_step), (256-color*grey_step)],[xp,yp],BLACK_COLOR]            	
                data_n = ['F', True, file_layer, [data_n[3][0], data_n[3][1], (255-color*grey_step), (255-color*grey_step)],[xp,yp],BLACK_COLOR]
#                print(faces,data_n)                
            else:
#                data_n = ['F', True, data_n[2], [n[0], n[1], (256-color*grey_step), (256-color*grey_step)],[xp,yp],BLACK_COLOR]                           	
                data_n = ['F', True, data_n[2], [n[0], n[1], (255-color*grey_step), (255-color*grey_step)],[xp,yp],BLACK_COLOR]               
            #print(faces,data_n)

            if face != ():    
                #print(face[0][0]+face[0][2])            
                x1 = face[0][0]
                y1 = face[0][1]-2*DELTA_POINT_Y0
                x2 = face[0][0]+face[0][2]
                y2 = face[0][1]+face[0][3]+DELTA_POINT_Y0
                if y1>=0:                	                
	                images.append(image[y1:y2, x1:x2])#image[y-2*DELTA_POINT_Y: y + h+DELTA_POINT_Y, x: x + w])
	                labels.append(nbr+max_len*color)
	                images_alt.append(data_n) # open the most correlated file
	                #faces.append(face) # append the most correlated
	                faces_num += 1          
	                cv2.imshow("Adding faces to traning set...", image[y1: y2, x1: x2])

	 #           cv2.waitKey(1)                
    
    # return the images list and labels list
    #cv2.waitKey(50)
    return images, labels, images_alt, faces, faces_num

###########################################
# input: sum of images and method of working on layers
def image_prediction(image_sum, layers):
    global predict_faces, predict_image
    global gen_conf, count_conf

    images = []
    # labels will contains the label that is assigned to the image
    labels = []

    images_alt = []
    aver_color = []

    key_name = layers['key_type']
    #fill new data after working on
    images, labels, images_alt, faces, faces_num = get_face_part(image_sum, layers)
    #assert faces_num == 0
    # color of the face
    #print(key_name)
    if faces_num == 0:
        #gen_conf += conf
        conf = 1000
        if DEBUG:            
            print ('EMPTY LAYER!')
            #image_sum[key_name][5] = image_sum['hair'][5]
        else:
            change_image()
        return conf
    # trainig based on collected information    
    recognizer.train(images, np.array(labels))

    # first step: fill predict_faces and predict_image
#    if layers['key_type'] == 'hair' and layers['axis'] == None:
#        predict_image = np.array(predict_image_pil, 'uint8')
#        predict_faces = faceCascade.detectMultiScale(predict_image)

    #we have square of face:
    #and thedn choose best from
    #nbr_predicted - number, conf - measure of prediction
    nbr_predicted = 0
    for (x, y, w, h) in predict_faces:
        nbr_predicted, conf = recognizer.predict(predict_image[y-2*DELTA_POINT_Y: y + h+DELTA_POINT_Y, x: x + w])        
        print "{} Recognized {}".format(nbr_predicted, conf)
        if nbr_predicted != -1:
            gen_conf += conf
            count_conf += 1                

    if nbr_predicted == -1:
        conf = 1000
        return conf
    lab_pr = labels.index(nbr_predicted)
    if len(labels) > 0:
        image_sum[key_name] = images_alt[lab_pr]
        print(lab_pr, faces_num)
        return conf
    #elif len(labels) > 1:
    #    print ("Too much faces. Input file with one face only.")
    #    return None
    else:
        print ("Error! Check input file.")
        return None

###########################   
# for sepia function     
def make_linear_ramp(white):
    # putpalette expects [r,g,b,r,g,b,...]
    ramp = []
    r, g, b = white
    for i in range(255):
        ramp.extend((r*i/255, g*i/255, b*i/255))
    return ramp

def test_function():
    im1 = Image.open('test1.png').convert('L')
    im2 = Image.open('test2.png').convert('L')
    imsum = Image.new('L',im1.size)
    imsum = Image.composite(im1, im2, im2)

    imsum.save('testsum.png','PNG')

    im1 = Image.open('test1.png').convert('RGB')
    im2 = Image.open('test2.png').convert('RGB')
#    imrgb_mask = Image.open('test_mask.png').convert('L')
    imrgb_mask = Image.open('test1.png').convert('L')
#    imrgb_mask = Image.open('test2.png').convert('L')        

    imsum = Image.new('RGBA',im1.size)    
    imsum = Image.composite(im1, im1, imrgb_mask)

    imsum.save('testsum_rgb.png','PNG')

    return 

# colorize image with given color
def grey(white):
    r, g, b = white
    color_grey = (r*299+g*587+b*114)/1000                
    return color_grey

# show avatar by composing and colorizing all collected layers
def show_avatar(table, time):
    global predict_faces
    x, y, h, w = predict_faces[0]

    skin_color = BLACK_COLOR
    mouth_color = BLACK_COLOR
    hair_color = BLACK_COLOR
    beard_color = BLACK_COLOR
    ears_color = BLACK_COLOR

    input_image_sum = AVERAGE_FACE
    input_image_rgb = AVERAGE_FACE.convert('RGBA')
    input_image_rgb = input_image_rgb.point(lambda i: 255 - i)

    scale = get_image_rgb(x+w//3, y+h//3, w//3, h//3)
    ears_color = scale[1]
    skin_color = scale[2]

    # and also!
    #table['hface'][5] = skin_color
    #table['bface'][5] = skin_color
    table['ears'][5] = ears_color
    print('skin_color', skin_color)

    #rgb_mask = Image.new('L', AVERAGE_FACE.size)
    # compose hair and beard face
#    image_layer_h = Image.open(dbfile('hface')).convert('L')    
#    image_layer_b = Image.open(dbfile('bface')).convert('L')    

    #image_layer_p = Image.composite(image_layer_h, image_layer_b, image_layer_b)
    #image_layer_p = Image.open(dpath + 'hair/average' + ext).convert('L')

    skin_grey = grey(BLACK_COLOR)
    print(skin_grey)
#    image2 = image_layer_p.point(lambda i: 255 if i == 255 else skin_grey)

#    sepia = make_linear_ramp(BLACK_COLOR)

#    image2_rgb = image2.copy()
#    image2_rgb.putpalette(sepia)

#    input_image_rgb.save('./results/rgb/'+'NOFACE0'+str(ANGLE)+ext, 'PNG')    

#    rgb_mask = image_layer_p
            
    #image2_rgb = image2_rgb.convert('RGB')
    #input_image_sum = Image.composite(input_image_sum, image2, image2)
    #input_image_rgb = Image.composite(input_image_rgb, image2_rgb, rgb_mask)#image_layer_rgb)            

############### all other layers
    #Compose all layers
    #print(table.values())
    i = 0
    for layer in table: #table.values():
        if (table[layer][1] == True):# and (layer != 'nose'):# and (layer != 'hair'):
            xp = table[layer][4][0]
            yp = table[layer][4][1]
            if layer == 'hair':
                scale = get_image_rgb(x+w//6, y, w//6, h//16)
                hair_color = scale[2] # more light
                table[layer][5] = hair_color
                #print('hair_color', hair_color)
            elif layer == 'beard':
                scale = get_image_rgb(x+w//4, y+7*h//8, w//8, h//8)
                beard_color = scale[1] # little light
                table[layer][5] = beard_color            
               #print('beard_color', beard_color)
            elif layer == 'mouth':
                scale = get_image_rgb(x+w*3//8, y+7*h//8, w//4, h//8)        
                mouth_color = scale[1] #little light
                table[layer][5] = mouth_color
                #print('mouth_color', mouth_color)
            elif layer == 'nose':
                scale = get_image_rgb(x+w*3//8, y+5*h//8, w//4, h//8)        
                nose_color = scale[1] #little light
                table[layer][5] = nose_color
                #print('nose_color', nose_color)
            elif layer == 'eyes':
                scale = get_image_rgb(x+w*2//8, y+2*h//8, w//8, h//8)        
                eyes_color = scale[1] #little light
                table[layer][5] = eyes_color
                #print('eyes_color', eyes_color)

            image_layer = Image.open(dbfile(layer)).convert('L')            
            image_layer = shift(image_layer, table[layer][0], table[layer][3][0]*xp, table[layer][3][1]*yp)
            if (1):#max(table[layer][5]) != 0: # [3][3] == 0
            	print('1st')
                image2 = image_layer.point(lambda i: 255 if i == 255 else grey(table[layer][5]))                
                sepia = make_linear_ramp(table[layer][5])
# make sepia ramp (tweak color as necessary)
                image2_rgb = image2.copy()
                image2_rgb.putpalette(sepia)
                rgb_mask = image_layer.convert('L')
            else: 
                print('2nd')        
#                color = (256-table[layer][3][3]) / grey_step               
#                image2 = image_layer.point(lambda i: 256-(256-i)*color/GREY_NUM)                
                color = (255-table[layer][3][3]) / grey_step               
                image2 = image_layer.point(lambda i: 255-(255-i)*color/GREY_NUM)
                image2_rgb = image2.copy()
                rgb_mask = image2.convert('L')
  
            image2_rgb = image2_rgb.convert('RGB')#('RGB')
            input_image_sum = Image.composite(input_image_sum, image2, image2)
            input_image_rgb = Image.composite(input_image_rgb, image2_rgb, rgb_mask)#image_layer_rgb)            
    	    #input_image_rgb.save('./results/rgb/'+'FACE'+str(layer)+str(ANGLE)+ext, 'PNG')
            #input_image_rgb.save('face'+str(i)+'.png', 'PNG')
            i += 1
        print(layer, table[layer])    

    image = np.array(input_image_sum, 'uint8')
    image_rgb = np.array(input_image_rgb, 'uint8')
    cv2.imshow("Avatar", image_rgb)
    input_image_sum.save('./results/bw/'+'FACE'+str(ANGLE)+ext, 'PNG')
    input_image_rgb.save('./results/rgb/'+'FACE'+str(ANGLE)+ext, 'PNG')
# for comfortable comparison
#    input_image_rgb.save(dpath+subject+'_'+str(ANGLE)+ext, 'PNG')
    cv2.waitKey(time)
    return
#########################
def dbfile(name):
    return dpath + name + '/' + image_parts[name][2] + ext
#########################
def change_image(image):
    print('Your image is not recognized. Try to rotate it or enlarge it')
    return
#########################
#MAIN
#test_function()
def sequence(name, layer_type):

    if (layer_type == 'form'):
        layers = dict(axis='F', range_type=None, key_type = name)
    elif (name == 'hair') or (name == 'beard'):
        layers = dict(axis='Y', range_type=RANGE_ZERO, key_type = name)
    elif (name == 'ears') or (name == 'mouth') or (name == 'nose') or (name == 'moustache'):
        layers = dict(axis='Y', range_type=RANGE_Y, key_type = name)
    else: #(name == 'eyes'):
        layers = dict(axis='XY', range_type=RANGE_XY, key_type = name)
#    elif image_parts['eyes'][3][0] == 0:
#        layers = dict(axis='XY', range_type=RANGE_YXclose, key_type=name)
#    elif image_parts['eyes'][3][0] == -2:
#        layers = dict(axis='XY', range_type=RANGE_YXfar, key_type=name)
#    else:
#        layers = dict(axis='XY', range_type=RANGE_Y, key_type=name)    

    image_parts[name][1] = False
    image_prediction(image_parts, layers)
    image_parts[name][1] = True
    print(image_parts[name][2])
    return

def demo_recognize():
    global AVERAGE_FACE
    global predict_image
    global predict_faces

    predict_image = np.array(predict_image_pil, 'uint8')
    predict_faces = faceCascade.detectMultiScale(predict_image)

    layer_seq = [
        ['mouth','form'],
        ['eyes', 'form'],
#        ['beard','form'],
#        ['hair','form'],

        ['mouth','place'],
        ['eyes', 'place'],        
        ['nose', 'form'],  
        ['ears','form'],        #test mode

        ['beard','form'],
        ['hair','form'],
        
        #['beard','place'],        #test mode
        #['hair','place'],        #test mode
        ]

    for pair in layer_seq:
        sequence(pair[0], pair[1])

    show_avatar(image_parts, 5000)
    cv2.destroyAllWindows()

    return

#for i in range (0, 10)
demo_recognize()
print (gen_conf/count_conf)