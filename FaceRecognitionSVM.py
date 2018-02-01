import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
from PIL import Image
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.svm import SVC

TRAIN_DIR = 'face/train/'
TEST_DIR = 'face/test/'
IMG_SIZE1 = 640
IMG_SIZE2=480
LR = 1e-3

MODEL_NAME = 'face2-{}-{}.model'.format(LR, 'SVM-basic')
cascadeLocation = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadeLocation)
def label_img(img):
    word_label = img.split('.')[-2]
    if word_label == 'm': return [0,1]
    elif word_label == 'f': return [1,0]
    print(word_label)

def create_train_data():
    training_data = []
    #images = []
    #labels = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        #image_pil = Image.open(img).convert('L')
        #img = np.array(image_pil, 'uint8')
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE1,IMG_SIZE2))
        faces = faceCascade.detectMultiScale(img)
        for (x,y,w,h) in faces:
            #training_data.append[np.array(img),np.array(label)]
            #cv2.imshow("Reading Faces ",img[y,x])
            training_data.append([np.array(img[y:y+IMG_SIZE2,x:x+IMG_SIZE1]),np.array(label)])
            cv2.imshow("Reading Faces ",img[y:y+IMG_SIZE2,x:x+IMG_SIZE1])
            cv2.waitKey(50)
    return images,labels,IMG_SIZE1,IMG_SIZE2
    #return training_data

images,labels,IMG_SIZE1,IMG_SIZE2=create_train_data()
def process_test_data():
    testing_data = []
    #images = []
    IMG_SIZE1=640
    IMG_SIZE2=480

    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE1,IMG_SIZE2))
        faces = faceCascade.detectMultiScale(img)
        for (x,y,w,h) in faces:
             testing_data.append([np.array(img[y:y+IMG_SIZE2,x:x+IMG_SIZE1]), img_num])
            #np.array(images.append(img[y:y+IMG_SIZE2,x:x+IMG_SIZE1]))
            #testing_data.append([np.array(img), img_num])
            #cv2.imshow("Reading Faces ",img[y:y+IMG_SIZE2,x:x+IMG_SIZE1])
            #cv2.waitKey(50)
    return testing_data

test_data=process_test_data()
n_components = 10
cv2.destroyAllWindows()
pca = RandomizedPCA(n_components=n_components, whiten=True)

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'),param_grid)

pca = pca.fit(test_data)

transformed = pca.transform(test_data)
clf.fit(transformed,labels)

image_paths = [os.path.join(directory, filename) for filename in os.listdir(directory)]
for image_path in image_paths:
    pred_image_pil = Image.open(image_path).convert('L')
    pred_image = np.array(pred_image_pil, 'uint8')
    faces = faceCascade.detectMultiScale(pred_image)
    for (x,y,w,h) in faces:
        X_test = pca.transform(np.array(pred_image[y:y+col,x:x+row]).flatten())
        mynbr = clf.predict(X_test)
nbr_act = int(os.path.split(image_path)[1].split('.')[-2])
