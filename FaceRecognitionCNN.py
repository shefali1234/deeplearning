import cv2                 
import numpy as np        
import os                  
from random import shuffle 
from tqdm import tqdm 
from PIL import Image
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

TRAIN_DIR = './train/'
TEST_DIR = './test/'
IMG_SIZE1 = 640
IMG_SIZE2=480
LR = 1e-3
MODEL_NAME = 'face2-{}-{}.model'.format(LR, '2conv-basic') # just so we remember which saved model is which,

def label_img(img):
    word_label = img.split('.')[-2]
    if word_label == 'm': return [0,1]
    elif word_label == 'f': return [1,0]
    print(word_label)

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE1,IMG_SIZE2))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE1,IMG_SIZE2))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

train_data=create_train_data()

convnet = input_data(shape=[None, IMG_SIZE1, IMG_SIZE2, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')
train = train_data[:100]
test = train_data[50:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE1,IMG_SIZE2,1)
Y = [i[1] for i in train]
#dataset_file = 'face/train/'
#dataset_file2 = 'face/test/'
# Build the preloader array, resize images to 128x128
#from tflearn.data_utils import image_preloader
#X, Y = image_preloader(dataset_file, image_shape=(640, 480),   mode='file', categorical_labels=True,   normalize=True)
#X, Y = image_preloader(dataset_file2, image_shape=(640, 480),   mode='file', categorical_labels=True,   normalize=True)
test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE1,IMG_SIZE2,1)
test_y = [i[1] for i in test]
# model.fit({'input':X},{'targets':Y})
model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), 
     snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
model.save(MODEL_NAME)


test_data = process_test_data()

fig=plt.figure()

for num,data in enumerate(test_data[:16]):
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(4,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE1,IMG_SIZE2,1)
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 1: str_label='m'
    else: str_label='f'
        
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

