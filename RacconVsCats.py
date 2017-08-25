"""
 Script for downloading images from Google Image search using selenium,  BeatifulSoup and urllib
 and train a classifier to distinguish between two categories (motorcycles and bikes)
 
 Please feel free to contact me for any suggestions or recommendations
 
 Tested with (Anaconda) Python3 on Ubuntu 14.04 LTS
 -- Use conda install to install missing packages in your platform
 
 Copyright- Dimitris Kastaniotis
 
 TODO: 
 -- remove duplicates and bad samples automatically
 -- Add TensorBoard with PyTorch
 -- Add TensorFlow
 -- Deploy on a web-server

 dkastaniotis@upatras.gr
 
"""
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
from PIL import Image as img

import time
import magic
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import json

from bs4 import BeautifulSoup  
import urllib
import urllib.request

import os
import sys
import datetime

plt.ion()  


#-----------------------------------------
#                     [Modify /home/dimitris/Desktop/chromedriver to your path]
#----------------------------------------
# Attention!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#change the filepath of your chrome driver.
ChromeDriver="/home/dimitris/Desktop/chromedriver";
ROOT        ='/tmp/split/'
IMAGES_DIR  = "/tmp/categories/";
CUDA        = True; # by default we are working on CPU
BATCH_SIZE  = 12;
EPOCHS      = 50;
TRAIN_DIR   = ROOT+'/train'
VAL_DIR     = ROOT+'/val'
WORK_DIR        = os.getcwd();
USER_PRETRAINED = False; # set to yes in order to use a model trained on imagenet
ONLY_TRAIN  = False; # Set this to true if you want to retrain your model on existing data
MaxNumImgs = 500 #                Number of Images per query
CUDA        =True
# check if file exists
if(not(os.path.exists(ChromeDriver))):
    print("================Error ===========================");
    print("You need to download chromedriver and change the filepath here");
    print("=================================================");
    sys.exit()
browser = webdriver.Chrome(ChromeDriver)



header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
 
#-----------------------------------------
#              create output folders
#-----------------------------------------
 
if(not(os.path.exists(ROOT))):
    os.mkdir(ROOT)
if(not(os.path.exists(IMAGES_DIR))):
    os.mkdir(IMAGES_DIR)    
#-----------------------------------------
#                     Put some queries here
#-----------------------------------------
# queries=['Motorcycle','Bicycle' ]
# Attributes=['riding','parking' ]

queries=['Raccoon','Cat' ]
Attributes=['running']
 
if(ONLY_TRAIN==False): 
    #-----------------------------------------
    #                   Part I: Download images
    #-----------------------------------------
    for q in queries:
        for att in Attributes:
            query = q+' '+att; #"acropolis"
            browser.get("https://www.google.gr/search?q="+query+"&client=ubuntu&hs=4fs&channel=fs&source=lnms&tbm=isch&tbs=itp:photo&sa=X&ved=0ahUKEwjq9IXixt_VAhVG7BQKHRDlCjUQ_AUICigB&biw=1422&bih=755")
            time.sleep(1)
            elem = browser.find_element_by_tag_name("body")
            scrol_down_times = 150
            for i in range(scrol_down_times):
                elem.send_keys(Keys.PAGE_DOWN)
                time.sleep(0.01)
            # show more results and scroll again
            browser.find_element_by_id('smb').click() 
            # scroll again
            for i in range(scrol_down_times):
                elem.send_keys(Keys.PAGE_DOWN)
                time.sleep(0.01)
            print("-----------------------------------------");
            print("Collecting <=",MaxNumImgs,"Images for ",query);
            print("-----------------------------------------");
            # pass page text to soup
            soup = BeautifulSoup(browser.page_source)
            Images=[] # init array 
            for a in soup.find_all("div",{"class":"rg_meta"}):
                link , Type =json.loads(a.text)["ou"]  ,json.loads(a.text)["ity"]
                if(Type=="jpg"): # keep only jpeg files
                    Images.append((link,Type))
            print("Total number of available (jpeg) images# "+ str(len(Images)))
            cnt=0; # count number of images
            for image,imtype in Images:
                cnt+=1
                if(cnt>MaxNumImgs):
                    break; 
                else:
                    try:
                        print( "Collecting #",cnt,"of ", MaxNumImgs  );
                        req = urllib.request.Request(image,  headers=header)
                        raw_img = urllib.request.urlopen(req).read()
                        imDir= IMAGES_DIR+q+"/"  # where to store images
                        if(not os.path.isdir(imDir)):
                            print(imDir)
                            os.mkdir(imDir)
                        cntr = len([i for i in os.listdir(imDir) if imtype in i]) + 1
                        # create a safe filename
                        extraFilename = str(datetime.datetime.now()).replace(' ', '_').replace(":","_")  # optional +     werkzeug.secure_filename(imagefile.filename)
                        Filename ="_"+q+"_"+ str(cntr)+extraFilename+"."+imtype;
                        f = open(imDir + Filename, 'wb')
                        f.write(raw_img)
                        f.close()
                        # clean corrupted files, or other file formats
                        try:
                            im=img.open(imDir + Filename);
                            # check that file is not gif
                            ch = magic.from_file(imDir + Filename);
                            if(ch.lower().find("jpeg")<0):
                               raise ValueError("Incopatible format") 
                        except  Exception :
                            os.remove(imDir + Filename)    
                    except:
                        # manage exceptions
                        pass
     
    browser.quit()
    #--------------------------------------------------------
    #     Part II. split dataset into training and validation
    # ---------------------------------------------------------


    if(not(os.path.exists(TRAIN_DIR))):  
        os.makedirs(TRAIN_DIR)
    if(not(os.path.exists(VAL_DIR))):
        os.makedirs(VAL_DIR)
     
    # move to images filder and list directories (each directory is one category)
    os.chdir(IMAGES_DIR)
    paths= os.listdir(IMAGES_DIR)

    print(paths)
     
    for p in paths:
        # create directories to store images
        if(not(os.path.exists(VAL_DIR+'/'+p))):
            os.makedirs(VAL_DIR+'/'+p)
        if(not(os.path.exists(TRAIN_DIR+'/'+p))):
            os.makedirs(TRAIN_DIR+'/'+p)
        os.chdir(IMAGES_DIR+p);
        print(IMAGES_DIR+p)
        files = os.listdir(".");
        # keep only 30/100 for val.
        val = int(len(files)/10);
        train = int(len(files)-val);
        print("Category:",p,"has",train,"train images and", val,"images")
        print("Creating training and validation set")
        for f in range(train):
            try:
                im = img.open(IMAGES_DIR+'/'+p+'/'+files[f]);
                im.save(TRAIN_DIR+'/'+p+'/'+files[f]);
            except Exception:
                print("Unexpected error:", sys.exc_info()[0])
                print("occured while processing ",files[f],"info:");        
           
        for f in range(val):
            try:
                im = img.open(IMAGES_DIR+'/'+p+'/'+files[f+train]);
                im.save(VAL_DIR +'/'+p+'/'+files[f+train]);
       
            except Exception:
                print("Unexpected error:", sys.exc_info()[0])
                print("occured while processing ",files[f+train]);
                
         
    os.chdir("..");

#--------------------------------------------------------
#     Part III. Train Classifier
#---------------------------------------------------------
 
# define image transformations during loading
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = ROOT
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
         for x in ['train', 'val']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=4)
                for x in ['train', 'val']}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes

use_gpu = False;
if(CUDA==True):
    use_gpu = torch.cuda.is_available()
 


# Get a batch of training data
inputs, classes = next(iter(dset_loaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
 
Acc_train=[]
Acc_val =[]
Loss_train=[]
Loss_val =[]
plt.figure("Loss")
plt.figure("Accuracy")

def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=200):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]
            if phase == 'train':
                Acc_train.append(epoch_acc)
                Loss_train.append(epoch_loss)
            if phase == 'val':
                Acc_val.append(epoch_acc)
                Loss_val.append(epoch_loss)
            plt.figure("Loss")
            plt.plot(Loss_train,color='blue')
            plt.plot(Loss_val,color='red')
            plt.legend(['train','val'])
            plt.show()
            plt.pause(1)
            plt.figure("Accuracy")
            plt.plot(Acc_train,color='blue')
            plt.plot(Acc_val,color='red')
            plt.legend(['train','val'])
            plt.show()
            plt.pause(1)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model

#--------------------------------------------
#             Custom learning rate scheduler 
#--------------------------------------------
def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=20):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
 
#--------------------------------------------
#                     Create Conv Net- Keel First two layers constant
#--------------------------------------------
if(USER_PRETRAINED):
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False
else:
    model_conv = torchvision.models.resnet18(pretrained=True)
        
# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, len(queries)) #---- output -----

if use_gpu:
    model_conv = model_conv.cuda()

criterion = nn.CrossEntropyLoss()

#----------------------------------
#                optimize layers active
#----------------------------------
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

#----------------------------------
#                 train network
#----------------------------------
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=EPOCHS)

#go back to script directory
os.chdir(WORK_DIR)

# save figures
plt.figure("Loss")
plt.savefig('Loss.png') 
plt.close()
plt.figure("Accuracy")
plt.savefig('Accuracy.png')
plt.close()
