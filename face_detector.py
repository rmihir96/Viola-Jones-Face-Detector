
# coding: utf-8

# In[315]:


import numpy as np
import cv2
import math


# In[316]:


# X1 = np.asarray([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]])
# X2 = np.asarray([[1,4,3,1], [5,9,7,8], [9,10,11,22], [13,11,15,16]])
# X3 = np.asarray([[1,2,13,4], [5,10,7,8], [9,20,11,12], [13,19,15,16]])
# X4 = np.asarray([[10,2,3,4], [5,6,11,8], [9,10,20,12], [13,14,15,16]])
# X5 = np.asarray([[1,12,3,4], [1,6,7,8], [9,10,11,12], [13,14,15,16]])

# image_list = np.asarray([(X1,1), (X2,1), (X3,0), (X4,0), (X5,1)])

# print(image_list.shape)


# In[6]:


#this function calculates the integral image of given image
def calculate_integral_image(img):
    return np.cumsum(np.cumsum(img, axis=0), axis=1)


# In[67]:



#Function to compute scores of the features using Integral Image and the feature
def calculate_score(I,tup):
    x,y,w,h = tup
    D = I[x][y]
    B = I[x-h][y]
    C = I[x][y-w]
    A = I[x-h][y-w]
    if x-h < 0 :
        B = 0
        A = 0
    if y-w < 0:
        C = 0
        A = 0
    #print(D,B,C,A)
        
    return (D-B-C+A)


# In[343]:



#This function creates haar features of 5 types

def haar_features(img):
    height, width = img.shape[0], img.shape[1]
    print(height, width)
#     if choice == "A": 
   # print("Haar feature type %s selected" % choice)
# Pattern A  ----- ####    -> (White|Black)
    total = 0
    featureA = []
    featureA_scores = []
    for i in range(0,height):     # image height
        for j in range(0,width):     # image width
            for w in range(1,width+1):     # 1 to width + 1
                for h in range(1,height+1):     # 1 to height + 1
                    if (i+h<=height) and (j+2*w<=width):
                        x1 = max(i+h-1,0)
                        y1 = max(j+w-1, 0)
                        y2 = max(j+2*w-1, 0)
                        #print([x1,y1,w,h], [x1,y2,w,h])
#                         S1 = calculate_score(Y, x1,y1,w,h)
#                         S2 = calculate_score(Y, x1,y2,w,h)
#                        # print(S2,S1)
#                         score = S2-S1
#                         featureA_scores.append(score)
                        featureA.append([[(x1,y2,w,h)],[(x1,y1,w,h)]])

    #                     print(i,j,w,h)  
    #                     print("S1", (X[i:i+h,j:j+w]))
    #                     print("S2", (X[i:i+h,j+w:j+2*w]))

                        total += 1
#         print(total)
#         print(featureA_scores)
#         print(featureA)
#         return featureA

#     elif choice == "B":
    #Define feature B  #### -> Black
    #                  ---- -> White
    total = 0
    featureB = []
    featureB_scores = []
    for i in range(0,4):     # image height
        for j in range(0,4):     # image width
            for h in range(1,5):     # 1 to width + 1
                for w in range(1,5):     # 1 to height + 1
                    if (i+2*h<=4) and (j+w<=4):
                        x1 = max(i+h-1,0)
                        x2 = max(i+2*h-1, 0)
                        y1 = max(j+w-1, 0)
#                         #print([x1,y1,w,h], [x1,y2,w,h])
#                         S1 = calculate_score(Y, x1,y1,w,h) #Darker pixels
#                         S2 = calculate_score(Y, x2,y1,w,h) #Lighter pixels
#                        # print(S2,S1)
#                         score = S1-S2
#                         featureB_scores.append(score)
                        featureB.append([[(x1,y1,w,h)], [(x1,y2,w,h)]])

    #                     print(i,j,w,h)  
    #                     print("S1", (X[i:i+h,j:j+w]))
    #                     print("S2", (X[i+h:i+2*h,j:j+w]))

                        total += 1
#         print(total)
#         print(featureB_scores)
#         print(featureB)
#         return featureB

#     elif choice == "C":    
    # Feature C  ----- #### ----    -> (White|Black|White)
    total = 0
    featureC = []
    featureC_scores = []
    for i in range(0,4):     # image height
        for j in range(0,4):     # image width
            for w in range(1,5):     # 1 to width + 1
                for h in range(1,5):     # 1 to height + 1
                    if (i+h<=4) and (j+3*w<=4):
                        x1 = max(i+h-1,0)
                        y1 = max(j+w-1, 0)
                        y2 = max(j+2*w-1, 0)
                        y3 = max(j+3*w-1,0)
                        #print([x1,y1,w,h], [x1,y2,w,h])
#                         S1 = calculate_score(Y, x1,y1,w,h)
#                         S2 = calculate_score(Y, x1,y2,w,h)
#                         S3 = calculate_score(Y, x1,y3,w,h)
                       # print(S2,S1)
#                         score = S2-(S1+S3)
#                         featureC_scores.append(score)
                        featureC.append([[(x1,y2,w,h)], [(x1,y1,w,h), (x1,y3,w,h)]])

    #                     print(i,j,w,h)  
    #                     print("S1", (X[i:i+h,j:j+w]))
    #                     print("S2", (X[i:i+h,j+w:j+2*w]))
    #                     print("S3", (X[i:i+h, j+2*w:j+3*w]))

                        total += 1
#         print(total)
#         print(featureC_scores)
#         print(featureC)
#         return featureC

#     elif choice=="D":

    #Define feature D  ----- -> White
                        #### -> Black
    #                  ---- -> White
    # 
    total = 0
    featureD = []
    featureD_scores = []
    for i in range(0,4):     # image height
        for j in range(0,4):     # image width
            for h in range(1,5):     # 1 to width + 1
                for w in range(1,5):     # 1 to height + 1
                    if (i+3*h<=4) and (j+w<=4):
                        x1 = max(i+h-1,0)
                        x2 = max(i+2*h-1, 0)
                        x3 = max(i+3*h-1,0)
                        y1 = max(j+w-1, 0)
                        #print([x1,y1,w,h], [x1,y2,w,h])
#                         S1 = calculate_score(Y, x1,y1,w,h) #Lighter pixels
#                         S2 = calculate_score(Y, x2,y1,w,h) #Darker pixels
#                         S3 = calculate_score(Y, x3,y1,w,h) #Lighter pixels
#                        # print(S2,S1)
#                         score = S2-(S1+S3) 
#                         featureD_scores.append(score)
                        featureD.append([[(x1,y2,w,h)], [(x1,y1,w,h), (x3,y1,w,h)]])

    #                     print(i,j,w,h)  
    #                     print("S1", (X[i:i+h,j:j+w]))
    #                     print("S2", (X[i+h:i+2*h,j:j+w]))
    #                     print("S3", (X[i+2*h:i+3*h,j:j+w]))

                        total += 1
#         print(total)
#         print(featureD_scores)
#         print(featureD)
#         return featureD

#     elif choice=="E":


    # Feature type E
    total = 0
    featureE = []
    featureE_scores = []
    for i in range(0,4):     # image height
        for j in range(0,4):     # image width
            for w in range(1,5):     # 1 to width + 1
                for h in range(1,5):     # 1 to height + 1
                    if (i+2*h<=4) and (j+2*w<=4):
                        x1 = max(i+h-1,0)
                        y1 = max(j+w-1, 0)
                        y2 = max(j+2*w-1, 0)
                        x2 = max(i+2*h-1,0)
                        #print([x1,y1,w,h], [x1,y2,w,h])
#                         S1 = calculate_score(Y, x1,y1,w,h) #W
#                         S2 = calculate_score(Y, x1,y2,w,h) #B
#                         S3 = calculate_score(Y, x2,y1,w,h ) #W
#                         S4 = calculate_score(Y, x2,y2,w,h) #B
#                        # print(S2,S1)
#                         score = (S2+S3)-(S1+S4)
#                         featureE_scores.append(score)
                        featureE.append([[(x1,y2,w,h), (x2,y1,w,h)], [(x1,y1,w,h), (x2,y2,w,h)]])

    #                     print(i,j,w,h)  
    #                     print("S1", (X[i:i+h,j:j+w]))
    #                     print("S2", (X[i:i+h,j+w:j+2*w]))
    #                     print("S3", (X[i+h:i+2*h,j:j+w]))
    #                     print("S4", (X[i+h:i+2*h,j+w:j+2*w]))
                        total += 1
#         print(total)
#         print(featureE_scores)
#         print(featureE)
#         return featureE
    features = featureA + featureB + featureC + featureD + featureE
    print(len(features))
    return features
   


# In[182]:


def apply_features(int_imgs, features):
    """
    args:
    I - Intergral image list
    features - Haar features
    return:
    X - list of feature score for all images
    """
    X = []
    y = []
#     X = np.zeros((len(features), len(I)))
#     print(X.shape)
    for img in int_imgs:
        j = []
        for pos, neg in features:
            pos_sum = sum([calculate_score(img[0], i) for i in pos])
            neg_sum = sum([calculate_score(img[0], i) for i in neg])
            j.append(pos_sum-neg_sum)
        X.append(j)
        y.append(img[1])
    X = np.asarray(X).T
    y = np.asarray(y)
    return X,y 
        


# In[340]:


def train(training, pos_num, neg_num):
    weights = np.zeros(len(training))
    training_data = []
    for x in range(len(training)):
        training_data.append((calculate_integral_image(training[x][0]), training[x][1]))
        if training[x][1] == 1:
            weights[x] = 1.0 / (2 * pos_num)
        else:
            weights[x] = 1.0 / (2 * neg_num)
    features = haar_features(training_data[0][0])
    #print(len(features))
    X, y = apply_features(training_data, features)
    final_weak_classifiers = []
    alpha_list = []
    for t in range(10): #number of weak classifiers
        weights = weights / np.linalg.norm(weights)
   # print(X.shape)
    #print(len(y))
        wc = weak_classifier(X, y, features, weights)
        #wc = train_weak(X, y ,features, weights)
        #print(wc[0])
        #print(len(wc))
        best_clf, error, accuracy = select_weak_classifier(wc, training_data, weights)
        beta = error / (1.0 - error)
        for i in range(len(accuracy)):
            weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
        final_weak_classifiers.append(best_clf)      
        alpha = math.log(1.0/beta)
        alpha_list.append(alpha)

    return final_weak_classifiers, alpha_list


# In[314]:


# clf, alpha = train(image_list, 3, 2)
# print(clf)
# print(alpha)

# X6 = np.asarray([[10,2,3,4], [5,6,11,8], [9,10,20,12], [13,14,15,16]]) 

# print(strong_classifier(X6, alpha, clf))


# In[277]:


def weak_classifier(X, y , features, weights):
    #print(X.shape)
    total_pos, total_neg = 0, 0
    for w, label in zip(weights, y):
        if label == 1:
            total_pos+= w
        else:
            total_neg+= w
    classifiers = []
    for i, f in enumerate(X):
       # print(f)
        #Sort weights as per feature values
        temp_feat = sorted(zip(weights, f, y), key = lambda k:k[1])
        pos_w, neg_w = 0, 0
        no_pos, no_neg = 0, 0
        min_error, max_feature, max_threshold, max_polarity = float('inf'), None, None, None
        #print(temp_feat)
        for w,feat, label in temp_feat:
            error = min(pos_w + total_neg - neg_w, neg_w + total_pos - pos_w)
            if error < min_error:
                min_error = error
                max_feature = features[i]
                max_threshold = feat
                max_polarity = 1 if no_pos > no_neg else -1
            if label == 1:
                pos_w+= w
                no_pos+=1
            else:
                neg_w+=1
                no_neg+=1
       # print(max_feature)
       # print(max_threshold)
       # print(max_polarity)
        classifiers.append((max_feature, max_threshold, max_polarity))
    return classifiers
            
                


# In[292]:


def select_weak_classifier(classifiers, training_data, weights):
    """
    args : Weak classifiers
    training_data : Integral images with label
    return : 
    """
    best_classifier, min_error, max_acc = None, float('inf'), None
    
    for clf in classifiers:
        error, accuracy = 0, []
        for img, w  in zip(training_data, weights):
            acc = abs(classify_img(clf, img[0]) - img[1])
            accuracy.append(acc)
            error += w * acc
        error = error / len(training_data)
        if error < min_error:
            min_error = error
            best_classifier = clf
            max_acc = accuracy
    return best_classifier, min_error, max_acc
        


# In[294]:


def classify_img(classifier, I):
    """
    args : Weak Classifier
    I : Integral image from the list
    return : classification as 1/0
    """
    
    feature, threshold, polarity = classifier
    
   # feature, threshold, polarity = wc[130]
    
    pos = feature[0]
    neg = feature[1]

    psum = sum([calculate_score(I, p) for p in pos])
    nsum = sum([calculate_score(I,n) for n in neg])
    
    f_score = psum - nsum
    
    return 1 if polarity * f_score < polarity * threshold else 0


    


# In[312]:


def strong_classifier(image, alpha, classifier):
    int_img = calculate_integral_image(image)
    total = 0
    for a, wc in zip(alpha, classifier):
        total+= a * classify_img(wc, int_img)
    return 1 if total >= sum(alpha) else 0


# In[364]:


import os
path = r"D:\Study Material\CVIP\Project3\train\face"

imagePaths = (os.listdir(path))
#print(imagePaths)
images = []
for imagePath in imagePaths[:1000]:
    image = cv2.imread(os.path.join(path, imagePath), 0)
    if image is not None:
        images.append((image,1))

#import os
path = r"D:\Study Material\CVIP\Project3\train\non-face"

imagePaths = (os.listdir(path))
#print(imagePaths)
for imagePath in imagePaths[:1000]:
    image = cv2.imread(os.path.join(path, imagePath), 0)
    if image is not None:
        images.append((image,0))


# In[365]:


import time
ctime = time.time()
clf, alpha = train(images, 1000, 1000)

print(time.time()-ctime)


# In[346]:


print(clf)


# In[347]:


print(alpha)


# In[362]:


test = cv2.imread(r"D:\Study Material\CVIP\Project3\train\non-face\B1_00103.pgm", 0)


# In[363]:


print(strong_classifier(test, alpha, clf))

