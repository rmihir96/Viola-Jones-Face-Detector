

import argparse
import numpy as np
import cv2
import math
import pickle
import os
import json
from collections import Counter
from nms import nms

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 2.")
    parser.add_argument(
        "dir_path", type=str, default="",
        help="path to the images used for  (do not change this arg)")
    args = parser.parse_args()
    return args

#this function calculates the integral image of given image
def calculate_integral_image(img):
    return np.cumsum(np.cumsum(img, axis=0), axis=1)


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
    
    return (D-B-C+A)
  



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

                        featureA.append([[(x1,y2,w,h)],[(x1,y1,w,h)]])
    


                        total += 1

#     elif choice == "B":
    #Define feature B  #### -> Black
    #                  ---- -> White
    total = 0
    featureB = []
    featureB_scores = []
    for i in range(0,height):     # image height
        for j in range(0, width):     # image width
            for h in range(1,height+1):     # 1 to width + 1
                for w in range(1,width+1):     # 1 to height + 1
                    if (i+2*h<=height) and (j+w<=width):
                        x1 = max(i+h-1,0)
                        x2 = max(i+2*h-1, 0)
                        y1 = max(j+w-1, 0)

                        featureB.append([[(x1,y1,w,h)], [(x1,y2,w,h)]])

    #                     print(i,j,w,h)  
    #                     print("S1", (X[i:i+h,j:j+w]))
    #                     print("S2", (X[i+h:i+2*h,j:j+w]))

                        total += 1


#     elif choice == "C":    
    # Feature C  ----- #### ----    -> (White|Black|White)
    total = 0
    featureC = []
    featureC_scores = []
    for i in range(0,height):     # image height
        for j in range(0,width):     # image width
            for w in range(1,width+1):     # 1 to width + 1
                for h in range(1,height+1):     # 1 to height + 1
                    if (i+h<=height) and (j+3*w<=width):
                        x1 = max(i+h-1,0)
                        y1 = max(j+w-1, 0)
                        y2 = max(j+2*w-1, 0)
                        y3 = max(j+3*w-1,0)

                        featureC.append([[(x1,y2,w,h)], [(x1,y1,w,h), (x1,y3,w,h)]])



                        total += 1


    #Define feature D  ----- -> White
                        #### -> Black
    #                  ---- -> White
    # 
    total = 0
    featureD = []
    featureD_scores = []
    for i in range(0,height):     # image height
        for j in range(0,width):     # image width
            for h in range(1,height+1):     # 1 to width + 1
                for w in range(1,width+1):     # 1 to height + 1
                    if (i+3*h<=height) and (j+w<=width):
                        x1 = max(i+h-1,0)
                        x2 = max(i+2*h-1, 0)
                        x3 = max(i+3*h-1,0)
                        y1 = max(j+w-1, 0)

                        featureD.append([[(x1,y2,w,h)], [(x1,y1,w,h), (x3,y1,w,h)]])



                        total += 1

    total = 0
    featureE = []
    featureE_scores = []
    for i in range(0,height):     # image height
        for j in range(0,width):     # image width
            for w in range(1,width+1):     # 1 to width + 1
                for h in range(1,height+1):     # 1 to height + 1
                    if (i+2*h<=height) and (j+2*w<=width):
                        x1 = max(i+h-1,0)
                        y1 = max(j+w-1, 0)
                        y2 = max(j+2*w-1, 0)
                        x2 = max(i+2*h-1,0)
  
                        featureE.append([[(x1,y2,w,h), (x2,y1,w,h)], [(x1,y1,w,h), (x2,y2,w,h)]])


                        total += 1

    features = featureA + featureB + featureC + featureD + featureE
    print(len(features))
    #print(featureA)
    return features
   

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
        pos = features[0]
        neg = features[1]
#             print("positive", pos)
#             print("Negative", neg)
        pos_sum = sum([calculate_score(img[0], i) for i in pos])
        neg_sum = sum([calculate_score(img[0], i) for i in neg])
        X.append(pos_sum-neg_sum)
     
        y.append(img[1])
    X = np.asarray(X).T
    y = np.asarray(y)
    return X,y 
        

def img_labels(train_imgs):
    labels = []
    for img in train_imgs:
        labels.append(img[1])
    return labels

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
    #X, y = apply_features(training_data, features)
    final_weak_classifiers = []
    alpha_list = []
    labels = img_labels(training_data)
    for t in range(2): #number of weak classifiers
        weights = weights / np.linalg.norm(weights)
        print(weights)
   # print(X.shape)
    #print(len(y))
        
        wc = weak_classifier(training_data, labels, features, weights)
        #wc = train_weak(X, y ,features, weights)
        #print(wc[0])
        #print(len(wc))
        best_clf, error, accuracy = select_weak_classifier(wc, training_data, weights)
        print("best clf",best_clf)
        print("Error",error)
        print("Acc", accuracy)
        beta = error / (1.0 - error)
        print("Beta", beta)
        for i in range(len(accuracy)):
            #print(weights[i], beta, accuracy[i])
            weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
            #print("i - weights", (i, weights[i]))
        final_weak_classifiers.append(best_clf)      
        #print(1.0/beta)
        alpha = math.log(1.0/beta)
        alpha_list.append(alpha)

    return final_weak_classifiers, alpha_list



def weak_classifier(training_data, labels, features, weights):
    #print(X.shape)
    total_pos, total_neg = 0, 0
    for w, label in zip(weights, labels):
        if label == 1:
            total_pos+=w
        else:
            total_neg+=w
    classifiers = []
    for feature in features:
        f, y = apply_features(training_data, feature)
        #print(f)
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
                max_feature = feature
                max_threshold = feat
                max_polarity = 1 if no_pos > no_neg else -1
            if label == 1:
                pos_w+=w
                no_pos+=1
            else:
                neg_w+=1
                no_neg+=1
#         print(max_feature)
#         print(max_threshold)
#         print(max_polarity)
        classifiers.append((max_feature, max_threshold, max_polarity))
    #print(classifiers)
    return classifiers
            
                

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
            #print(classify_img(clf, img[0]) , img[1])
            acc = abs(classify_img(clf, img[0]) - img[1])
            
            accuracy.append(acc)
            error+= w * acc
        error = error / len(training_data)
        #print("Error", error)
        if error < min_error:
            min_error = error
            best_classifier = clf
            max_acc = accuracy
        #print(best_classifier, min_error, max_acc)
    return best_classifier, min_error, max_acc
        

def classify_img(classifier, I):
    """
    args : Weak Classifier
    I : Integral image from the list
    return : classification as 1/0
    """
    
    feature, threshold, polarity = classifier
    #print(feature, threshold, polarity)
   # feature, threshold, polarity = wc[130]
    
    pos = feature[0]
    neg = feature[1]
#     print("pos", pos)
#     print("neg", neg)
    psum = sum([calculate_score(I, p) for p in pos])
    nsum = sum([calculate_score(I,n) for n in neg])
    
    f_score = psum - nsum
    
    return 1 if polarity * f_score < polarity * threshold else 0


    

def strong_classifier(image, alpha, classifier):
    int_img = calculate_integral_image(image)
#     int_img = int_img - np.mean(int_img) / np.std(int_img)
    total = 0
    for a, wc in zip(alpha, classifier):
        total+= a * classify_img(wc, int_img)
    return total, 1 if total >= 0.6*sum(alpha) else 0


# In[ ]:

def main():
    #Training commented out:
    ##################################################################

    # clf, alpha = train(images, 2429, 4548)
    # print(clf)
    # print(alpha)

    # final = zip(clf, alpha)

    # pickle.dump(final, open('./final_classifier.pickle', 'wb'), protocol=2)




    #########################################
    args = parse_args()
    print("loading images...")
    imagePaths = (os.listdir(args.dir_path))
    


    with open(r"./final_classifier.pickle", "rb") as f:
        file = pickle.load(f)

    clf = []
    alpha = []
    for i, j in file:
    #     print(i,j)
        clf.append(i)
        alpha.append(j)

    #Weak Classifier List
    # clf = [([[(5, 18, 17, 3)], [(8, 18, 17, 3)]], -1050.0, 1), ([[(5, 13, 9, 6)], [(11, 13, 9, 6)]], -216.0, -1), ([[(5, 13, 10, 6)], [(11, 13, 10, 6)]], -365.0, 1), ([[(4, 13, 11, 5)], [(9, 13, 11, 5)]], -312.0, -1), ([[(8, 13, 10, 6)], [(14, 13, 10, 6)]], -367.0, -1), ([[(5, 14, 13, 5)], [(10, 14, 13, 5)]], -1042.0, 1), ([[(7, 17, 17, 3)], [(10, 17, 17, 3)]], -49.0, -1), ([[(7, 17, 11, 3)], [(10, 17, 11, 3)]], -127.0, 1), ([[(7, 16, 10, 3)], [(10, 16, 10, 3)]], -74.0, -1), ([[(7, 9, 4, 2)], [(9, 9, 4, 2)]], -21.0, 1)]

    #Alpha List
    # alpha = [6.575151471691274, 7.266783069631535, 9.342094529936636, 12.262314436203779, 9.807209897582606, 11.805431834748102, 11.677685482457234, 10.758513212206026, 14.769595937010674, 10.26293208110991]
    json_list = []

    print(len(imagePaths))
    for imagePath in imagePaths:
        print(imagePath)
        image = cv2.imread(os.path.join(args.dir_path, imagePath), 0)
        ogimg = cv2.imread(os.path.join(args.dir_path, imagePath))
       
    #     ogimg = cv2.imread(os.path.join(path, imagePath))
        img = image.copy()
        faces = list()
        scores = list()
        for i in range(0, len(ogimg), 19):
                for j in range(0, len(ogimg[0]), 19):
                    t_img = (img[i:i+19, j:j+19])

                    if t_img.shape[0] != 19 or t_img.shape[1] != 19:
                        t_img = cv2.resize(t_img, (19, 19))

                    t, res = strong_classifier(t_img, alpha, clf)

                    if res == 1:
                        
                            faces.append((j, i, j+60, i+60))
                            scores.append(t)

        # print(len(faces))       

        z = sorted(zip(faces, scores), key=lambda x: x[1], reverse=True)
        faces = [i[0] for i in z]
        scores = [i[1] for i in z]
        
    #     print((scores))
        indicies = nms.boxes(faces, scores)
    #     print(len(indicies))
        # print(indicies)
        final_faces = []
        final_scores = []
        for c, i in enumerate(indicies[:4]):
            face = faces[i]
            final_faces.append([face[0], face[1], face[2]+200, face[3]+200])
            final_scores.append(scores[i])
            
            # print(faces[i], scores[i])

        
        index  = nms.boxes(final_faces, final_scores)
        # print(index)
        for i in index:
            face = final_faces[i]
            if (face[0] < ogimg.shape[1] and face[2]+50 < ogimg.shape[1] and face[1] < ogimg.shape[0] and face[3]+50 < ogimg.shape[0]):
                cv2.rectangle(ogimg, (face[0], face[1]), (face[2]-100, face[3]-100), (0,255,255), 2)
                element = {"iname": imagePath, "bbox" : [face[0], face[1], face[2]-100, face[3]-100]}
                json_list.append(element)
        print(len(json_list))
    # cv2.imwrite(os.path.join(respath, (imagePath)), ogimg)





    # print(json_list)
    #the result json file name
    output_json = "results.json"
    #dump json_list to result.json
    with open(output_json, 'w') as f:
        json.dump(json_list, f)

if __name__ == "__main__":
    
    main()
