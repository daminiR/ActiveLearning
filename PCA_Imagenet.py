import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
import numpy as np
from glob import glob
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
#from sklearn.datasets import load_digits

#Input the Data folder here
BASE_DATA_FOLDER = "/home/data/ilsvrc/ILSVRC/ILSVRC2012_Classification"#"../cat-and-dog"#../hymenoptera_data"#"../Medical_data"
TRAin_DATA_FOLDER = os.path.join(BASE_DATA_FOLDER, "train")


#Input the Highlight folder here
BASE_HIGHLIGHT_FOLDER = "/home/nobelletay/al/org_centers"


#Plot
def visualize_scatter(data_2d, label_ids, figsize=(10,10)):
    plt.figure(figsize=figsize)
    plt.grid()
    
    nb_classes = len(np.unique(label_ids))
    
    for label_id in np.unique(label_ids):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    color=  'b',
                    linewidth='0.8',
                    alpha=0.6,
                    label=id_to_label_dict[label_id])

    for label_id in np.unique(label_ids):
        if (id_to_label_dict[label_id] == "highlight"):
            plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                        data_2d[np.where(label_ids == label_id), 1],
                        marker='o',
                        color= 'red',
                        linewidth='1.5',
                        alpha=0.6,
                        label=id_to_label_dict[label_id])


    #plt.legend(loc='best')
    #plt.show() 
    plt.savefig("T-SNE_result.png")

images = []
labels = []



class_counter=0
class_limit=200
image_counter=0
image_limit=200
#First resize the image to 200*200 with grey scale
for class_folder_name in os.listdir(TRAin_DATA_FOLDER):
    print(class_folder_name)
    image_counter=0
    class_counter = class_counter + 1
    if (class_counter > class_limit):
        break
    class_folder_path = os.path.join(TRAin_DATA_FOLDER, class_folder_name)
    for image_path in glob(os.path.join(class_folder_path, "*.JPEG")):
        image_counter = image_counter + 1
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (100, 100))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.flatten()
        images.append(image)
        labels.append(class_folder_name)
        if(image_counter > image_limit):
            break
#Read Images from Highlight folder
image_counter=0
for image_path in glob(os.path.join(BASE_HIGHLIGHT_FOLDER, "*.jpg")):
        print(image_path)
        image_counter = image_counter + 1
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (100, 100))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.flatten()
        images.append(image)
        labels.append("highlight")
        if(image_counter > image_limit):
            break


images = np.array(images)
labels = np.array(labels)

#Make the order of the label names
label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}

#Normalize the data
label_ids = np.array([label_to_id_dict[x] for x in labels])
images_scaled = StandardScaler().fit_transform(images)

#plt.imshow(np.reshape(images[35], (200,200)), cmap="gray")

#visualize_scatter(images_scaled , label_ids)


#How many features
pca = PCA(n_components=3)
pca_result = pca.fit_transform(images_scaled)
pca_result_scaled = StandardScaler().fit_transform(pca_result)
#visualize_scatter(pca_result_scaled, label_ids)


#Based on https://distill.pub/2016/misread-tsne/, the perplexity value can significantly affect the output plot 

tsne = TSNE(n_components=2, perplexity=30.0)
tsne_result = tsne.fit_transform(pca_result)
tsne_result_scaled = StandardScaler().fit_transform(tsne_result)
visualize_scatter(tsne_result_scaled, label_ids)


