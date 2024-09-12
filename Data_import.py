import nibabel as nib
import matplotlib.pylab as plt
import numpy as np
import nilearn as nil
import scipy.ndimage as ndi
from nilearn import plotting

def load_image(path):
    img = nib.load(path)
    data = img.get_fdata()
    print(f"Image shape: {data.shape}")
    print("Metadata:")
    print(img.header)
    return img, data

def visualise_image(image):
    fig, axes = plt.subplots(7,7, figsize=(8,8))
    for i, ax in enumerate(axes.reshape(-1)):
        ax.imshow(ndi.rotate(image[:,:,i], 90))
    plt.axis("off")
    plt.show()

def plot_slice(image, slice):
    plt.imshow(ndi.rotate(image[:,:,slice], 90))
    #plt.axis('off')
    plt.show()

#Visualise one training image 
tr_path_img_001 = "/Users/emillundin/Desktop/D7043E/Project/Task08_HepaticVessel/imagesTr/hepaticvessel_001.nii.gz"
tr_img_001, tr_data_001 = load_image(tr_path_img_001)
#visualise_image(tr_data_001)
plot_slice(tr_data_001, 30)

#Visualise one label image
label_path_img_001 = "/Users/emillundin/Desktop/D7043E/Project/Task08_HepaticVessel/labelsTr/hepaticvessel_001.nii.gz"
label_img_001, label_data_001 = load_image(label_path_img_001)
#visualise_image(label_data_001)
plot_slice(label_data_001, 30)