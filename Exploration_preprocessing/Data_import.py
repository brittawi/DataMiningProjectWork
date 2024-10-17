import nibabel as nib
import matplotlib.pylab as plt
import numpy as np
import nilearn as nil
import scipy.ndimage as ndi
from nilearn import plotting
import math

def load_image(path):
    img = nib.load(path)
    data = img.get_fdata()
    print(f"Image shape: {data.shape}")
    print("Metadata:")
    print(img.header)
    return img, data

def visualise_image(image):
    dims = int(math.ceil(math.sqrt(image.shape[2])))
    print(dims)
    fig, axes = plt.subplots(dims, dims, figsize=(dims,dims))
    fig.tight_layout(pad=0)
    plt.subplots_adjust(left=None, bottom=None, right=1, top=1, wspace=None, hspace=None)
    for i, ax in enumerate(axes.reshape(-1)):
        ax.axis("off")
        if i >= image.shape[2]:
            continue
        ax.imshow(ndi.rotate(image[:,:,i], 90))
    plt.show()

def plot_slice(image, slice):
    plt.imshow(ndi.rotate(image[:,:,slice], 90))
    #plt.axis('off')
    plt.show()

#Visualise one training image 
tr_path_img_001 = "Data/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz"
tr_path_img_003 = "Data/Task02_Heart/imagesTr/la_003.nii.gz"
#tr_img_001, tr_data_001 = load_image(tr_path_img_003)
#visualise_image(tr_data_001)
#plot_slice(tr_data_001, 30)

#Visualise one label image
label_path_img_001 = "/Users/emillundin/Desktop/D7043E/Project/Task08_HepaticVessel/labelsTr/hepaticvessel_001.nii.gz"
# label_img_001, label_data_001 = load_image(label_path_img_001)
# visualise_image(label_data_001)
# plot_slice(label_data_001, 30)