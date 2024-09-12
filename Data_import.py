import nibabel as nib
import matplotlib.pylab as plt
import numpy as np

def visualise_image(image):
    fig, axes = plt.subplots(7,7, figsize=(8,8))
    for i, ax in enumerate(axes.reshape(-1)):
        ax.imshow(image[:,:,i])
    plt.show()

#Visualise one training image 
tr_path_img_001 = "/Users/emillundin/Desktop/D7043E/Project/Task08_HepaticVessel/imagesTr/hepaticvessel_001.nii.gz"
tr_img_001 = nib.load(tr_path_img_001).get_fdata()
print(tr_img_001.shape)
visualise_image(tr_img_001)

#Visualise one label image
label_path_img_001 = "/Users/emillundin/Desktop/D7043E/Project/Task08_HepaticVessel/labelsTr/hepaticvessel_001.nii.gz"
label_img_001 = nib.load(label_path_img_001).get_fdata()
print(label_img_001.shape)
visualise_image(label_img_001)
    