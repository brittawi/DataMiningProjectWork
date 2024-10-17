import matplotlib.pyplot as plt

def showResultTransformations(data_dict, after_data):
    # visualization
    bf_image, bf_label = data_dict["image"], data_dict["label"]
    af_image, af_label = after_data["image"], after_data["label"]
    images = [bf_image, bf_label, af_image, af_label]
    titles = ["original", "label", "After transfrom", "label"]
    slice_number = 55
    fig, axes = plt.subplots(nrows = 2, ncols =2, figsize = (10, 5))
    k = 0
    for i in range(len(images)//2):
        for j in range(len(images)//2):
            if  k%2 == 0:
                axes[i, j].imshow(images[k][0][:, :, slice_number], cmap= 'gray')
            else:
                axes[i, j].imshow(images[k][0][:, :, slice_number])
            axes[i, j].set_title(titles[k])
            axes[i, j].set_axis_off()
            k+=1
    plt.show()

def showResultSpacing(data_dict, after_data):
    # visualization
    bf_image, bf_label = data_dict["image"], data_dict["label"]
    af_image, af_label = after_data["image"], after_data["label"]
    images = [bf_image, bf_label, af_image, af_label]
    titles = ["original", "label", "After transfrom", "label"]
    slice_number = 55
    fig, axes = plt.subplots(nrows = 2, ncols =2, figsize = (10, 5))
    k = 0
    for i in range(len(images)//2):
        for j in range(len(images)//2):
            if  i == 1:
                slice_number = int(slice_number * (2/3))
            else:
                slice_number = 55
            if  k%2 == 0:
                axes[i, j].imshow(images[k][0][:, :, slice_number], cmap= 'gray')
            else:
                axes[i, j].imshow(images[k][0][:, :, slice_number])
            axes[i, j].set_title(titles[k])
            axes[i, j].set_axis_off()
            k+=1
    plt.show()