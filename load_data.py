# this file is used to load the data and split the data into training and testing sets
# %config InlineBackend.figure_format = 'retina'
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split


def load_data(path):
    # Specify the file path by reading through the directory
    n_images = 50
    image_array_collection = []
    mask_array_collection = []
    for i in range(n_images):
        temp = "{:02d}".format(i)
        file_path = (f"{path}/Case{temp}.mhd", f"{path}/Case{temp}_segmentation.mhd")

        # print(file_path)
        # Read the .mhd file using SimpleITK
        image = sitk.ReadImage(file_path[0])
        mask = sitk.ReadImage(file_path[1])

        # Convert the image to a NumPy array
        image_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)

        # Get the image dimensions
        size = image.GetSize()
        curr_index = size[2] // 2
        # print(
        #     f"For Patient {i+1}, Image dimensions: {size},"
        #     f" image example from layer {curr_index}"
        # )
        image_array_collection.append(image_array[curr_index, :, :])
        mask_array_collection.append(mask_array[curr_index, :, :])

    return image_array_collection, mask_array_collection


def split_data(image_array_collect, mask_array_collect, split_ratio=0.2):
    # Split the data into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(
        image_array_collect, mask_array_collect, test_size=split_ratio
    )

    return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    basepath = os.getcwd()
    # print(basepath)
    # path = basepath.join("./Train")
    rel_path = "./Train"
    image_array_collection, mask_array_collection = load_data(rel_path)
    X_train, X_val, y_train, y_val = split_data(
        image_array_collection, mask_array_collection
    )
    print(f"Training set: {len(X_train)}")
    print(f"Validation set: {len(X_val)}")
    # print(X_train[-1])
    print(y_train[-1])
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(X_train[0], cmap="gray")
    plt.title("Original Image X")
    plt.subplot(1, 2, 2)
    plt.imshow(y_train[0], cmap="gray")
    plt.title("Labeled Image y")
    plt.show()
