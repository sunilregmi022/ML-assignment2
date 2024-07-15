import os
import cv2
import matplotlib.pyplot as plt

# Define the dataset path
dataset_path = "./dataset"

# Function to read and display images
def display_images(dataset_path):
    # Get class names from subdirectories
    class_names = sorted(os.listdir(dataset_path))

    # Create a figure to display images, dynamically setting the number of columns
    num_cols = 10  # You can adjust this to set the number of images per row
    fig, axs = plt.subplots(len(class_names), num_cols, figsize=(20, len(class_names) * 2))

    # Iterate through each class
    for i, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        image_files = sorted(os.listdir(class_path))  # Sort to read images in order

        # Iterate through each image in the class
        for j, image_file in enumerate(image_files):
            if j >= num_cols:  # Skip images beyond the first num_cols
                break
            image_path = os.path.join(class_path, image_file)

            # Read the image using OpenCV
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

                # Display the image
                axs[i, j].imshow(image)
                # Set the title to the image file name
                axs[i, j].set_title(image_file, fontsize=8)
                axs[i, j].axis('off')
            else:
                axs[i, j].axis('off')

        # Fill remaining columns in the row with empty axes if less than num_cols images
        for k in range(j + 1, num_cols):
            axs[i, k].axis('off')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

# Call the function to display images
display_images(dataset_path)

