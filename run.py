from files_ops import get_stories, check_file_type, convert_video_to_np
from tensorflow import keras
import tensorflow as tf
from helper_funcs import plot_color_image


if __name__ == '__main__':
    folder_name = 'insta_stories'
    insta_dict = get_stories(folder_name)
    for key, vals in insta_dict.items():
        for val in vals:
            if check_file_type(val):
                images = convert_video_to_np(val)
                plot_color_image(images[0])
                images_resized = tf.image.resize_with_pad(images, 224, 224).numpy().astype(int)
                plot_color_image(images_resized[0])

    model = keras.applications.resnet50.ResNet50(weights="imagenet")


