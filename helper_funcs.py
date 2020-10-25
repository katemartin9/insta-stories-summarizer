import matplotlib.pyplot as plt


def plot_color_image(image):
    plt.imshow(image, interpolation="nearest")
    plt.axis("off")
    plt.show()

