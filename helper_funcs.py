import matplotlib.pyplot as plt
import re


def plot_color_image(image):
    plt.imshow(image)
    plt.show()


def replace_chars(target: str) -> str:
    chars = ['\\n', '\\x0c']
    for char in chars:
        target = target.replace(char, '')
    return ' '.join(target.split('\n'))


def remove_special_chars(my_str):
    my_str = replace_chars(my_str).strip().lower()
    target = re.findall(r'(?u)\w+', my_str.lower())
    return target
