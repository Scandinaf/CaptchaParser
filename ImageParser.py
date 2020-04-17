import io
import json
import random
import string
import urllib.request

import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from cairosvg import svg2png

__white_pixel = np.array([255, 255, 255])


def get_images(url):
    document = urllib.request.urlopen(url).read()
    json_document = json.loads(document)
    rgb_array = __svg_to_rgb(json_document["data"])
    return __split_to_images(rgb_array)


def collect_images_for_model(url, path_to_folder):
    images = get_images(url)
    base_name = __generate_image_name()
    for index in range(len(images)):
        __save_image(images[index], base_name, index, path_to_folder)


def __svg_to_rgb(document):
    soup = BeautifulSoup(document, 'lxml')
    [__customize_element(e) for e in list(soup.svg.children)]
    __add_background(soup)

    svg_bytes = svg2png(bytestring=str(soup.svg))
    image = Image.open(io.BytesIO(svg_bytes))
    return np.array(image)


def __split_to_images(rgb_array):
    images = []
    buffer = []
    for cell in rgb_array.transpose((1, 0, 2)):
        buffer_len = len(buffer)
        for rgb in cell:
            if not __is_white_pixel(rgb):
                buffer.append(cell)
                break
        if 0 < buffer_len == len(buffer):
            new_rgb_array = np.array(buffer).transpose(1, 0, 2)
            image = Image.fromarray(__exclude_white_rows(new_rgb_array), 'RGB')
            images.append(image)
            buffer = []
    return images


def __exclude_white_rows(rgb_array):
    buffer = []
    for row in rgb_array:
        if next((rgb for rgb in row if not __is_white_pixel(rgb)), None) is not None:
            buffer.append(row)
    return np.array(buffer)


def __is_white_pixel(rgb):
    return (rgb == __white_pixel).all()


def __save_image(img, base_name, index, path_to_folder):
    img.save('{b1}/{b2}_{b3}.png'.format(b1=path_to_folder, b2=base_name, b3=index))


def __generate_image_name():
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(10))


def __add_background(soup):
    new_rect = soup.new_tag(name='rect', attrs={"width": '100%', "height": '100%', 'fill': 'white'})
    soup.svg.insert(0, new_rect)


def __customize_element(el):
    if 'stroke' in el.attrs:
        el.extract()
    else:
        el.attrs['fill'] = 'black'
