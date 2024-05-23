import os
from os import scandir
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def merge(ls):
    merge_list = []
    for i in range(len(ls)):
        merge_list.extend(ls[i])
    return merge_list


def pass_file(file, ext):
    return file.is_file() and file.name.endswith(tuple(ext))


def q_files(path, ext):
    quantity = 0
    with scandir(path) as images:
        for image in images:
            if pass_file(image, ext):
                quantity += 1
    return quantity


def q_recursive_files(src, ext, initial_value):
    with scandir(src) as entries:
        for entry in entries:
            if entry.is_dir():
                initial_value = q_recursive_files(
                    entry.path, ext, initial_value)
            elif pass_file(entry, ext):
                initial_value += 1
    return initial_value


def distribute(dt, init_size):
    ref_batch = round(len(dt) * init_size)
    return [dt[:ref_batch], dt[ref_batch:]]

def extensible_img(
        img_cv2,
        image_data_generator,
        quantity,
        ext='jpg',
        save_format='jpg'):
    img = img_cv2.reshape((1,) + img_cv2.shape)
    i = 0
    for batch in image_data_generator.flow(img, save_prefix='ext-img', save_format=save_format):
        #image_array = np.array(batch[0])
        image_array = cv2.cvtColor(batch[0], cv2.COLOR_BGR2RGB) #cv2.cvtColor(batch[0], cv2.COLOR_BGR2GRAY)
        yield image_array
        i += 1
        if i >= quantity:
            break

def imgs_to_array(
        src,
        exclude_filename: str,
        dim=cv2.IMREAD_GRAYSCALE,
        callback=None
):
    routes = []
    image_names = []
    array = []
    with scandir(src) as images:
        for image in images:
            if image.name == exclude_filename:
                continue
            route = os.path.abspath(image)
            routes.append(route)
            image_names.append(image.name)
    
    for (route, image_name) in zip(routes,image_names):
        payload = cv2.imread(route, dim)
        if callback:
            payload = callback(payload, image_name)
        array.append(payload)        
    
    return array


def resize(scale, img):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dimensions = (width, height)
    img_resize = cv2.resize(img, dimensions)
    #img_resize = tf.expand_dims(img_resize, axis=-1)
    return img_resize


def reduce_flow(
        src,
        ext,
        dim=cv2.IMREAD_GRAYSCALE,
        callback=None,
        conf={
            'same_scale': True,
            'percentage': 0.5,
            'factor': {
                'size': (400, 400),
                'fx': 0.5,
                'fy': 0.7,
                'interpolation': cv2.INTER_AREA
            }
        }):
    with scandir(src) as images:
        for img in images:
            if pass_file(img, ext):
                img_obj = cv2.imdecode(np.fromfile(
                    img.path, dtype=np.uint8), dim)
                #if dim == cv2.IMREAD_UNCHANGED:
                #    img_obj = cv2.rotate(img_obj, cv2.ROTATE_90_CLOCKWISE)

                img_resize = None
                
                if conf['same_scale']:
                    img_resize = resize(conf['percentage'], img_obj)
                else:
                    img_resize = cv2.resize(
                        img_obj,
                        conf['factor']['size'],
                        fx=conf['factor']['fx'], 
                        fy=conf['factor']['fy'], 
                        interpolation = conf['factor']['interpolation'])

                if callback is not None:
                    callback(img_resize)
                yield img_resize


def extendible_flow(
    images,
    image_data_generator,
    u_quantity
):
    data = []
    for image in images:
        ext_images = list(
            extensible_img(
                image,
                image_data_generator,
                u_quantity,
            )
        )
        data.extend(ext_images)
    return data

def join_paths_from_extensions(src, ext, paths=[]):
    with scandir(src) as entries:
        for entry in entries:
            if entry.is_dir():
                paths = join_paths_from_extensions(entry.path, ext, paths)
            elif pass_file(entry, ext):
                paths.append(os.path.abspath(
                    os.path.join(entry.path, os.pardir)))
    return [*set(paths)]

# images preview

def show_img(index, data, label):
    plt.figure()
    plt.imshow(data[index])
    plt.colorbar()
    plt.xlabel(label[index])
    plt.grid(False)
    plt.show()

def search_in_list(list_, value):
    for index, item in list_:
        if item == value:
            return index
    return None
