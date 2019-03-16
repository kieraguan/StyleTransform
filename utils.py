import os
import skimage
import skimage.io
import skimage.transform
import numpy as np

def load_image(path, shape = (224,224)):
    # load image
    image = skimage.io.imread(path)
    image = image / 255.0
    # Crop image from center
    edge_length = min(image.shape[:2])
    y = int((image.shape[0] - edge_length) / 2)
    x = int((image.shape[1] - edge_length) / 2)
    image= image[y: y + edge_length, x: x + edge_length]
    image = skimage.transform.resize(image, (shape[0], shape[1])) * 255.0
    return image

def get_files(img_dir):
    files = os.listdir(img_dir)
    paths = []
    for x in files:
        paths.append(os.path.join(img_dir, x))
    return paths
        
def load_images(folder, batch_shape, start):
    files = np.asarray(get_files(folder))
    X_batch = np.zeros(batch_shape, dtype=np.float32)

    idx = 0
    i = start
    while idx < batch_shape[0]: 
        try:
            f = files[i]
            i += 1
            img = load_image(f, batch_shape[1:3])
            if len(img.shape) < 3:
                continue
            if idx >= batch_shape[0]:
                break
            X_batch[idx] = img
            assert(not np.isnan(X_batch[idx].min()))
            idx += 1
            if idx % 500 == 0:
                print("Load ", idx)
        except Exception as e:
            print(e)
            print (f)
            break

    return X_batch