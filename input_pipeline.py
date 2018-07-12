import tensorflow as tf

def _parse_function(filename, label):
  r_image_string = tf.read_file(filename)
  r_image_decoded = tf.image.decode_jpeg(r_image_string,channels=3)
  r_image_decoded = tf.reshape(r_image_decoded,[256,256,3])
  r_image_decoded = tf.image.convert_image_dtype(r_image_decoded,tf.float32)

  s_image_string = tf.read_file(label)
  s_image_decoded = tf.image.decode_jpeg(s_image_string,channels=3)
  s_image_decoded = tf.reshape(s_image_decoded,[256,256,3])
  s_image_decoded = tf.image.convert_image_dtype(s_image_decoded,tf.float32)

  r_image_decoded = tf.divide(r_image_decoded,[255])
  s_image_decoded = tf.divide(s_image_decoded,[255])


  r_image_decoded = tf.image.per_image_standardization(r_image_decoded)
  s_image_decoded = tf.image.per_image_standardization(s_image_decoded)

  # r_image_decoded = tf.image.resize_images(r_image_decoded,[64,64])
  # s_image_decoded = tf.image.resize_images(s_image_decoded,[64,64])
  print(r_image_decoded)

  return r_image_decoded, s_image_decoded

# used once to resize the images from 256 to 64
def imread(path):
    return scipy.misc.imread(path).astype(np.float)

def resize_width(image, width=64.):
    h, w = np.shape(image)[:2]
    return scipy.misc.imresize(image,[int((float(h)/w)*width),width])
        
def center_crop(x, height=64):
    h= np.shape(x)[0]
    j = int(round((h - height)/2.))
    return x[j:j+height,:,:]

def get_image(image_path, width=64, height=64):
    return center_crop(resize_width(imread(image_path), width = width),height=height)


def parse():
  rainy_start = 637
  rainy_end = 1207

  sunny_start = 1
  sunny_end = 571

  rainy_files = []
  sunny_files = []

  for i in range(rainy_start,rainy_end):
    file_id = "{0:0=4d}".format(i)
    rainy_files.append('./RainRemoval/rainy/000'+str(file_id)+'.jpeg')

  for i in range(sunny_start,sunny_end):
    file_id = "{0:0=3d}".format(i)
    sunny_files.append('./RainRemoval/sunny/0000'+str(file_id)+'.jpeg')


  rainy_filenames = tf.constant(rainy_files)
  sunny_filenames = tf.constant(sunny_files)
  dataset = tf.data.Dataset.from_tensor_slices((rainy_filenames, sunny_filenames))
  dataset = dataset.map(_parse_function).repeat().shuffle(buffer_size=50).apply(tf.contrib.data.batch_and_drop_remainder(1))
  # dataset = dataset.prefetch(4)

  return dataset
