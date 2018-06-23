import tensorflow as tf

def _parse_function(filename, label):
  r_image_string = tf.read_file(filename)
  r_image_decoded = tf.image.decode_image(r_image_string,channels=3)
  r_image_resized = tf.image.resize_image_with_crop_or_pad(r_image_decoded, 256, 256)

  s_image_string = tf.read_file(label)
  s_image_decoded = tf.image.decode_image(s_image_string)
  s_image_resized = tf.image.resize_image_with_crop_or_pad(s_image_decoded, 256, 256)

  return r_image_resized, s_image_resized


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
  dataset = dataset.map(_parse_function).shuffle(buffer_size=50).apply(tf.contrib.data.batch_and_drop_remainder(4))
  # dataset = dataset.prefetch(4)

  return dataset
