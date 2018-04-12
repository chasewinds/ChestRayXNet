import os
import glob
import logging
import tensorflow as tf
import tensorflow.contrib.data as tf_data


def get_file(fname_pattern):
    logging.debug("fname_pattern: %s", fname_pattern)
    if isinstance(fname_pattern, list):
        # if it is list, it should be a filename list
        tfnames = fname_pattern
    else:
        if os.path.splitext(os.path.basename(fname_pattern))[1] == ".tfrecord":
            # if tfname has .tfrecord extension, it means it is a single filename and has no match pattern
            tfnames = [fname_pattern]
        else:
            tfnames = glob.glob(fname_pattern + "_*-of-*.tfrecord")

    if len(tfnames) == 0:
        raise ValueError("Can not find file pattern: %s" % fname_pattern)


    # self.n_readers = len(tfnames) ##

    return tfnames# , n_samples

def parse_tfrecord(example_proto):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature((), tf.string, default_value='10'),
        ## make sence?? that is m2
    }
    parsed = tf.parse_single_example(example_proto, keys_to_features)
    raw_image = tf.image.decode_image(parsed['image/encoded'], channels=1)
    image = raw_image
    image = tf.to_float(image)
    label = parsed['image/class/label']
    label = tf.string_to_number(tf.string_split([label], delimiter="").values, tf.float32)
    label = tf.to_int64(label)
    return {'image': image, 'label': label}




if __name__ == "__main__":
    tf_names = get_file('data/tfrecords/train ')
    dataset = tf_data.TFRecordDataset(tf_names)  ##tf.data.TFRecordDataset
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=1)
    image = dataset['image']
    print image
