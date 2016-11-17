# First check the Python version
import sys
if sys.version_info < (3,4):
    print('You are running an older version of Python!\n\n' \
          'You should consider updating to Python 3.4.0 or ' \
          'higher as the libraries built for this course ' \
          'have only been tested in Python 3.4 and higher.\n')
    print('Try installing the Python 3.5 version of anaconda '
          'and then restart `jupyter notebook`:\n' \
          'https://www.continuum.io/downloads\n\n')

# Now get necessary libraries
try:
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from skimage.transform import resize
    from skimage import data
    from scipy.misc import imresize
    import IPython.display as ipyd
except ImportError:
    print('You are missing some packages! ' \
          'We will try installing them before continuing!')
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from skimage.transform import resize
    from skimage import data
    from scipy.misc import imresize
    import IPython.display as ipyd
    print('Done!')

# Import Tensorflow
try:
    import tensorflow as tf
except ImportError:
    print("You do not have tensorflow installed!")
    print("Follow the instructions on the following link")
    print("to install tensorflow before continuing:")
    print("")
    print("https://github.com/pkmital/CADL#installation-preliminaries")

# This cell includes the provided libraries from the zip file
# and a library for displaying images from ipython, which
# we will use to display the gif
from libs import utils, gif, datasets, dataset_utils, vae, dft

class ScanFile(object):
    def __init__(self,directory,prefix=None,postfix='.jpg'):
        self.directory=directory
        self.prefix=prefix
        self.postfix=postfix

    def scan_files(self):
        files_list=[]

        for dirpath,dirnames,filenames in os.walk(self.directory):
            '''''
            dirpath is a string, the path to the directory.
            dirnames is a list of the names of the subdirectories in dirpath (excluding '.' and '..').
            filenames is a list of the names of the non-directory files in dirpath.
            '''
            for special_file in filenames:
                if self.postfix:
                    special_file.endswith(self.postfix)
                    files_list.append(os.path.join(dirpath,special_file))
                elif self.prefix:
                    special_file.startswith(self.prefix)
                    files_list.append(os.path.join(dirpath,special_file))
                else:
                    files_list.append(os.path.join(dirpath,special_file))

        return files_list

    def scan_subdir(self):
        subdir_list=[]
        for dirpath,dirnames,files in os.walk(self.directory):
            subdir_list.append(dirpath)
        return subdir_list

use_csv=True
if use_csv:
    files_img="/Users/yidawang/Documents/pynote/list_annotated_img.csv"
    files_obj="/Users/yidawang/Documents/pynote/list_annotated_obj.csv"
else:
    # Get a list of jpg file (Only JPG works!)
    image_dir = '/Users/yidawang/Documents/database/annotated_data/annotated_img'
    scan1=ScanFile(image_dir)
    files_img=scan1.scan_files()
    object_dir = '/Users/yidawang/Documents/database/annotated_data/annotated_obj'
    scan2=ScanFile(object_dir)
    files_obj=scan2.scan_files()
    assert len(files_obj) == len(files_img)
    print('Files assertion passed, ', len(files_img), 'training files in total')
input_shape = [100, 100, 3]
# files_img = [os.path.join(image_dir, file_i) for file_i in os.listdir(image_dir) if file_i.endswith('.jpg')]
# files_obj = [os.path.join(object_dir, file_i) for file_i in os.listdir(object_dir) if file_i.endswith('.jpg')]

# Train it!  Change these parameters!
tf.reset_default_graph()
vae.train_vae(files_img,
              files_obj,
              input_shape,
              use_csv=use_csv,
              learning_rate=0.0001,
              batch_size=64,
              n_epochs=50,
              n_examples=10,
              crop_shape=[95, 95, 3],
              crop_factor=1,
              n_filters=[100, 100, 100, 100],
              n_hidden=256,
              n_code=50,
              convolutional=True,
              variational=True,
              filter_sizes=[3, 3, 3, 3],
              dropout=True,
              keep_prob=0.8,
              activation=tf.nn.relu,
              img_step=100,
              save_step=100,
              ckpt_name="./vae.ckpt")
