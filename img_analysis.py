import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from skimage import data
from scipy.misc import imresize
import IPython.display as ipyd


from libs import utils, gif, datasets, dataset_utils, vae, dft

class ScanFile(object):
    def __init__(self,directory,prefix=None,postfix=None):
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

if __name__=="__main__":
	dir=r"/Users/yidawang/Documents/database/PASCAL3D+_release1.1/real_annotated/PASCAL"
	scan=ScanFile(dir)
	subdirs=scan.scan_subdir()
	for subdir in subdirs[1:]:
		print('Current subdir scaned is: '+subdir)
		sub_scan=ScanFile(subdir)
		files=sub_scan.scan_files()
		myown_imgs = [plt.imread(f_i) for f_i in files]

		# Then resize the square image to 100 x 100 pixels
		myown_imgs = [resize(img_i, (100, 100, 3)) for img_i in myown_imgs]
		plt.figure(figsize=(10, 10))

		imgs = np.array(myown_imgs).copy()*255


		# Then convert the list of images to a 4d array (e.g. use np.array to convert a list to a 4d array):
		Xs = imgs

		print(Xs.shape)
		assert(Xs.ndim == 4 and Xs.shape[1] <= 100 and Xs.shape[2] <= 100)

		ds = datasets.Dataset(Xs)
		# ds = datasets.CIFAR10(flatten=False)

		mean_img = ds.mean().astype(np.float32)
		name='./mean_std/pascal_mean_'+subdir.split('/')[-1]+'.png'
		plt.imsave(name, mean_img.astype(np.uint8))

		std_img = ds.std().astype(np.float32)
		plt.imshow(std_img.astype(np.uint8))

		std_img = np.mean(std_img, axis=2).astype(np.float32)
		name='./mean_std/pascal_std_'+subdir.split('/')[-1]+'.png'
		plt.imsave(name, std_img.astype(np.uint8))
	os.system('montage ./mean_std/shapenet*.png ./mean_std/imagenet*.png ./mean_std/pascal*.png -geometry +1+1 -tile 12x6 montage.png | bash /Users/yidawang/Documents/buildboat/shnote/imgcat.sh ./montage.png')
