import numpy as np
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import cPickle
from sklearn.cross_validation import train_test_split
if __name__ == '__main__':
    #im = misc.imread('../training/0/999502.jpg', flatten=1)
    #im_small = block_mean(im, 10)
    #plt.imshow(im_small, cmap=cm.Greys_r)
    #plt.show()

    prefix = '../training/'
    classes = os.listdir(prefix)
    samples = os.listdir(prefix + '0')

    labels = np.ndarray([len(samples) * len(classes)])

    idx = 0
    for alphnum in classes:
        for sample in os.listdir(prefix + alphnum):
            labels[idx] = alphnum

            im = misc.imread(prefix + alphnum + '/' + sample, flatten=1).\
                flatten()

            assert len(im) == 784, \
                "image %s is not 28x28 pixels" % idx

            if idx == 0:
                features = np.ndarray([labels.shape[0], len(im)])

            features[idx, :] = im

            idx += 1
            if idx % 1000 == 0:
                print idx

        with open('alphnum.pkl', 'wb') as fout:
            cPickle.dump(features, fout)

    # generate x/y train, test and validate
    x_train, x_rest, y_train, y_rest = train_test_split(
        features,labels, test_size=.2)
    x_test, x_validate, y_test, y_validate = train_test_split(
        x_rest,y_rest, test_size=.5)

    datasets = [ [x_train, y_train], [x_validate, y_validate],
                 [x_test, y_test]]

    with open('alphanum.pkl', 'wb') as fout:
        cPickle.dump(datasets, fout)

