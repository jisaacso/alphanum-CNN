import numpy as np
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import cPickle
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit

if __name__ == '__main__':
    #im = misc.imread('../training/0/999502.jpg', flatten=1)
    #im_small = block_mean(im, 10)
    #plt.imshow(im_small, cmap=cm.Greys_r)
    #plt.show()

    prefix = '../charset/'
    classes = os.listdir(prefix)
    samples = os.listdir(prefix + '48')

    f_dim = len(samples) * len(classes)
    labels = np.ndarray([f_dim])
    # labels = list()

    idx = 0
    for alphnum in classes:
        if alphnum == '.gitignore':
            continue
        for sample in os.listdir(prefix + alphnum):
            labels[idx] = alphnum # ord() names

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

    sss = StratifiedShuffleSplit(labels, n_iter=1, test_size=.2)
    sss_v = StratifiedShuffleSplit(labels, n_iter=1, test_size=.5)



    # generate x/y train, test and validate
    for train_index, rest_index in sss:
        x_train = features[train_index, :]
        x_rest = features[rest_index, :]
        y_train = labels[train_index]
        y_rest = labels[rest_index]

    for test_index, validate_index in sss_v:
        x_test = features[test_index, :]
        x_validate = features[validate_index, :]
        y_test = labels[test_index]
        y_validate = labels[validate_index]

        #x_train, x_rest, y_train, y_rest = train_test_split(
        #    features,labels, test_size=.2)
        #x_test, x_validate, y_test, y_validate = train_test_split(
        #    x_rest,y_rest, test_size=.5)

    print [chr(int(i)) for i in np.unique(y_train)]
    print [chr(int(i)) for i in np.unique(y_validate)]
    print [chr(int(i)) for i in np.unique(y_test)]


    datasets = [ [x_train, y_train], [x_validate, y_validate],
                 [x_test, y_test]]

    with open('alphanum.pkl', 'wb') as fout:
        cPickle.dump(datasets, fout)

