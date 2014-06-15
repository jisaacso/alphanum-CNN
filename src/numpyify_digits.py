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
        print 'classname is %s' %alphnum

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
            #if idx % 1000 == 0:
            #    print idx


    r = np.random.rand(len(labels))
    ind1 = np.where(r <= .1)[0]
    ind2 = np.where( (r > .1) & (r <= .2))[0]
    ind3 = np.where(r > .2)[0]

    x_train = features[ind3, :]
    y_train = labels[ind3]
    x_test = features[ind2, :]
    y_test = labels[ind2]
    x_validate = features[ind1, :]
    y_validate = labels[ind1]

    print [chr(int(i)) for i in np.unique(y_train)]
    print [chr(int(i)) for i in np.unique(y_validate)]
    print [chr(int(i)) for i in np.unique(y_test)]


    datasets = [ [x_train, y_train], [x_validate, y_validate],
                 [x_test, y_test]]

    with open('alphanum.pkl', 'wb') as fout:
        cPickle.dump(datasets, fout)

