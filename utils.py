import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d
from scipy.ndimage import gaussian_filter
import os
from sklearn.decomposition import PCA as sklearn_pca
from scipy.ndimage.morphology import distance_transform_edt
import pandas as pd

## Function for Non-Local Means Method
def NonLocalMeans(image, d, verbose = False):
    ## Pad the image with mirror reflections of itself with a width d
    pad = (d, d)
    padimage = np.pad(image, (pad, pad, (0, 0)), mode='reflect')  # (top,bottom),(left,right),(0,0)

    ## For the ith pixel, make a (2d + 1) by (2d + 1) patch centered at pixel i
    patches = extract_patches_2d(padimage, (2 * d + 1, 2 * d + 1))

    ## For the jth, (j = 1; 2; 3) band, apply a Gaussian kernel on this patch
    u = np.zeros((2 * d + 1, 2 * d + 1))
    u[d, d] = 1
    G = gaussian_filter(u, d / 2, mode='constant', cval=0)
    patches = patches * G[np.newaxis, :, :, np.newaxis]
    if verbose:
        print(patches.shape)
    ## Form the feature matrix F by letting each row of F be a feature vector of a pixel
    F = patches.reshape((patches.shape[0], patches.shape[1] * patches.shape[2] * patches.shape[3]))
    if verbose:
        print("feature vector shape: ", F.shape)

    return F

# Function to perform PCA on image
def PCA(image, component=False):
    # Perform PCA with 60 components
    pca = sklearn_pca(n_components=60)
    X = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    pca.fit(X)
    variance = 100 * (pca.explained_variance_ratio_)

    # Get the number of components with variance greater than 0.005%
    num_components = len(variance[variance > 5e-3])

    # Perform PCA with the new number of components
    pca = sklearn_pca(n_components=num_components)
    pca_image = pca.fit_transform(X)
    print("Total Variation (%d components): " % num_components, np.sum(pca.explained_variance_ratio_))
    pca_image = pca_image.reshape(image.shape[0], image.shape[1], num_components)
    print("pca image shape: ", pca_image.shape)

    if component == True:
        return pca_image, num_components
    return pca_image

## uncertainty
def UQ_laplace(output_laplace, gt_labels, filename, Nbins=20, drop=0.05,
               make_hist=True, img_size=[256, 256], make_img=True,
               save_or_not=False, save_dir='UQ_plots'):
    normalized_u = output_laplace / np.sum(output_laplace, axis=1).reshape(-1, 1)
    labels_laplace = np.argmax(output_laplace, axis=1)

    confidence_score = np.sum(np.power(normalized_u - np.mean(normalized_u, axis=1).reshape(-1, 1), 2), axis=1) / 3

    if drop == 0:
        cut_score = confidence_score
        correct_ind = labels_laplace == gt_labels
    elif drop >= 0:
        sort_ind = np.argsort(confidence_score)
        N = len(confidence_score)
        cut_ind = sort_ind[np.round(N * drop).astype(int):-np.round(N * drop).astype(int)]
        cut_score = confidence_score[cut_ind]
        correct_ind = (labels_laplace == gt_labels)[cut_ind]

    M_conf = np.max(cut_score)
    m_conf = np.min(cut_score)

    bins = np.linspace(m_conf, M_conf + 1e-8, Nbins + 1)
    UQ_num = np.zeros(Nbins)
    UQ_accuracy = np.zeros(Nbins)
    UQ_indx = {}

    for i in range(Nbins):
        UQ_num[i] = np.sum(np.logical_and(cut_score >= bins[i], cut_score < bins[i + 1]))
        UQ_accuracy[i] = np.sum(np.logical_and(correct_ind,
                                               np.logical_and(cut_score >= bins[i], cut_score < bins[i + 1]))) / UQ_num[
                             i]
        UQ_indx[str(i)] = cut_ind[np.logical_and(cut_score >= bins[i], cut_score < bins[i + 1])]

    Class_names = np.unique(gt_labels)
    NClass = len(Class_names)
    accuracy = np.zeros([NClass + 1])
    g_accuracy = np.sum(labels_laplace == gt_labels) / len(gt_labels)
    accuracy[0] = g_accuracy
    for i in range(NClass):
        accuracy[i + 1] = np.sum(labels_laplace[gt_labels == Class_names[i]]
                                 == gt_labels[gt_labels == Class_names[i]]) / len(
            gt_labels[gt_labels == Class_names[i]])

    if make_hist:
        xvalues = (bins[:-1] + bins[1:]) / 2
        Wid = M_conf - m_conf + 1e-8

        plt.bar(xvalues, UQ_accuracy, width=Wid / Nbins)
        plt.xlabel('Confidence Score')
        plt.ylabel('Accuracy')
        plt.xticks(np.linspace(m_conf, M_conf, 11),
                   ['0', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
        plt.title('Global Accuracy: {:.2f}%'.format(g_accuracy * 100))
        if save_or_not:
            UQacc_save_dir = os.path.join(save_dir, filename[:-7] + 'uq_acc.png')
            plt.savefig(UQacc_save_dir)
        plt.show()

        plt.bar(xvalues, UQ_num, width=Wid / Nbins)
        plt.xlabel('Confidence Score')
        plt.ylabel('Num of Pixels')
        plt.xticks(np.linspace(m_conf, M_conf, 11),
                   ['0', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
        plt.title('Totel Num: {:d}'.format(int(len(cut_ind))))
        if save_or_not:
            UQnum_save_dir = os.path.join(save_dir, filename[:-7] + 'uq_num.png')
            plt.savefig(UQnum_save_dir)
        plt.show()

    if make_img:
        plt.imshow(confidence_score.reshape(img_size[0], img_size[1]))
        plt.colorbar()
        plt.title('Confidence Map')
        plt.show()

        plt.imshow(labels_laplace.reshape(img_size[0], img_size[1]))
        plt.colorbar()
        plt.title('Predicted Labels')
        plt.show()

    return UQ_num, UQ_accuracy, UQ_indx, accuracy, confidence_score

#split train/test sets
def train_test_split(data, names, trainpercent):
    '''
        data: dictionary
        names: list
        trainpercent: float in [0,1]
    '''

    for i in range(len(names)):
        label = data[names[i]]['label']
        image = data[names[i]]['image']
        filenames = np.array(data[names[i]]['filenames'])
        randperm = np.random.permutation(label.shape[0])
        N = np.round(label.shape[0] * trainpercent).astype(int)

        if i == 0:
            traindata = image[randperm[:N]]
            testdata = image[randperm[N:]]
            trainlabel = label[randperm[:N]]
            testlabel = label[randperm[N:]]
            trainfilenames = filenames[randperm[:N]]
            testfilenames = filenames[randperm[N:]]
        else:
            traindata = np.concatenate((traindata, image[randperm[:N]]), axis=0)
            testdata = np.concatenate((testdata, image[randperm[N:]]), axis=0)
            trainlabel = np.concatenate((trainlabel, label[randperm[:N]]), axis=0)
            testlabel = np.concatenate((testlabel, label[randperm[N:]]), axis=0)
            trainfilenames = np.append(trainfilenames, filenames[randperm[:N]])
            testfilenames = np.append(testfilenames, filenames[randperm[N:]])

    trainperm = np.random.permutation(traindata.shape[0])
    traindata = traindata[trainperm]
    trainlabel = trainlabel[trainperm]
    trainfilenames = trainfilenames[trainperm]
    testperm = np.random.permutation(testdata.shape[0])
    testdata = testdata[testperm]
    testlabel = testlabel[testperm]
    testfilenames = testfilenames[testperm]

    dic = {'train':{'label':trainlabel, 'image':traindata, 'filenames':trainfilenames},
           'test':{'label': testlabel, 'image':testdata, 'filenames':testfilenames}}

    return dic

## get neighborhood patch of each pixels
def neighbor_patchs(image, d, max_patches, random_state = 42, verbose = False):
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    ## Pad the image with mirror reflections of itself with a width d
    pad = (d, d)
    padimage = np.pad(image, ((0, 0), pad, pad, (0, 0)), mode='reflect')  # (0,0),(top,bottom),(left,right),(0,0)

    N = image.shape[0]
    for i in range(N):
        ## For the ith pixel, make a (2d + 1) by (2d + 1) patch centered at pixel i
        patches = extract_patches_2d(padimage[i], (2 * d + 1, 2 * d + 1), max_patches=max_patches, random_state=random_state)
        if i == 0:
            patches_all = patches
        else:
            patches_all = np.concatenate((patches_all, patches), axis=0)

    if verbose:
        print("extracted pathces shape: ", patches_all.shape)

    return patches_all

## extract features based on mean and std
def extract_features(patches, type = 'both', std_type = 2):
    '''
        patches are with the size (num_data, 2*d+1, 2*d+1, num_channels)
    '''

    d = int(patches.shape[1] / 2)

    sum_list = np.zeros([patches.shape[0], d + 2, patches.shape[-1]])
    std_list = np.zeros([patches.shape[0], d + 1, patches.shape[-1]])
    for i in range(d + 1):
        sum_list[:, i + 1, :] = np.sum(patches[:, d - i:d + i + 1, d - i:d + i + 1, :], axis=(1, 2))
        std_list[:, i, :] = np.std(patches[:, d - i:d + i + 1, d - i:d + i + 1, :], axis=(1, 2))

    neighbor_num = np.expand_dims(
        np.concatenate((np.array([1]), np.arange(3, 2 * d + 2, 2) ** 2 - np.arange(1, 2 * d, 2) ** 2)), axis=[0, -1])
    mean_list = (sum_list[:, 1:, :] - sum_list[:, :-1, :]) / neighbor_num

    std_list_2 = np.zeros([patches.shape[0], d, patches.shape[-1]])
    for i in range(1, d + 1):
        std_list_2[:, i - 1, :] = np.sqrt((np.sum((patches[:, d - i:d + i + 1, d - i:d + i + 1, :] -
                                                   np.expand_dims(mean_list[:, i, :], axis=[1, 2])) ** 2, axis=(1, 2))
                                           - np.sum((patches[:, d - i + 1:d + i, d - i + 1:d + i, :] -
                                                     np.expand_dims(mean_list[:, i, :], axis=[1, 2])) ** 2, axis=(1, 2)))
                                          / neighbor_num[:, i, :])

    if std_type == 1:
        std_feature = std_list.reshape(patches.shape[0], -1)
    elif std_type == 2:
        std_feature = std_list_2.reshape(patches.shape[0], -1)
    else:
        raise ValueError('std_type is either 1 or 2')
    mean_feature = mean_list.reshape(patches.shape[0], -1)

    if type == 'both':
        return np.concatenate((mean_feature, std_feature), axis=1)
    elif type == 'mean':
        return mean_feature
    elif type == 'std':
        return std_feature
    else:
        raise ValueError('type should be both, mean or std')

def low_dim_visualization(feature, labels, dim=2, class_names=['Land', 'Water', 'Sediment'], markersize=10, alpha=0.5,
                          tag='', save_or_not=False, save_path = 'ld_vis', specify_angle=None):

    class_values, counts = np.unique(labels, return_counts=True)
    pca = sklearn_pca(n_components=dim, svd_solver='arpack')
    num_class = len(class_values)

    pca_feature = pca.fit_transform(feature)
    evr = pca.explained_variance_ratio_

    if dim == 2:
        fig = plt.figure()
        ax = fig.add_subplot()
        for i in range(num_class):
            color = class_names[i] + ': ' + str(counts[i])
            ax.scatter(pca_feature[labels == class_values[i], 0], pca_feature[labels == class_values[i], 1],
                       label=color, alpha=alpha, s=markersize)

        ax.legend()
        ax.grid(True)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        plt.title('2D visualization')

        if save_or_not:
            plt.savefig(os.path.join(save_path, tag + '_2d.png'))

        plt.show()
    elif dim == 3:
        if specify_angle is not None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            for i in range(num_class):
                color = class_names[i]
                ax.scatter(pca_feature[labels == class_values[i], 0], pca_feature[labels == class_values[i], 1],
                           pca_feature[labels == class_values[i], 2], label=color, alpha=alpha, s=markersize)

            ax.legend()
            ax.grid(True)
            ax.view_init(azim=specify_angle[0], elev=specify_angle[1])
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            ax.axes.zaxis.set_ticks([])

            if save_or_not:
                plt.savefig(os.path.join(save_path, tag + 'specified angle' + '_3d.png'))

            plt.show()
        else:
            for view_ind in range(4):
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                for i in range(num_class):
                    color = class_names[i]
                    ax.scatter(pca_feature[labels == class_values[i], 0], pca_feature[labels == class_values[i], 1],
                               pca_feature[labels == class_values[i], 2], label=color, alpha=alpha, s=markersize)

                ax.legend()
                ax.grid(True)
                if view_ind != 0:
                    ax.view_init(azim=0, elev=90 * (view_ind - 1))
                ax.axes.xaxis.set_ticks([])
                ax.axes.yaxis.set_ticks([])
                ax.axes.zaxis.set_ticks([])

                if save_or_not:
                    plt.savefig(os.path.join(save_path, tag + str(view_ind) + '_3d.png'))

                plt.show()
    else:
        raise ValueError('dim should be either 2 or 3')

    return counts, evr


def output_accuracy(img_data, gt_labels, pred_labels, filenames,
                    make_plot=False, csv_name='results.csv',
                    image_prefix='sample_figs', simple_output=False,
                    image_save_folder='sample_images', rgb_normalize=None):
    class_names = np.unique(gt_labels)
    num_class = len(class_names)
    num_pixels = gt_labels.shape[1] * gt_labels.shape[2]

    class_correct_num = np.zeros([num_class, gt_labels.shape[0]])
    class_gt_num = np.zeros([num_class, gt_labels.shape[0]])
    class_FP_num = np.zeros([num_class, gt_labels.shape[0]])
    accuracy = np.zeros(gt_labels.shape[0])

    dist_3 = np.zeros([3, gt_labels.shape[0]])
    dist_10 = np.zeros([3, gt_labels.shape[0]])

    for image_ind in range(gt_labels.shape[0]):

        test_labels_pred = pred_labels[image_ind]
        test_labels_gt = gt_labels[image_ind]

        dist_map = np.zeros_like(test_labels_gt, dtype=np.float64)
        for i in range(num_class):
            ind = test_labels_gt == i
            pred_ind = test_labels_pred == i
            class_correct_num[i, image_ind] = np.sum(test_labels_pred[ind] == test_labels_gt[ind])
            class_FP_num[i, image_ind] = np.sum(test_labels_pred[pred_ind] != test_labels_gt[pred_ind])
            class_gt_num[i, image_ind] = np.sum(ind)
            dist_map += distance_transform_edt((test_labels_gt == i).astype(int))

        dist_3[0, image_ind] = np.sum(test_labels_pred[dist_map <= 3] == test_labels_gt[dist_map <= 3])
        dist_3[1, image_ind] = np.sum(dist_map <= 3)
        dist_3[2, image_ind] = dist_3[0, image_ind] / dist_3[1, image_ind]

        dist_10[0, image_ind] = np.sum(test_labels_pred[dist_map <= 10] == test_labels_gt[dist_map <= 10])
        dist_10[1, image_ind] = np.sum(dist_map <= 10)
        dist_10[2, image_ind] = dist_10[0, image_ind] / dist_10[1, image_ind]

        accuracy[image_ind] = np.sum(class_correct_num[:, image_ind]) / np.sum(class_gt_num[:, image_ind])

        if make_plot:
            (filepath, tempfilename) = os.path.split(filenames[image_ind])
            (filename, _) = os.path.splitext(tempfilename)

            img = img_data[image_ind]
            if rgb_normalize is None:
                modified_img = img[:, :, [2, 1, 0]] / np.max(img[:, :, [2, 1, 0]])
            else:
                modified_img = img[:, :, [2, 1, 0]] * rgb_normalize
                if np.max(modified_img > 1):
                    modified_img = modified_img / np.max(modified_img)
            fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
            ax0.imshow(modified_img)
            ax1.imshow(test_labels_pred)
            ax2.imshow(test_labels_gt)
            plt.imsave(os.path.join(image_save_folder, filename[:-7] + 'rgb.png'), modified_img)
            if num_class == 2:
                save_img_pred = test_labels_pred
                save_img_gt = test_labels_gt
                save_img_pred[0, 0] = 2
                save_img_gt[0, 0] = 2
                plt.imsave(os.path.join(image_save_folder, image_prefix + filename[:-7] + '_pred.png'),
                           save_img_pred)
                plt.imsave(os.path.join(image_save_folder, filename[:-7] + 'gt2.png'), save_img_gt)
            else:
                plt.imsave(os.path.join(image_save_folder, image_prefix + filename[:-7] + '_pred.png'),
                           test_labels_pred)
                plt.imsave(os.path.join(image_save_folder, filename[:-7] + 'gt.png'), test_labels_gt)
            plt.show()

    total_acc = np.sum(class_correct_num) / np.sum(class_gt_num)
    class_total_acc = np.sum(class_correct_num, axis=1) / np.sum(class_gt_num, axis=1)
    class_acc = class_correct_num / class_gt_num
    class_FPR = class_FP_num / (num_pixels - class_gt_num)
    class_total_FPR = np.sum(class_FP_num, axis=1) / np.sum(num_pixels - class_gt_num, axis=1)

    class_num_ratio = class_gt_num / num_pixels
    nor_class_FPR = class_FPR / class_num_ratio
    class_total_NFPR = class_total_FPR / np.sum(class_gt_num, axis=1) * np.sum(class_gt_num)

    sum_dist_3 = np.sum(dist_3, axis=1)
    sum_dist_10 = np.sum(dist_10, axis=1)

    if num_class == 2:
        df = pd.DataFrame.from_dict({"Land TPR": [f"{num * 100:.2f}" for num in class_acc[0, :]],
                                     "Land FPR": [f"{num * 100:.2f}" for num in class_FPR[0, :]],
                                     "Land NFPR": [f"{num * 100:.2f}" for num in nor_class_FPR[0, :]],
                                     "Water TPR": [f"{num * 100:.2f}" for num in class_acc[1, :]],
                                     "Water FPR": [f"{num * 100:.2f}" for num in class_FPR[1, :]],
                                     "Water NFPR": [f"{num * 100:.2f}" for num in nor_class_FPR[1, :]],
                                     "Dist 3": [f"{num * 100:.2f}" for num in dist_3[2, :]],
                                     "Dist 10": [f"{num * 100:.2f}" for num in dist_10[2, :]],
                                     "Whole": [f"{num * 100:.2f}" for num in accuracy]}, orient='index',
                                     columns=[i for i in range(class_acc.shape[1])])
        df['Total'] = [f"{num * 100:.2f}" for num in [class_total_acc[0], class_total_FPR[0], class_total_NFPR[0],
                                                      class_total_acc[1], class_total_FPR[1], class_total_NFPR[1],
                                                      sum_dist_3[0] / sum_dist_3[1],
                                                      sum_dist_10[0] / sum_dist_10[1], total_acc]]
    elif num_class == 3:
        df = pd.DataFrame.from_dict({"Land TPR": [f"{num * 100:.2f}" for num in class_acc[0, :]],
                                     "Land FPR": [f"{num * 100:.2f}" for num in class_FPR[0, :]],
                                     "Land NFPR": [f"{num * 100:.2f}" for num in nor_class_FPR[0, :]],
                                     "Water TPR": [f"{num * 100:.2f}" for num in class_acc[1, :]],
                                     "Water FPR": [f"{num * 100:.2f}" for num in class_FPR[1, :]],
                                     "Water NFPR": [f"{num * 100:.2f}" for num in nor_class_FPR[1, :]],
                                     "Sediment TPR": [f"{num * 100:.2f}" for num in class_acc[2, :]],
                                     "Sediment FPR": [f"{num * 100:.2f}" for num in class_FPR[2, :]],
                                     "Sediment NFPR": [f"{num * 100:.2f}" for num in nor_class_FPR[2, :]],
                                     "Dist 3": [f"{num * 100:.2f}" for num in dist_3[2, :]],
                                     "Dist 10": [f"{num * 100:.2f}" for num in dist_10[2, :]],
                                     "Whole": [f"{num * 100:.2f}" for num in accuracy]}, orient='index',
                                     columns=[i for i in range(class_acc.shape[1])])
        df['Total'] = [f"{num * 100:.2f}" for num in [class_total_acc[0], class_total_FPR[0], class_total_NFPR[0],
                                                      class_total_acc[1], class_total_FPR[1], class_total_NFPR[1],
                                                      class_total_acc[2], class_total_FPR[2], class_total_NFPR[2],
                                                      sum_dist_3[0] / sum_dist_3[1],
                                                      sum_dist_10[0] / sum_dist_10[1], total_acc]]

    if simple_output:
        df = df.iloc[:, -1]

    df.to_csv(csv_name)

    return class_acc, class_FPR, dist_3, dist_10, accuracy, df


