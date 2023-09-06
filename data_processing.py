import matplotlib.pyplot as plt
import scipy.io
import os
import sys
import shutil
import pandas as pd
from itertools import permutations

import matplotlib.ticker as ticker
from matplotlib.patches import Arrow, Circle

# custom
import batch_active_learning as bal
import time
from SSU import *

def select_indices(labels, n, seed=42):
    rng = np.random.default_rng(seed)
    unique_labels = np.unique(labels)
    selected_indices = []

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        if isinstance(n, int):
            num_to_select = min(n, len(label_indices))
        elif isinstance(n, float) and 0.0 <= n <= 1.0:
            num_to_select = int(len(label_indices) * n)
        else:
            raise ValueError("n must be an integer or a float between 0 and 1")
        selected_indices.extend(rng.choice(label_indices, size=num_to_select, replace=False))
    return np.array(selected_indices).astype(int)

def find_perm(A_true, A_hat):
    P = A_true.shape[0]
    ords = list(permutations(range(P)))
    n = len(ords)
    errs = 100 * np.ones(n)

    for idx in range(n):
        A = A_hat[ords[idx], :]
        errs[idx] = RMSE_new(A_true, A)

    I = np.argmin(errs)
    indx = ords[I]

    return indx

def add_noise_to_image(image, snr_db):
    """Add Gaussian white noise to the given image.

    Parameters:
    image (np.ndarray): Input image as a 2D numpy array (channels x pixels).
    snr_db (float): Desired signal-to-noise ratio in dB.

    Returns:
    np.ndarray: Image with added noise.
    """
    # Calculate signal power and convert signal power to dB
    signal_power = np.mean(image ** 2, axis=1)[:, np.newaxis]
    signal_power_db = 10 * np.log10(signal_power)

    # Calculate noise power based on desired SNR
    noise_power_db = signal_power_db - snr_db
    noise_power = 10 ** (noise_power_db / 10)

    # Generate Gaussian white noise
    mean = 0
    std_dev = np.sqrt(noise_power)
    noise = np.random.normal(mean, std_dev, size=image.shape)

    # Add noise to the image
    noisy_image = image + noise

    return noisy_image

def get_indices(A, threshold):
    indices = np.where(A > threshold)[0]
    indices = indices[np.argsort(A[indices])[::-1]]
    return indices

def graph_learning(feature_vectors, label_inds, labels, similarity='angular'):
    knn_data = gl.weightmatrix.knnsearch(feature_vectors, 30, method='annoy', similarity=similarity)
    W = gl.weightmatrix.knn(None, 50, kernel = 'gaussian', knn_data=knn_data)
    model = gl.ssl.laplace(W)
    pred_labels = model.fit_predict(label_inds, labels)

    return pred_labels

def process_dataset(dataset, datafolder='data', save_result=True, save_folder='processed_data', copy_files=False):
    dir_path = os.path.join(save_folder, dataset)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    datafolder = os.path.join(datafolder, dataset)
    output_file = os.path.join(datafolder, dataset + '_output.mat')
    msc_file = 'MSC_results.mat'

    if dataset == 'jasper':
        h_data_file = 'jasperRidge2_R198.mat'
        ref_data_file = 'end4.mat'
    elif dataset == 'urban':
        h_data_file = 'Urban_R162.mat'
        ref_data_file = 'end4_groundTruth.mat'
    elif dataset == 'samson':
        h_data_file = 'samson_1.mat'
        ref_data_file = 'end3.mat'
    elif dataset == 'apex':
        h_data_file = 'Y_clean.mat'
        ref_data_file = 'A_true.mat'
    else:
        sys.exit("Invalid dataset: " + dataset)

    h_data = scipy.io.loadmat(os.path.join(datafolder, h_data_file))

    if dataset == 'jasper' or dataset == 'urban':
        n = h_data['nRow'].item()
        m = h_data['nCol'].item()
        X = h_data['Y']/h_data['maxValue']

        ref_data = scipy.io.loadmat(os.path.join(datafolder, ref_data_file))
        A_ref = ref_data['A']
        S_ref = ref_data['M']
    elif dataset == 'samson':
        n = h_data['nRow'].item()
        m = h_data['nCol'].item()
        X = h_data['V']
        ref_data = scipy.io.loadmat(os.path.join(datafolder, ref_data_file))
        A_ref = ref_data['A']
        S_ref = ref_data['M']
    elif dataset == 'apex':
        X = h_data['Y_clean']
        n = X.shape[1]
        m = X.shape[0]
        X = np.transpose(X,(2,1,0)).reshape(-1, n*m)
        A_ref = scipy.io.loadmat(os.path.join(datafolder, ref_data_file))['A_true']
        A_ref = np.transpose(A_ref, (2,1,0)).reshape(-1, m*n)
        S_ref = scipy.io.loadmat(os.path.join(datafolder, 'E.mat'))['E']
    else:
        sys.exit("Invalid dataset: " + dataset)

    results = scipy.io.loadmat(output_file)

    A_fclsu = results.get('A_init', None)
    S_fclsu = results.get('S_init', None)
    A_MBO = results.get('A_MBO', None)
    A_graphL = results.get('A_graphL', None)
    S_MBO = results.get('S_MBO', None)
    S_graphL = results.get('S_graphL', None)
    A_qmv = results.get('A_qmv', None)
    S_qmv = results.get('S_qmv', None)
    A_nmf = results.get('A_nmf', None)
    S_nmf = results.get('S_nmf', None)
    err_all = results.get('errs', None)

    msc_results = scipy.io.loadmat(os.path.join(datafolder, msc_file))
    A_MSC = msc_results['out_avg_np']
    S_MSC = msc_results['Eest']
    t_MSC = msc_results['time'][0][0]

    EGU_results = np.load(os.path.join(datafolder, f"{dataset}_EGU_VCA.npy"), allow_pickle=True).item()
    A_EGU = EGU_results["A"].T
    S_EGU = EGU_results["S"]
    t_EGU = EGU_results["time"][0][0]

    # if dataset == "jasper":
    #     A_MSC = A_MSC.transpose(2,1,0).reshape(-1, m*n)[[0, 3, 1, 2]]
    #     S_MSC = S_MSC[:,[0, 3, 1, 2]]
    # elif dataset == "urban":
    #     A_MSC = A_MSC.transpose(2, 1, 0).reshape(-1, m * n)[[2, 0, 3, 1]]
    #     S_MSC = S_MSC[:, [2, 0, 3, 1]]
    # elif dataset == "samson":
    #     A_MSC = A_MSC.transpose(2, 1, 0).reshape(-1, m * n)[::-1]
    #     S_MSC = S_MSC[:, ::-1]
    # elif dataset == 'apex':
    #     A_MSC = A_MSC.transpose(2, 1, 0).reshape(-1, m * n)[[0, 3, 2, 1]]
    #     S_MSC = S_MSC[:, [0, 3, 2, 1]]
    A_MSC = A_MSC.transpose(2, 1, 0).reshape(-1, m * n)
    perm_indx = find_perm(A_ref, A_MSC)
    A_MSC = A_MSC[np.array(perm_indx)]
    S_MSC = S_MSC[:, np.array(perm_indx)]

    result_dict = {
        'A': {
            'A_ref': A_ref,
            'A_fclsu': A_fclsu,
            'A_MBO': A_MBO,
            'A_graphL': A_graphL,
            'A_MSC': A_MSC,
            'A_qmv': A_qmv,
            'A_nmf': A_nmf,
            'A_EGU': A_EGU,
        },
        'S': {
            'S_ref': S_ref,
            'S_fclsu': S_fclsu,
            'S_MBO': S_MBO,
            'S_graphL': S_graphL,
            'S_MSC': S_MSC,
            'S_qmv': S_qmv,
            'S_nmf': S_nmf,
            'S_EGU': S_EGU,
        },
        'err_all': err_all,
        'X': X,
        'dataset': dataset,
        'shape': (n, m),
        't_MSC': t_MSC,
        't_EGU': t_EGU,
    }

    if save_result:
        np.save(os.path.join(dir_path, f"{dataset}_processed_data.npy"), result_dict)

        if copy_files:
            shutil.copy2(os.path.join(datafolder, h_data_file), dir_path)
            shutil.copy2(os.path.join(datafolder, ref_data_file), dir_path)
            shutil.copy2(os.path.join(datafolder, msc_file), dir_path)
            shutil.copy2(output_file, dir_path)

    return result_dict


def get_training_labels(dataset, train_percentage=0.4, save_data=True, load_folder="processed_data", save_folder="processed_data"):
    # dataset can be a dict or a str for the dataset
    if isinstance(dataset, dict):
        result_dict = dataset
        dataset = result_dict['dataset']
    elif isinstance(dataset, str):
        result_dict = np.load(os.path.join(os.path.join(load_folder, dataset), f"{dataset}_processed_data.npy"), allow_pickle=True).item()
    else:
        sys.exit("Invalid dataset type: " + str(type(dataset)))

    X = result_dict['X']
    A_MBO_fixed = result_dict['A']['A_MBO']
    S_MBO_fixed = result_dict['S']['S_MBO']
    A_ref = result_dict['A']['A_ref']

    pseudo_labels = np.argmax(A_ref, axis=0)
    class_names = np.unique(pseudo_labels)
    num_class = len(class_names)

    num_sample = int(train_percentage * X.shape[1] / 100)
    print(num_sample)

    uc_val = uc_unmixing(X, A_MBO_fixed, S_MBO_fixed)
    MBO_pred_labels = np.argmax(A_MBO_fixed, axis=0)

    # confidence sample
    train_cfd_indx = confidence_selection(uc_val, MBO_pred_labels, 3 * int(num_sample/num_class), num_class)
    cfd_labels = MBO_pred_labels[train_cfd_indx]

    feature_vectors = X.T
    knn_data = gl.weightmatrix.knnsearch(feature_vectors, 50, method='annoy', similarity='angular')
    W = gl.weightmatrix.knn(feature_vectors, 50, kernel='gaussian', knn_data=knn_data)
    G = gl.graph(W)

    np.random.seed(0)
    initial = gl.trainsets.generate(pseudo_labels, rate=1).tolist()

    coreset = bal.coreset_dijkstras(G, rad=2, DEBUGGING=False, data=X.T, initial=initial)
    pp_coreset = bal.plusplus_coreset(G, num_points=num_sample, random_state=None, method='dijkstra', eik_p=1.0, tau=0.1,
                                      ofs=0.2, q=1.0, knn_dist=None, kernel='gaussian', plot=False, X=X.T,
                                      initial=initial)

    np.random.seed(0)
    initial = gl.trainsets.generate(pseudo_labels, rate=1).tolist()

    if num_sample > 300:
        al_mtd = "local_max"
        batch_size = 10
        num_iter = int(num_sample / batch_size)
    else:
        al_mtd = "global_max"
        batch_size = 1
        num_iter = num_sample

    acq_fun = 'uc'
    uc_ind, _, _ = bal.coreset_run_experiment(X.T, pseudo_labels, W, initial, num_iter=num_iter, method='Laplace',
                                              display=False, use_prior=False, al_mtd=al_mtd, debug=False,
                                              acq_fun=acq_fun, knn_data=knn_data, mtd_para=(np.inf, 0, 'new'),
                                              savefig=False, savefig_folder='../BAL_figures', batchsize=batch_size,
                                              dist_metric='angular', knn_size=50, q=1, thresholding=0, randseed=0,
                                              dropout=0)

    acq_fun = 'mc'
    mc_ind, _, _ = bal.coreset_run_experiment(X.T, pseudo_labels, W, initial, num_iter=num_iter, method='Laplace',
                                              display=False, use_prior=False, al_mtd=al_mtd, debug=False,
                                              acq_fun=acq_fun, knn_data=knn_data, mtd_para=(np.inf, 0, 'new'),
                                              savefig=False, savefig_folder='../BAL_figures', batchsize=batch_size,
                                              dist_metric='angular', knn_size=50, q=1, thresholding=0, randseed=0,
                                              dropout=0)

    acq_fun = 'vopt'
    vopt_ind, _, _ = bal.coreset_run_experiment(X.T, pseudo_labels, W, initial, num_iter=num_iter, method='Laplace',
                                                display=False, use_prior=False, al_mtd=al_mtd, debug=False,
                                                acq_fun=acq_fun, knn_data=knn_data, mtd_para=(np.inf, 0, 'new'),
                                                savefig=False, savefig_folder='../BAL_figures', batchsize=batch_size,
                                                dist_metric='angular', knn_size=50, q=1, thresholding=0, randseed=0,
                                                dropout=0)

    acq_fun = 'mcvopt'
    mcvopt_ind, _, _ = bal.coreset_run_experiment(X.T, pseudo_labels, W, initial, num_iter=num_iter, method='Laplace',
                                                  display=False, use_prior=False, al_mtd=al_mtd, debug=False,
                                                  acq_fun=acq_fun, knn_data=knn_data, mtd_para=(np.inf, 0, 'new'),
                                                  savefig=False, savefig_folder='../BAL_figures', batchsize=batch_size,
                                                  dist_metric='angular', knn_size=50, q=1, thresholding=0, randseed=0,
                                                  dropout=0)

    data_dic = {
        'dataset': dataset,
        'train_cfd_indx': train_cfd_indx,
        'cfd_labels': cfd_labels,
        'coreset': coreset,
        'pp_coreset': pp_coreset,
        'uc_ind': uc_ind,
        'mc_ind': mc_ind,
        'vopt_ind': vopt_ind,
        'mcvopt_ind': mcvopt_ind,
    }

    if save_data:
        np.save(os.path.join(save_folder, dataset, "train_inds.npy"), data_dic)
    return data_dic

def process_image_GL(dataset, labels_type='al_vopt', save_data=True, data_folder='processed_data'):
    if isinstance(dataset, tuple):
        result_dict, data_dict = dataset
        dataset = result_dict['dataset']
    elif isinstance(dataset, str):
        result_dict = np.load(os.path.join(data_folder, dataset, f"{dataset}_processed_data.npy"),
                              allow_pickle=True).item()
        data_dict = np.load(os.path.join(data_folder, dataset, "train_inds.npy"),
                              allow_pickle=True).item()
    else:
        sys.exit("Invalid dataset type: " + str(type(dataset)))
    n, m = result_dict['shape']
    X = result_dict['X']
    A_ref = result_dict['A']['A_ref']

    pseudo_labels = np.argmax(A_ref, axis=0)
    class_names = np.unique(pseudo_labels)
    num_class = len(class_names)

    img_shape = (n, m, X.shape[0])
    A_GL = np.zeros((2,num_class,n*m))
    S_GL = np.zeros((2,X.shape[0],num_class))

    if labels_type=='cfd':
        label_inds = data_dict['train_cfd_indx']
        ref_labels = data_dict['cfd_labels']
    elif labels_type=='coreset':
        label_inds = data_dict['coreset']
        ref_labels = pseudo_labels[data_dict['coreset']]
    elif labels_type=='pp_coreset':
        label_inds = data_dict['pp_coreset']
        ref_labels = pseudo_labels[data_dict['pp_coreset']]
    elif labels_type=='al_uc':
        label_inds = data_dict['uc_ind']
        ref_labels = pseudo_labels[data_dict['uc_ind']]
    elif labels_type=='al_mc':
        label_inds = data_dict['mc_ind']
        ref_labels = pseudo_labels[data_dict['mc_ind']]
    elif labels_type=='al_vopt':
        label_inds = data_dict['vopt_ind']
        ref_labels = pseudo_labels[data_dict['vopt_ind']]
    elif labels_type=='al_mcvopt':
        label_inds = data_dict['mcvopt_ind']
        ref_labels = pseudo_labels[data_dict['mcvopt_ind']]
    else:
        sys.exit("Invalid dataset: " + labels_type)

    X_labeled = X[:, label_inds]
    A_labeled_onehot = np.zeros((num_class, len(label_inds)))
    for i in range(num_class):
        A_labeled_onehot[i, ref_labels==i] = 1
    A_labeled_exact = A_ref[:, label_inds]

    t = time.time()
    A_GL[0] = GL_unmixing(X, label_inds, ref_labels, img_shape, d=0)
    S_GL[0] = gl_fitS(X_labeled, A_labeled_onehot)
    t1 = time.time() - t

    t = time.time()
    exact_amap = A_ref[:, label_inds].T
    A_GL[1] = GL_unmixing(X, label_inds, exact_amap, img_shape, d=0, use_exact_amap=True)
    S_GL[1] = gl_fitS(X_labeled,A_labeled_exact)
    t2 = time.time() - t

    GL_dic = {
        'A_GL': A_GL,
        'S_GL': S_GL,
        'time': [t1, t2],
        'ref_labels': ref_labels,
        'label_inds': label_inds,
    }

    if save_data:
        np.save(os.path.join(data_folder, dataset, "GL_result.npy"), GL_dic)
    return GL_dic


def process_image_GRSU(dataset, labels_type='al_vopt', use_new_init=True, para=None, alpha=None,
                       save_data=True, data_folder='processed_data'):
    if isinstance(dataset, tuple):
        result_dict, data_dict = dataset
        dataset = result_dict['dataset']
    elif isinstance(dataset, str):
        result_dict = np.load(os.path.join(data_folder, dataset, f"{dataset}_processed_data.npy"),
                              allow_pickle=True).item()
        data_dict = np.load(os.path.join(data_folder, dataset, "train_inds.npy"),
                            allow_pickle=True).item()
    else:
        sys.exit("Invalid dataset type: " + str(type(dataset)))

    tol = 1e-3
    if para is None:
        if dataset[:6] == 'jasper':
            para = (200, 1e0, 1e0, 1e0, 'GL_laplace_ref', tol)
            alpha = (1e1)**2
        elif dataset[:5] == 'urban':
            para = (100, 5e2, 1e-1, 1e-1, 'GL_laplace_ref', tol)
            alpha = (5e1)**2
        elif dataset[:6] == 'samson':
            para = (200, 5e1, 1e-1, 1e-1, 'GL_laplace_ref', tol)
            alpha = (2e1)**2
        elif dataset[:4] == 'apex':
            para = (200, 5e1, 1e0, 1e0, 'GL_laplace_ref', tol)
            alpha = (1e1)**2
        else:
            sys.exit("Invalid dataset type: " + str(type(dataset)))

    n, m = result_dict['shape']
    X = result_dict['X']
    A_ref = result_dict['A']['A_ref']
    A_fclsu = result_dict['A']['A_fclsu']
    S_fclsu = result_dict['S']['S_fclsu']
    A_MBO_fixed = result_dict['A']['A_MBO']

    pseudo_labels = np.argmax(A_ref, axis=0)
    class_names = np.unique(pseudo_labels)
    num_class = len(class_names)

    A_GRSU = np.zeros((2,num_class,n*m))
    S_GRSU = np.zeros((2,X.shape[0],num_class))

    if labels_type=='cfd':
        label_inds = data_dict['train_cfd_indx']
        ref_labels = data_dict['cfd_labels']
    elif labels_type=='coreset':
        label_inds = data_dict['coreset']
        ref_labels = pseudo_labels[data_dict['coreset']]
    elif labels_type=='pp_coreset':
        label_inds = data_dict['pp_coreset']
        ref_labels = pseudo_labels[data_dict['pp_coreset']]
    elif labels_type=='al_uc':
        label_inds = data_dict['uc_ind']
        ref_labels = pseudo_labels[data_dict['uc_ind']]
    elif labels_type=='al_mc':
        label_inds = data_dict['mc_ind']
        ref_labels = pseudo_labels[data_dict['mc_ind']]
    elif labels_type=='al_vopt':
        label_inds = data_dict['vopt_ind']
        ref_labels = pseudo_labels[data_dict['vopt_ind']]
    elif labels_type=='al_mcvopt':
        label_inds = data_dict['mcvopt_ind']
        ref_labels = pseudo_labels[data_dict['mcvopt_ind']]
    else:
        sys.exit("Invalid dataset: " + labels_type)

    X_labeled = X[:, label_inds]
    A_labeled_onehot = np.zeros((num_class, len(label_inds)))
    for i in range(num_class):
        A_labeled_onehot[i, ref_labels==i] = 1

    if not use_new_init:
        A_init = A_fclsu
        S_init = S_fclsu
    else:
        if labels_type=='cfd':
            S_init = gl_fitS(X_labeled, A_MBO_fixed[:, label_inds])
            A_init = init_fitA(X, S_init)
        else:
            S_init = gl_fitS(X_labeled, A_labeled_onehot)
            A_init = init_fitA(X, S_init)

    t = time.time()
    s1,a1,b1,it = GRSU_unmixing(X, S_init, A_init, label_inds, ref_labels, para, r=1e-1, alpha=alpha, A_ref=A_ref)
    S_GRSU[0] = s1
    A_GRSU[0] = a1
    t1 = time.time() - t

    t = time.time()
    if labels_type=='cfd':
        exact_amap = A_MBO_fixed[:, label_inds].T
    else:
        exact_amap = A_ref[:, label_inds].T
    s2,a2,b2,it1 = GRSU_unmixing(X, S_init, A_init, label_inds, exact_amap, para, r=1e-1, alpha=alpha, A_ref=A_ref, use_exact_amap=True)
    S_GRSU[1] = s2
    A_GRSU[1] = a2
    t2 = time.time() - t

    T = [t1, t2]
    iters = [it, it1]

    GRSU_dic = {
        'A_GRSU': A_GRSU,
        'S_GRSU': S_GRSU,
        'time': T,
        'iter': iters,
        'ref_labels': ref_labels,
        'label_inds': label_inds,
    }

    if save_data:
        np.save(os.path.join(data_folder, dataset, "GRSU_result.npy"), GRSU_dic)
    return GRSU_dic


############## output results
def output_results(dataset, save_data=True, data_folder='processed_data', dpi_vals=(200,300)):
    if isinstance(dataset, tuple):
        result_dict, data_dict, GL_dic, GRSU_dic = dataset
        dataset = result_dict['dataset']
    elif isinstance(dataset, str):
        result_dict = np.load(os.path.join(data_folder, dataset, f"{dataset}_processed_data.npy"),
                              allow_pickle=True).item()
        GL_dic = np.load(os.path.join(data_folder, dataset, "GL_result.npy"),
                            allow_pickle=True).item()
        GRSU_dic = np.load(os.path.join(data_folder, dataset, "GRSU_result.npy"),
                         allow_pickle=True).item()
    else:
        sys.exit("Invalid dataset type: " + str(type(dataset)))

    #method_names = ["FCLSU", "Fractional", "Sunsal-TV", ""GLNMF", "NMF_QMV", "Graph_L", "gtvMBO", "gtvMBO fixed ratio"]

    A_ref = result_dict['A']['A_ref']
    S_ref = result_dict['S']['S_ref']

    pseudo_labels = np.argmax(A_ref, axis=0)
    class_names = np.unique(pseudo_labels)
    num_class = len(class_names)
    n, m = result_dict['shape']

    A_fclsu = result_dict['A']['A_fclsu']
    S_fclsu = result_dict['S']['S_fclsu']
    t_fclsu = result_dict['err_all'][8, 0]

    A_MBO = result_dict['A']['A_MBO']
    S_MBO = result_dict['S']['S_MBO']
    t_MBO = result_dict['err_all'][8, 6]

    A_MSC = result_dict['A']['A_MSC']
    S_MSC = result_dict['S']['S_MSC']
    t_MSC = result_dict['t_MSC']

    A_nmf = result_dict['A']['A_nmf']
    S_nmf = result_dict['S']['S_nmf']
    t_nmf = result_dict['err_all'][8, 3]

    A_qmv = result_dict['A']['A_qmv']
    S_qmv = result_dict['S']['S_qmv']
    t_qmv = result_dict['err_all'][8, 4]

    A_EGU = result_dict['A']['A_EGU']
    S_EGU = result_dict['S']['S_EGU']
    t_EGU = result_dict['t_EGU']

    A_GL = GL_dic['A_GL']
    S_GL = GL_dic['S_GL']
    t_GL = GL_dic['time']

    A_GRSU = GRSU_dic['A_GRSU']
    S_GRSU = GRSU_dic['S_GRSU']
    t_GRSU = GRSU_dic['time']

    ref_labels = GL_dic['ref_labels']
    label_inds =GL_dic['label_inds']

    if dataset == "jasper":
        label_names = ['Tree', 'Water', 'Dirt', 'Road']
    elif dataset == "urban":
        label_names = ['Asphalt', 'Grass', 'Tree', 'Roof']
    elif dataset == "samson":
        label_names = ['Soil', 'Tree', 'Water']
    elif dataset == "apex":
        label_names = ['Road', 'Tree', 'Roof', 'Water']

    err_all = np.zeros((9, 2 * num_class + 2))

    err_all[0] = class_metrics(A_nmf, S_nmf, A_ref, S_ref)
    err_all[1] = class_metrics(A_qmv, S_qmv, A_ref, S_ref)
    err_all[2] = class_metrics(A_MBO, S_MBO, A_ref, S_ref)
    err_all[3] = class_metrics(A_MSC, S_MSC, A_ref, S_ref)
    err_all[4] = class_metrics(A_EGU, S_EGU, A_ref, S_ref)
    err_all[5] = class_metrics(A_GL[0], S_GL[0], A_ref, S_ref)
    err_all[6] = class_metrics(A_GL[1], S_GL[1], A_ref, S_ref)
    err_all[7] = class_metrics(A_GRSU[0], S_GRSU[0], A_ref, S_ref)
    err_all[8] = class_metrics(A_GRSU[1], S_GRSU[1], A_ref, S_ref)

    column_names = ['RMSE'] + label_names + ['SAD'] + label_names

    df_dic = {}
    row_name = 'GLNMF'
    df_dic[row_name] = [f"{num:.2f}" for num in err_all[0, :]]
    row_name = 'QMV'
    df_dic[row_name] = [f"{num:.2f}" for num in err_all[1, :]]
    row_name = 'GTVMBO'
    df_dic[row_name] = [f"{num:.2f}" for num in err_all[2, :]]
    row_name = 'MSC'
    df_dic[row_name] = [f"{num:.2f}" for num in err_all[3, :]]
    row_name = 'EGU'
    df_dic[row_name] = [f"{num:.2f}" for num in err_all[4, :]]
    row_name = 'GLU_OH'
    df_dic[row_name] = [f"{num:.2f}" for num in err_all[5, :]]
    row_name = 'GLU_exact'
    df_dic[row_name] = [f"{num:.2f}" for num in err_all[6, :]]
    row_name = 'GRSU_OH'
    df_dic[row_name] = [f"{num:.2f}" for num in err_all[7, :]]
    row_name = 'GRSU_exact'
    df_dic[row_name] = [f"{num:.2f}" for num in err_all[8, :]]

    df = pd.DataFrame.from_dict(df_dic, orient='index',
                                columns=column_names)
    times_dic = {
        'GLNMF': t_nmf,
        'QMV': t_qmv,
        'GTVMBO': t_MBO,
        'MSC': t_MSC,
        'EGU': t_EGU,
        'GLU_OH': t_GL[0],
        'GLU_exact': t_GL[1],
        'GRSU_OH': t_GRSU[0],
        'GRSU_exact': t_GRSU[1]
    }
    times_series = pd.Series(times_dic, name='Time')

    df = df.join(times_series)

    if save_data:
        csv_name = os.path.join(data_folder, dataset, dataset + '_results_all.csv')
        df.to_csv(csv_name)

    A_label = np.zeros_like(A_ref)
    for i in range(len(label_inds)):
        A_label[ref_labels[i], label_inds[i]] = 1

    if dataset == "samson":
        labelsize = 15
        titlesize = 14
    else:
        labelsize = 14
        titlesize = 11
    n_row, n_col = num_class, 11
    figsize = plt.figaspect(n_row / n_col)

    fig, ax = plt.subplots(n_row, n_col, figsize=figsize)
    for i in range(n_row):
        ax[i, 0].imshow(A_nmf[i].reshape(n, m).T)
        ax[i, 0].set_ylabel(label_names[i], fontsize=labelsize)
        ax[i, 0].xaxis.set_major_locator(ticker.NullLocator())
        ax[i, 0].yaxis.set_major_locator(ticker.NullLocator())
        ax[i, 1].imshow(A_qmv[i].reshape(n, m).T)
        ax[i, 1].axis('off')
        ax[i, 2].imshow(A_MBO[i].reshape(n, m).T)
        ax[i, 2].axis('off')
        ax[i, 3].imshow(A_MSC[i].reshape(n, m).T)
        ax[i, 3].axis('off')
        ax[i, 4].imshow(A_EGU[i].reshape(n, m).T)
        ax[i, 4].axis('off')
        ax[i, 5].imshow(A_GL[0][i].reshape(n, m).T)
        ax[i, 5].axis('off')
        ax[i, 6].imshow(A_GL[1][i].reshape(n, m).T)
        ax[i, 6].axis('off')
        ax[i, 7].imshow(A_GRSU[0][i].reshape(n, m).T)
        ax[i, 7].axis('off')
        ax[i, 8].imshow(A_GRSU[1][i].reshape(n, m).T)
        ax[i, 8].axis('off')
        ax[i, 9].imshow(A_ref[i].reshape(n, m).T)
        ax[i, 9].axis('off')
        ## last plot with red dots
        label_map = A_label[i].reshape(n, m).T
        xy_coords = np.column_stack(np.where(label_map))
        im = ax[i, 10].imshow(A_ref[i].reshape(n, m).T)
        for label_ind in range(len(xy_coords)):
            circ = Circle((xy_coords[label_ind][1], xy_coords[label_ind][0]), radius=2, color='red')
            ax[i, 10].add_patch(circ)
        ax[i, 10].axis('off')
        im.set_clim(0, 1)
        if i == 0:
            # ax[i,0].title.set_text('Labels')
            # ax[i,1].title.set_text('GT')
            ax[i, 0].title.set_text('GLNMF')
            ax[i, 1].title.set_text('QMV')
            ax[i, 2].title.set_text('GTVMBO')
            ax[i, 3].title.set_text('MSC')
            ax[i, 4].title.set_text('EGU')
            ax[i, 5].title.set_text('GLU-OH')
            ax[i, 6].title.set_text('GLU-EXT')
            ax[i, 7].title.set_text('GRSU-OH')
            ax[i, 8].title.set_text('GRSU-EXT')
            ax[i, 9].title.set_text('GT')
            ax[i, 10].title.set_text('Labels')
    for j in range(n_col):
        ax[0, j].title.set_size(titlesize)
    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.96, 0.3, 0.02, 0.4])
    fig.colorbar(im, cax=cbar_ax)
    if save_data:
        plt.savefig(os.path.join(data_folder, dataset, dataset + '_A.eps'), format='eps', bbox_inches='tight', dpi=dpi_vals[0])
    plt.show()

    labelsize = 15
    titlesize = 15

    dx, dy = 1.5, 1
    n_row, n_col = num_class, 9
    figsize = plt.figaspect(float(dy * n_row) / float(dx * n_col))

    fig, ax = plt.subplots(n_row, n_col, sharex='col', sharey='row', figsize=figsize)
    for i in range(n_row):
        ax[i, 0].plot(normalize(S_ref[:, i]))
        ax[i, 0].plot(normalize(S_nmf[:, i]))
        ax[i, 0].set_ylabel(label_names[i], fontsize=labelsize)
        ax[i, 1].plot(normalize(S_ref[:, i]))
        ax[i, 1].plot(normalize(S_qmv[:, i]))
        ax[i, 2].plot(normalize(S_ref[:, i]))
        ax[i, 2].plot(normalize(S_MBO[:, i]))
        ax[i, 3].plot(normalize(S_ref[:, i]))
        ax[i, 3].plot(normalize(S_MSC[:, i]))
        ax[i, 4].plot(normalize(S_ref[:, i]))
        ax[i, 4].plot(normalize(S_EGU[:, i]))
        ax[i, 5].plot(normalize(S_ref[:, i]))
        ax[i, 5].plot(normalize(S_GL[0][:, i]))
        ax[i, 6].plot(normalize(S_ref[:, i]))
        ax[i, 6].plot(normalize(S_GL[1][:, i]))
        ax[i, 7].plot(normalize(S_ref[:, i]))
        ax[i, 7].plot(normalize(S_GRSU[0][:, i]))
        ax[i, 8].plot(normalize(S_ref[:, i]))
        ax[i, 8].plot(normalize(S_GRSU[1][:, i]))
        if i == 0:
            ax[i, 0].title.set_text('GLNMF')
            ax[i, 1].title.set_text('QMV')
            ax[i, 2].title.set_text('GTVMBO')
            ax[i, 3].title.set_text('MSC')
            ax[i, 4].title.set_text('EGU')
            ax[i, 5].title.set_text('GLU-OH')
            ax[i, 6].title.set_text('GLU-EXT')
            ax[i, 7].title.set_text('GRSU-OH')
            ax[i, 8].title.set_text('GRSU-EXT')
    for j in range(n_col):
        ax[0, j].title.set_size(titlesize)
    if save_data:
        plt.savefig(os.path.join(data_folder, dataset, dataset + '_S.eps'), format='eps', bbox_inches='tight', dpi=dpi_vals[0])
    plt.show()

    return df