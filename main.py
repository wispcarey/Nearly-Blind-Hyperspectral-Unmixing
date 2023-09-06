import argparse
from data_processing import *
from SSU import *
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set parameters for the program.')

    # dataset argument
    parser.add_argument('--dataset', type=str, default='jasper',
                        help='If one of "jasper", "urban", "samson", "apex", precomputed results will be used. Otherwise, it is considered as the path to the saved npy file.')

    # train_percentage argument
    parser.add_argument('--train_percentage', type=float, default=0.4,
                        help='The percentage of labels to be used for training. Should be a float value between 0 and 1.')

    # AL_method argument
    parser.add_argument('--AL_method', type=str, default='mcvopt', choices=['vopt', 'uc', 'mc', 'mcvopt'],
                        help='The type of acquisition function to be used in active learning. Choices are "vopt", "uc", "mc", "mcvopt".')

    args = parser.parse_args()

    pd.set_option('display.max_columns', None)

    precomputed_result = args.dataset in ["jasper", "urban", "samson", "apex"]

    if precomputed_result:
        df = output_results(args.dataset, save_data=False, dpi_vals=(200,200))
        print(df)
    else:
        data_dict = np.load(args.dataset, allow_pickle=True).item()
        try:
            X = data_dict["X"]
            A_gt = data_dict["A_gt"]
            S_gt = data_dict["S_gt"]
        except KeyError as e:
            print(f"The key {e} is missing from data_dict. data_dict must contain the keys 'X', 'A_gt', 'S_gt'.")
        m,n = X.shape[0], X.shape[1]
        X = X.transpose(1,0,2).reshape(-1, X.shape[2]).T
        pseudo_labels = np.argmax(A_gt, axis=0)
        class_names = np.unique(pseudo_labels)
        num_class = len(class_names)

        num_sample = int(args.train_percentage * X.shape[1] / 100)
        print("Number of training samples:", num_sample)

        feature_vectors = X.T
        knn_data = gl.weightmatrix.knnsearch(feature_vectors, 50, method='annoy', similarity='angular')
        W = gl.weightmatrix.knn(feature_vectors, 50, kernel='gaussian', knn_data=knn_data)
        G = gl.graph(W)

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

        label_inds, _, _ = bal.coreset_run_experiment(X.T, pseudo_labels, W, initial, num_iter=num_iter, method='Laplace',
                                                  display=False, use_prior=False, al_mtd=al_mtd, debug=False,
                                                  acq_fun=args.AL_method, knn_data=knn_data, mtd_para=(np.inf, 0, 'new'),
                                                  savefig=False, savefig_folder='../BAL_figures', batchsize=batch_size,
                                                  dist_metric='angular', knn_size=50, q=1, thresholding=0, randseed=0,
                                                  dropout=0)
        print("AL process finished.")

        img_shape = (n, m, X.shape[0])
        A_GL = np.zeros((2, num_class, n * m))
        S_GL = np.zeros((2, X.shape[0], num_class))

        ref_labels = pseudo_labels[label_inds]

        X_labeled = X[:, label_inds]
        A_labeled_onehot = np.zeros((num_class, len(label_inds)))
        for i in range(num_class):
            A_labeled_onehot[i, ref_labels == i] = 1
        A_labeled_exact = A_gt[:, label_inds]

        t = time.time()
        A_GL[0] = GL_unmixing(X, label_inds, ref_labels, img_shape, d=0)
        S_GL[0] = gl_fitS(X_labeled, A_labeled_onehot)
        GL_t1 = time.time() - t

        t = time.time()
        exact_amap = A_gt[:, label_inds].T
        A_GL[1] = GL_unmixing(X, label_inds, exact_amap, img_shape, d=0, use_exact_amap=True)
        S_GL[1] = gl_fitS(X_labeled, A_labeled_exact)
        GL_t2 = time.time() - t

        para = (200, 5e1, 1e-1, 1e-1, 'GL_laplace_ref', 1e-3)
        alpha = (2e1) ** 2

        A_GRSU = np.zeros((2, num_class, n * m))
        S_GRSU = np.zeros((2, X.shape[0], num_class))

        S_init = gl_fitS(X_labeled, A_labeled_onehot)
        A_init = init_fitA(X, S_init)

        t = time.time()
        s1, a1, b1, it = GRSU_unmixing(X, S_init, A_init, label_inds, ref_labels, para, r=1e-1, alpha=alpha,
                                       A_ref=A_gt)
        S_GRSU[0] = s1
        A_GRSU[0] = a1
        GRSU_t1 = time.time() - t

        t = time.time()
        exact_amap = A_gt[:, label_inds].T
        s2, a2, b2, it1 = GRSU_unmixing(X, S_init, A_init, label_inds, exact_amap, para, r=1e-1, alpha=alpha,
                                        A_ref=A_gt, use_exact_amap=True)
        S_GRSU[1] = s2
        A_GRSU[1] = a2
        GRSU_t2 = time.time() - t

        print("GLU and GRSU finished (with both exact abundance map and pseudo labels).")

        label_names = [f'Class{i}' for i in range(num_class)]

        err_all = np.zeros((4, 2 * num_class + 2))

        err_all[0] = class_metrics(A_GL[0], S_GL[0], A_gt, S_gt)
        err_all[1] = class_metrics(A_GL[1], S_GL[1], A_gt, S_gt)
        err_all[2] = class_metrics(A_GRSU[0], S_GRSU[0], A_gt, S_gt)
        err_all[3] = class_metrics(A_GRSU[1], S_GRSU[1], A_gt, S_gt)

        column_names = ['RMSE'] + label_names + ['SAD'] + label_names

        df_dic = {}
        row_name = 'GLU_OH'
        df_dic[row_name] = [f"{num:.2f}" for num in err_all[0, :]]
        row_name = 'GLU_exact'
        df_dic[row_name] = [f"{num:.2f}" for num in err_all[1, :]]
        row_name = 'GRSU_OH'
        df_dic[row_name] = [f"{num:.2f}" for num in err_all[2, :]]
        row_name = 'GRSU_exact'
        df_dic[row_name] = [f"{num:.2f}" for num in err_all[3, :]]

        df = pd.DataFrame.from_dict(df_dic, orient='index',
                                    columns=column_names)
        times_dic = {
            'GLU_OH': GL_t1,
            'GLU_exact': GL_t2,
            'GRSU_OH': GRSU_t1,
            'GRSU_exact': GRSU_t2
        }
        times_series = pd.Series(times_dic, name='Time')

        df = df.join(times_series)

        print("Result table finished. Making figures...")

        A_label = np.zeros_like(A_gt)
        for i in range(len(label_inds)):
            A_label[ref_labels[i], label_inds[i]] = 1
        labelsize = 14
        titlesize = 11

        n_row, n_col = num_class, 6
        figsize = plt.figaspect(n_row / n_col)

        fig, ax = plt.subplots(n_row, n_col, figsize=figsize)
        for i in range(n_row):
            ax[i, 0].imshow(A_GL[0][i].reshape(n, m).T)
            ax[i, 0].set_ylabel(label_names[i], fontsize=labelsize)
            ax[i, 0].xaxis.set_major_locator(ticker.NullLocator())
            ax[i, 0].yaxis.set_major_locator(ticker.NullLocator())
            ax[i, 1].imshow(A_GL[1][i].reshape(n, m).T)
            ax[i, 1].axis('off')
            ax[i, 2].imshow(A_GRSU[0][i].reshape(n, m).T)
            ax[i, 2].axis('off')
            ax[i, 3].imshow(A_GRSU[1][i].reshape(n, m).T)
            ax[i, 3].axis('off')
            ax[i, 4].imshow(A_gt[i].reshape(n, m).T)
            ax[i, 4].axis('off')
            ## last plot with red dots
            label_map = A_label[i].reshape(n, m).T
            xy_coords = np.column_stack(np.where(label_map))
            im = ax[i, 5].imshow(A_gt[i].reshape(n, m).T)
            for label_ind in range(len(xy_coords)):
                circ = Circle((xy_coords[label_ind][1], xy_coords[label_ind][0]), radius=2, color='red')
                ax[i, 5].add_patch(circ)
            ax[i, 5].axis('off')
            im.set_clim(0, 1)
            if i == 0:
                ax[i, 0].title.set_text('GLU-OH')
                ax[i, 1].title.set_text('GLU-EXT')
                ax[i, 2].title.set_text('GRSU-OH')
                ax[i, 3].title.set_text('GRSU-EXT')
                ax[i, 4].title.set_text('GT')
                ax[i, 5].title.set_text('Labels')
        for j in range(n_col):
            ax[0, j].title.set_size(titlesize)
        fig.subplots_adjust(right=0.95)
        cbar_ax = fig.add_axes([0.96, 0.3, 0.02, 0.4])
        fig.colorbar(im, cax=cbar_ax)
        plt.show()

        labelsize = 15
        titlesize = 15

        dx, dy = 1.5, 1
        n_row, n_col = num_class, 4
        figsize = plt.figaspect(float(dy * n_row) / float(dx * n_col))

        fig, ax = plt.subplots(n_row, n_col, sharex='col', sharey='row', figsize=figsize)
        for i in range(n_row):
            ax[i, 0].plot(normalize(S_gt[:, i]))
            ax[i, 0].plot(normalize(S_GL[0][:, i]))
            ax[i, 0].set_ylabel(label_names[i], fontsize=labelsize)
            ax[i, 1].plot(normalize(S_gt[:, i]))
            ax[i, 1].plot(normalize(S_GL[1][:, i]))
            ax[i, 2].plot(normalize(S_gt[:, i]))
            ax[i, 2].plot(normalize(S_GRSU[0][:, i]))
            ax[i, 3].plot(normalize(S_gt[:, i]))
            ax[i, 3].plot(normalize(S_GRSU[1][:, i]))
            if i == 0:
                ax[i, 0].title.set_text('GLU-OH')
                ax[i, 1].title.set_text('GLU-EXT')
                ax[i, 2].title.set_text('GRSU-OH')
                ax[i, 3].title.set_text('GRSU-EXT')
        for j in range(n_col):
            ax[0, j].title.set_size(titlesize)
        plt.show()
        print(df)

