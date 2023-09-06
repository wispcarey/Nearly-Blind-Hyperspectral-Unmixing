import graphlearning as gl
from graphlearning import utils, graph

import numpy as np
import scipy.sparse as sparse
from scipy.special import softmax

import utils as utils_gl

class GL_GRSU(gl.ssl.ssl):
    def __init__(self, W=None, class_priors=None, X=None, reweighting='none', normalization='combinatorial',
                 tau=0, mean_shift=False, tol=1e-5, alpha=2, zeta=1e7, r=0.1, rho=1e-5):
        super().__init__(W, class_priors)

        ## r is the parameter for the GR graph learning

        self.reweighting = reweighting
        self.normalization = normalization
        self.mean_shift = mean_shift
        self.tol = tol
        self.X = X
        self.r = r
        self.rho = rho

        #Set up tau
        if type(tau) in [float,int]:
            self.tau = np.ones(self.graph.num_nodes)*tau
        elif type(tau) is np.ndarray:
            self.tau = tau

        #Setup accuracy filename
        fname = '_laplace'
        self.name = 'Laplace Learning'
        if self.reweighting != 'none':
            fname += '_' + self.reweighting
            self.name += ': ' + self.reweighting + ' reweighted'
        if self.normalization != 'combinatorial':
            fname += '_' + self.normalization
            self.name += ' ' + self.normalization
        if self.mean_shift:
            fname += '_meanshift'
            self.name += ' with meanshift'
        if np.max(self.tau) > 0:
            fname += '_tau_%.3f'%np.max(self.tau)
            self.name += ' tau=%.3f'%np.max(self.tau)

        self.accuracy_filename = fname

    def _fitGRSU_GR(self, train_ind, train_labels, unlabeled_ref, all_labels=None, use_exact_amap=False):

        #Reweighting
        if self.reweighting == 'none':
            G = self.graph
        else:
            W = self.graph.reweight(train_ind, method=self.reweighting, normalization=self.normalization, X=self.X)
            G = graph.graph(W)

        #Get some attributes
        n = G.num_nodes
        if not use_exact_amap:
            unique_labels = np.unique(train_labels)
            k = len(unique_labels)
        else:
            k = train_labels.shape[1]

        #tau + Graph Laplacian and one-hot labels
        L = sparse.spdiags(self.tau, 0, G.num_nodes, G.num_nodes) + G.laplacian(normalization=self.normalization)
        if not use_exact_amap:
            F = utils.labels_to_onehot(train_labels)
        else:
            F = train_labels

        #Locations of unlabeled points
        idx = np.full((n,), True, dtype=bool)
        idx[train_ind] = False

        #Right hand side
        b = np.zeros((n, k))
        b[idx, :] = self.rho * unlabeled_ref
        b[train_ind, :] = 1/(self.r ** 2) * F

        #Left hand side matrix
        A1 = sparse.spdiags(self.rho*idx.astype(float),0,n,n).tocsr()
        A2 = sparse.spdiags(1/(self.r**2)*(1-idx.astype(float)),0,n,n).tocsr()
        A = L + A1 + A2

        #Preconditioner
        M = A.diagonal()
        M = sparse.spdiags(1/np.sqrt(M+1e-10),0,n,n).tocsr()

        #Conjugate gradient solver
        v = utils.conjgrad(M*A*M, M*b, tol=self.tol)
        u = M*v

        # #Add labels back into array
        # u = np.zeros((n,k))
        # u[idx,:] = v
        # u[train_ind,:] = F

        #Mean shift
        if self.mean_shift:
            u -= np.mean(u,axis=0)

        return u

    def _fit(self, train_ind, train_labels, all_labels=None, use_exact_amap=False):

        #Reweighting
        if self.reweighting == 'none':
            G = self.graph
        else:
            W = self.graph.reweight(train_ind, method=self.reweighting, normalization=self.normalization, X=self.X)
            #W = self.graph.reweight(train_ind, method=self.reweighting, X=self.X)
            G = graph.graph(W)

        #Get some attributes
        n = G.num_nodes
        if not use_exact_amap:
            unique_labels = np.unique(train_labels)
            k = len(unique_labels)
        else:
            k = train_labels.shape[1]

        #tau + Graph Laplacian and one-hot labels
        L = sparse.spdiags(self.tau, 0, G.num_nodes, G.num_nodes) + G.laplacian(normalization=self.normalization)
        if not use_exact_amap:
            F = utils.labels_to_onehot(train_labels)
        else:
            F = train_labels

        #Locations of unlabeled points
        idx = np.full((n,), True, dtype=bool)
        idx[train_ind] = False

        #Right hand side
        b = -L[:,train_ind]*F
        b = b[idx,:]

        #Left hand side matrix
        A = L[idx,:]
        A = A[:,idx]

        #Preconditioner
        m = A.shape[0]
        M = A.diagonal()
        M = sparse.spdiags(1/np.sqrt(M+1e-10),0,m,m).tocsr()

        #Conjugate gradient solver
        v = utils.conjgrad(M*A*M, M*b, tol=self.tol)
        v = M*v

        #Add labels back into array
        u = np.zeros((n,k))
        u[idx,:] = v
        u[train_ind,:] = F

        #Mean shift
        if self.mean_shift:
            u -= np.mean(u,axis=0)

        return u

    def _fitGRSU_laplace(self, train_ind, train_labels, unlabeled_ref, all_labels=None, use_exact_amap=False):

        #Reweighting
        if self.reweighting == 'none':
            G = self.graph
        else:
            W = self.graph.reweight(train_ind, method=self.reweighting, normalization=self.normalization, X=self.X)
            #W = self.graph.reweight(train_ind, method=self.reweighting, X=self.X)
            G = graph.graph(W)

        #Get some attributes
        n = G.num_nodes
        if not use_exact_amap:
            unique_labels = np.unique(train_labels)
            k = len(unique_labels)
        else:
            k = train_labels.shape[1]

        #tau + Graph Laplacian and one-hot labels
        L = sparse.spdiags(self.tau, 0, G.num_nodes, G.num_nodes) + G.laplacian(normalization=self.normalization)
        if not use_exact_amap:
            F = utils.labels_to_onehot(train_labels)
        else:
            F = train_labels

        #Locations of unlabeled points
        idx = np.full((n,), True, dtype=bool)
        idx[train_ind] = False

        #Right hand side
        b = -L[:,train_ind]*F
        b = b[idx,:]
        b = b + self.rho * unlabeled_ref

        #Left hand side matrix
        A = L[idx,:]
        A = A[:,idx]
        m = A.shape[0]
        A = A + self.rho * sparse.spdiags(np.ones(m),0,m,m).tocsr()

        #Preconditioner
        M = A.diagonal()
        M = sparse.spdiags(1/np.sqrt(M+1e-10),0,m,m).tocsr()

        #Conjugate gradient solver
        v = utils.conjgrad(M*A*M, M*b, tol=self.tol)
        v = M*v

        #Add labels back into array
        u = np.zeros((n,k))
        u[idx,:] = v
        u[train_ind,:] = F

        #Mean shift
        if self.mean_shift:
            u -= np.mean(u,axis=0)

        return u

## simplx projection
def SimplexProj(X):
    K = X.shape[0]
    return np.maximum(X - np.max((np.cumsum(np.sort(X, axis=0)[::-1,:], axis=0)-1) / np.arange(1, K+1).reshape(K,1), axis=0), 0)

## GRSU function
def GRSU_unmixing(X, S0, A0, train_inds, train_labels, para, r=0.1, alpha=1, A_ref=None, verbose=False, use_exact_amap=False):

    maxiter, lam, rho, gamma, method, tol = para

    K = S0.shape[1]
    N = X.shape[1]

    num_class = len(np.unique(train_labels))

    S = S0
    A = A0
    B = A
    Btilde = np.zeros_like(A)
    Ctilde = np.zeros_like(S)

    idx = np.full((N,), True, dtype=bool)
    idx[train_inds] = False

    if verbose:
        print(method)

    feature_vectors = X.T
    knn_data = gl.weightmatrix.knnsearch(feature_vectors, 50, method='annoy', similarity='angular')
    W = gl.weightmatrix.knn(feature_vectors, 50, kernel = 'gaussian', knn_data=knn_data)
    GR_GL = GL_GRSU(W, class_priors=None, X=feature_vectors, r=r, rho=rho/lam)

    X_ref_labeled = X[:,train_inds]
    if use_exact_amap:
        if A_ref is not None:
            A_ref_labeled = A_ref[:,train_inds]
        else:
            A_ref_labeled = A0[:,train_inds]
    else:
        A_ref_labeled = np.zeros((num_class, len(train_inds)))
        for i in range(num_class):
            A_ref_labeled[i, train_labels==i] = 1

    C1 = X_ref_labeled @ A_ref_labeled.T
    C2 = A_ref_labeled @ A_ref_labeled.T


    for i in range(maxiter):
        if np.mod(i-1,50) == 0 and verbose:
            print('iter: %d'%i)
            print(np.linalg.cond(A @ A.T))
            print(np.linalg.cond(S.T @ S))

        Apre = A
        Spre = S

        # C-sub problem
        C = (X @ A.T + alpha * C1 + gamma * (S + Ctilde)) @ np.linalg.inv(A@A.T + alpha * C2 + gamma * np.eye(K))

        # S-subproblem
        S = np.maximum(0,C-Ctilde)

        # A-subproblem
        LHS = S.T @ S + rho * np.eye(K)
        RHS = S.T @ X + rho * (B - Btilde)
        A = np.linalg.inv(LHS) @ RHS

        if np.any(np.isnan(A)) or np.any(np.isnan(S)):
            it = i
            break

        if np.max(A) > 1e3 or np.max(S) > 1e3:
            it = i
            break

        # projection onto the probability simplex
        A = SimplexProj(A)

        # B-subproblem
        if method == 'GL_GR':
            unlabeled_ref = A + Btilde
            unlabeled_ref = unlabeled_ref[:, idx].T
            B = GR_GL._fitGRSU_GR(train_inds, train_labels, unlabeled_ref, all_labels=None, use_exact_amap=use_exact_amap)
            B = B.T
        elif method == 'GL_laplace_ref':
            unlabeled_ref = A + Btilde
            unlabeled_ref = unlabeled_ref[:, idx].T
            B = GR_GL._fitGRSU_laplace(train_inds, train_labels, unlabeled_ref, all_labels=None, use_exact_amap=use_exact_amap)
            B = B.T
        elif method == 'GL_laplace':
            if i == 0:
                B = GR_GL._fit(train_inds, train_labels, all_labels=None, use_exact_amap=use_exact_amap)
                B = B.T
        else:
            raise ValueError('method should be GL')

        # update of auxiliary variables
        Btilde = Btilde + A - B
        Ctilde = Ctilde + S - C

        # stopping criteria
        if np.linalg.norm(Spre-S,'fro')<tol*np.linalg.norm(Spre) and np.linalg.norm(Apre-A,'fro')<tol*np.linalg.norm(Apre):
           break

    it = i
    return S, A, B, it

def GL_unmixing(X, train_inds, train_labels, img_shape, d=0, knn_num=50, use_exact_amap=False):

    if d > 0:
        X_img = X.T.reshape(img_shape)
        feature_vectors = utils_gl.NonLocalMeans(X_img, d)
    else:
        feature_vectors = X.T
    knn_data = gl.weightmatrix.knnsearch(feature_vectors, knn_num, method='annoy', similarity='angular')
    W = gl.weightmatrix.knn(feature_vectors, knn_num, kernel = 'gaussian', knn_data=knn_data)
    GR_GL = GL_GRSU(W, class_priors=None, X=feature_vectors)
    A = GR_GL._fit(train_inds, train_labels, use_exact_amap=use_exact_amap)

    return A.T

def gl_fitS(X,A):

    S = (X @ A.T) @ np.linalg.inv(A@A.T)
    S = np.maximum(S,0)

    return S

def init_fitA(X,S):

    A = np.linalg.inv(S.T@S) @ (S.T @ X)
    A = SimplexProj(A)

    return A

def nMSE(x, xhat):
    return np.linalg.norm(x-xhat, 'fro')/np.linalg.norm(x, 'fro') * 100

def RMSE(x, xhat):
    m = x.shape[0]
    return np.mean(np.sqrt(1/m*np.sum((x-xhat)**2,axis=0))) * 100

def RMSE_new(x, xhat):
    m,n = x.shape
    return 100*np.sqrt(np.sum((x-xhat)**2)/m/n)

def SAD(x, xhat):
    if x.ndim == 2:
        m,n = x.shape
        sum = 0
        for i in range(n):
            norm1 = np.linalg.norm(x[:,i]) + 1e-8
            norm2 = np.linalg.norm(xhat[:,i]) + 1e-8
            sum += np.arccos( np.inner(x[:,i],xhat[:,i])/norm1/norm2 )
        return 180/np.pi * sum/n
    else:
        norm1 = np.linalg.norm(x) + 1e-8
        norm2 = np.linalg.norm(xhat) + 1e-8
        sum = np.arccos( np.inner(x,xhat)/norm1/norm2 )
        return 180/np.pi * sum

def uc_unmixing(X, A, S, normalize=True):

    diff = X - S@A
    uc_val = -np.linalg.norm(diff, axis=0)

    if normalize:
        norm_X = np.linalg.norm(X, axis=0)
        uc_val = uc_val / norm_X

    return uc_val

def uc_smallest_margin(A, num_class):

    A_sort = np.sort(A,axis=0)
    A_softmax = softmax(A_sort, axis=0)
    uc_val = 1 - (A_softmax[num_class-1,:] - A_softmax[num_class-2,:])

    return uc_val

def confidence_selection(uc_val, pred_labels, sample_rate, num_class):
    sort_cfd_indx = np.argsort(uc_val)
    sample_num_list = np.zeros((num_class,))
    train_cfd_indx = np.array([], dtype=int)

    curr_ind = 0
    while np.min(sample_num_list) < sample_rate:
        cfd_ind = sort_cfd_indx[curr_ind]
        if sample_num_list[pred_labels[cfd_ind]] < sample_rate:
            sample_num_list[pred_labels[cfd_ind]] += 1
            train_cfd_indx = np.append(train_cfd_indx, cfd_ind)
        curr_ind += 1

    return train_cfd_indx

def class_metrics(A, S, A_ref, S_ref, gt_labels=None):

    if gt_labels is None:
        gt_labels = np.argmax(A_ref, axis=0)

    num_class = A.shape[0]
    err_val = np.zeros((2*num_class+2,))

    err_val[0] = RMSE_new(A, A_ref)
    err_val[num_class+1] = SAD(S, S_ref)
    for i in range(num_class):
        indx = gt_labels == i
        err_val[i+1] = RMSE_new(A[:,indx], A_ref[:,indx])
        err_val[i+num_class+2] = SAD(S[:,i], S_ref[:,i])

    return err_val

def normalize(x):
    return x / np.linalg.norm(x)