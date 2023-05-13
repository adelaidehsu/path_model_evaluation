from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, paired_distances
from scipy.stats import spearmanr
import collections
import random
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, confusion_matrix

def center(W):
    # W = N x d
    # centering features
    mean = np.mean(W, axis=0, keepdims=True)
    norm_W = W - mean
    return norm_W

# Calculates the representational geometry of a set of embeddings
def calculate_geometry(X, Y, dis_metric = 'euc', flag='tri'):
    tri_idxes = None
    if dis_metric == 'euc':
        d_mat = euclidean_distances(X, Y)
    else:
        d_mat = cosine_similarity(X, Y)
    
    if flag == 'tri': # overlapped samples
        tri_idxes = np.triu_indices(X.shape[0], 1)
        geometry = d_mat[tri_idxes].reshape(-1)
    else: # nonoverlapped samples
        geometry = paired_distances(X,Y)

    return geometry, tri_idxes, d_mat

def compute_RSA(num_layers, sample_flag, geo_flag, dis_flag,
                ft_layer_cls_logits1, un_layer_cls_logits1, ft_layer_cls_logits2=None, un_layer_cls_logits2=None):
    # compute RSA
    layer_sims = collections.defaultdict(list)
    layer_affns = collections.defaultdict(list)
    for i in range(num_layers):
        ft_r1 = ft_layer_cls_logits1[i] #[N, d]
        un_r1 = un_layer_cls_logits1[i] #[N, d]
        if sample_flag != 'nonoverlap':
            ft_r2 = ft_r1
            un_r2 = un_r1
        else:
            ft_r2 = ft_layer_cls_logits2[i] #[N, d]
            un_r2 = un_layer_cls_logits2[i] #[N, d]
            
        ft_g_v, label_indices, ft_affn = calculate_geometry(ft_r1, ft_r2, dis_metric=dis_flag, flag=geo_flag)
        un_g_v, _, un_affn = calculate_geometry(un_r1, un_r2, flag=geo_flag)
            
        sim = spearmanr(ft_g_v, un_g_v)
        layer_sims[i].append(sim[0])
        layer_affns[i].append((ft_affn, un_affn))
        
    return layer_sims, layer_affns, label_indices

def get_projection(nW):
    u, s, vh = np.linalg.svd(nW) # u: [N, N]; s: [768]; vh: [768, 768]
    proj = (nW @ vh.T) [:, :2]
    return proj #[N, 2]

def low_rank_approx(k, A):
    # A: [N, d]
    # build from bottom k principle components of matrix A
    u, s, vh = np.linalg.svd(A, full_matrices=False) #[N, 768], [768], [768, 768]
    num_c = vh.shape[0]
    
    v = vh.T #[768, 768]
    v_k = v[:, k:]
    s_k = np.diag(s[k:])
    u_k = u[:, k:]
    A_k = np.matmul(np.matmul(u_k, s_k), v_k.T)

    return A_k

def rank_1_approx(k, A):
    # A: [N, d]
    # build kth principle components of matrix A
    u, s, vh = np.linalg.svd(A, full_matrices=False) #[N, 768], [768], [768, 768]
    num_c = vh.shape[0]
    
    v = vh.T #[768, 768]
    v_k = np.expand_dims(v[:, k], axis=1)
    s_k = np.expand_dims(np.expand_dims(s[k], axis=0), axis=1)
    u_k = np.expand_dims(u[:, k], axis=1)
    A_k = np.matmul(np.matmul(u_k, s_k), v_k.T)

    return A_k
    
def get_acc(logits, true_labels_str):
    logits = logits.detach().cpu().numpy()
    y_preds = np.argmax(logits, axis=1)
    dist = collections.Counter(y_preds)
    
    true_label_ids = np.array([le_dict[l] for l in true_labels_str])
    
    f1_macro = np.round(f1_score(true_label_ids, y_preds, average='macro'), 3)
    
    # per class acc    
    cm = confusion_matrix(true_label_ids, y_preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    per_class_acc = cm.diagonal()
    

    return f1_macro, per_class_acc, dist

def get_stimuli(N, corpus, labels, flag='fixed'):
    if flag == 'random':
        ###### N random samples
        idx = random.sample(range(0, len(corpus)), N)
        samples = [[corpus[i] for i in idx]]
        labels = [[labels[i] for i in idx]]
    elif flag == 'nonoverlap':
        ###### N random nonoverlapped samples
        idx = random.sample(range(0, len(corpus)), N)
        remain = [j for j in range(0, len(corpus)) if j not in idx]
        idx2 = random.sample(remain, N)
        
        sample1 = [corpus[i] for i in idx]
        label1 = [labels[i] for i in idx]
        sample2 = [corpus[i] for i in idx2]
        label2 = [labels[i] for i in idx2]
        
        samples = [sample1, sample2]
        labels = [label1, label2]
    else:
        ###### N fixed samples
        #samples = [corpus[N:2*N]]
        #labels = [labels[N:2*N]]
        samples = [corpus[:N]]
        labels = [labels[:N]]
        
    return samples, labels