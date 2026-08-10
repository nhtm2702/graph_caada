import numpy as np
import torch
import time
from sklearn.metrics import f1_score, log_loss
from model import LogisticRegression
import logging
import random
import torch.backends.cudnn as cudnn
import datetime
import torch.nn.functional as F
from train_gcn import *

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tsk = 'VisDa2017' # 'OfficeHome/Ar2Cl'  'DomainNet/c2p' 
    
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler("./log/" + tsk + "/gcn_" + timestamp + ".log",  mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(message)s'))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("Session started")
    

def train_and_eval(train_x, train_y, val_x, val_y, one_idx, C):
    model = LogisticRegression(C)
    model.fit(train_x, train_y)
    
    return model.model.score(val_x, val_y), model.model.score(val_x[one_idx], val_y[one_idx]), f1_score(model.model.predict(val_x), val_y, average='binary'), model

def update_datasets(train_x, val_x, train_y, val_y, q_idxs):
    # query new dataset and retrain
    train_x_new = np.concatenate((train_x, val_x[q_idxs]))
    train_y_new = np.concatenate((train_y, val_y[q_idxs]))

    valid_x = val_x[q_idxs]
    valid_y = val_y[q_idxs]

    val_x_new = np.delete(val_x, q_idxs, axis=0)
    val_y_new = np.delete(val_y, q_idxs, axis=0)
    
    return  train_x_new, val_x_new, train_y_new, val_y_new, valid_x, valid_y

def retrain_model(train_x_new, train_y_new, val_x, val_y, one_idx, C, weights=None):
    model = LogisticRegression(C)
    model.fit(train_x_new, train_y_new, sample_weight=weights)
    
    return model.model.score(val_x, val_y), model.model.score(val_x[one_idx], val_y[one_idx]), f1_score(model.model.predict(val_x), val_y, average='binary'), model

def balanced_source_subset(source_labels,max_per_class):
    
    generator = torch.Generator()
    source_labels = torch.tensor(source_labels)

    positive = torch.where(source_labels == 1)[0]
    negative = torch.where(source_labels == 0)[0]

    perm = torch.randperm(len(positive),generator=generator)[:]
    positive = positive[perm]

    perm = torch.randperm(len(negative),generator=generator)[:max_per_class]
    negative = negative[perm]

    return torch.cat([positive, negative])

def build_source_target_adjacency(source_features,target_features,k_source=5,k_target=15,temperature=0.1,chunk_size=2048):
    """
    Construct a sparse source-target adjacency matrix for a GCN.

    Each node connects to:
        - k_source nearest source nodes
        - k_target nearest target nodes

    Parameters
    ----------
    
    temperature:
        Temperature for converting cosine similarities to edge weights:
            exp((cosine_similarity - 1) / temperature)

    chunk_size:
        Number of query nodes processed simultaneously. Reduce this if
        adjacency construction causes GPU-memory problems.

    Returns
    -------
    torch.Tensor:
        Sparse COO adjacency matrix of shape:
            [N_source + N_target, N_source + N_target]

        This output can be passed directly to torch.sparse.mm or torch.spmm.
    """

    source_features = torch.as_tensor(source_features,dtype=torch.float32,device=device)
    target_features = torch.as_tensor(target_features,dtype=torch.float32,device=device)

    num_source = source_features.shape[0]
    num_target = target_features.shape[0]
    num_nodes = num_source + num_target

    all_features = torch.cat([source_features, target_features],dim=0)

    graph_features = F.normalize(all_features,p=2,dim=1,eps=1e-12)

    source_reference = graph_features[:num_source]
    target_reference = graph_features[num_source:]

    all_rows: list[torch.Tensor] = []
    all_cols: list[torch.Tensor] = []
    all_values: list[torch.Tensor] = []

    def add_domain_neighbors(query_features,query_global_indices,reference_features,reference_global_offset,number_of_neighbors,reference_is_source):
        if number_of_neighbors == 0:
            return

        num_reference = reference_features.shape[0]
        k_effective = min(number_of_neighbors, num_reference)

        if k_effective == 0:
            return

        similarities = query_features @ reference_features.T

        # Remove self-connections during k-NN search.
        # Self-loops are explicitly added afterward.
        if reference_is_source:
            same_domain_mask = query_global_indices < num_source

            valid_rows = torch.where(same_domain_mask)[0]

            if valid_rows.numel() > 0:
                self_columns = query_global_indices[valid_rows]
                similarities[valid_rows, self_columns] = -torch.inf

        else:
            same_domain_mask = query_global_indices >= num_source

            valid_rows = torch.where(same_domain_mask)[0]

            if valid_rows.numel() > 0:
                self_columns = (
                    query_global_indices[valid_rows] - num_source
                )

                similarities[valid_rows, self_columns] = -torch.inf

        top_values, top_indices = torch.topk(
            similarities,
            k=k_effective,
            dim=1,
            largest=True,
            sorted=False,
        )

        valid_edges = torch.isfinite(top_values)

        if not valid_edges.any():
            return

        rows = query_global_indices[:, None].expand_as(top_indices)
        cols = top_indices + reference_global_offset

        edge_values = torch.exp((top_values - 1.0) / temperature)

        rows = rows[valid_edges]
        cols = cols[valid_edges]
        edge_values = edge_values[valid_edges]

        # Remove numerically zero edges.
        nonzero = edge_values > 0

        all_rows.append(rows[nonzero])
        all_cols.append(cols[nonzero])
        all_values.append(edge_values[nonzero])

    # Process query nodes in chunks to avoid storing the full N x N matrix.
    for start in range(0, num_nodes, chunk_size):
        end = min(start + chunk_size, num_nodes)

        query_features = graph_features[start:end]

        query_global_indices = torch.arange(start,end,device=device,dtype=torch.long)

        # Each query node receives source neighbors.
        add_domain_neighbors(
            query_features=query_features,
            query_global_indices=query_global_indices,
            reference_features=source_reference,
            reference_global_offset=0,
            number_of_neighbors=k_source,
            reference_is_source=True,
        )

        # Each query node receives target neighbors.
        add_domain_neighbors(
            query_features=query_features,
            query_global_indices=query_global_indices,
            reference_features=target_reference,
            reference_global_offset=num_source,
            number_of_neighbors=k_target,
            reference_is_source=False,
        )

    rows = torch.cat(all_rows)
    cols = torch.cat(all_cols)
    values = torch.cat(all_values)

    # ------------------------------------------------------------------
    # Symmetrize:
    #     A_ij = A_ji = average(A_ij, A_ji)
    # ------------------------------------------------------------------
    directed_indices = torch.stack([rows, cols], dim=0)

    reverse_indices = torch.stack([cols, rows], dim=0)

    symmetric_indices = torch.cat(
        [directed_indices, reverse_indices],
        dim=1,
    )

    symmetric_values = torch.cat(
        [values, values],
        dim=0,
    )

    value_sum = torch.sparse_coo_tensor(
        indices=symmetric_indices,
        values=symmetric_values,
        size=(num_nodes, num_nodes),
        device=device,
    ).coalesce()

    edge_count = torch.sparse_coo_tensor(
        indices=symmetric_indices,
        values=torch.ones_like(symmetric_values),
        size=(num_nodes, num_nodes),
        device=device,
    ).coalesce()

    symmetric_values = (
        value_sum.values() / edge_count.values()
    )

    adjacency = torch.sparse_coo_tensor(
        indices=value_sum.indices(),
        values=symmetric_values,
        size=(num_nodes, num_nodes),
        device=device,
    ).coalesce()

    # ------------------------------------------------------------------
    # Add self-loops.
    # ------------------------------------------------------------------
    
    diagonal = torch.arange(
            num_nodes,
            device=device,
            dtype=torch.long,
        )

    self_loop_indices = torch.stack(
            [diagonal, diagonal],
            dim=0,
        )

    combined_indices = torch.cat(
            [adjacency.indices(), self_loop_indices],
            dim=1,
        )

    combined_values = torch.cat(
            [
                adjacency.values(),
                torch.ones(
                    num_nodes,
                    dtype=adjacency.dtype,
                    device=device,
                ),
            ],
            dim=0,
        )

    adjacency = torch.sparse_coo_tensor(
            indices=combined_indices,
            values=combined_values,
            size=(num_nodes, num_nodes),
            device=device,
        ).coalesce()

    # ------------------------------------------------------------------
    # Symmetric GCN normalization:
    #     A_hat = D^{-1/2} A D^{-1/2}
    # ------------------------------------------------------------------
    indices = adjacency.indices()
    values = adjacency.values()

    row_indices = indices[0]
    col_indices = indices[1]

    degrees = torch.zeros(
            num_nodes,
            dtype=values.dtype,
            device=device,
        )

    degrees.scatter_add_(
            dim=0,
            index=row_indices,
            src=values,
        )

    inverse_sqrt_degree = degrees.clamp_min(1e-12).pow(-0.5)

    normalized_values = (
            values
            * inverse_sqrt_degree[row_indices]
            * inverse_sqrt_degree[col_indices]
        )

    adjacency = torch.sparse_coo_tensor(
            indices=indices,
            values=normalized_values,
            size=(num_nodes, num_nodes),
            device=device,
        ).coalesce()

    return adjacency

def cap_target_source_mass(adj,num_source,max_source_ratio = 0.5):

    adj = adj.coalesce()

    indices = adj.indices()
    values = adj.values().clone()

    rows, cols = indices
    num_nodes = adj.shape[0]
    num_target = num_nodes - num_source

    target_edge_mask = rows >= num_source
    target_rows = rows[target_edge_mask] - num_source
    target_cols = cols[target_edge_mask]
    target_values = values[target_edge_mask]

    source_neighbor_mask = target_cols < num_source
    target_neighbor_mask = target_cols >= num_source

    source_mass = torch.zeros(
        num_target,
        dtype=values.dtype,
        device=values.device,
    )

    target_mass = torch.zeros_like(source_mass)

    source_mass.scatter_add_(
        0,
        target_rows[source_neighbor_mask],
        target_values[source_neighbor_mask],
    )

    target_mass.scatter_add_(
        0,
        target_rows[target_neighbor_mask],
        target_values[target_neighbor_mask],
    )

    # We want:
    # scaled_source / (scaled_source + target_mass)
    # <= max_source_ratio.
    #
    # Therefore:
    # scaled_source <= alpha/(1-alpha) * target_mass.
    max_allowed_source_mass = (
        max_source_ratio
        / max(1.0 - max_source_ratio, 1e-12)
    ) * target_mass

    source_scale = torch.ones_like(source_mass)

    must_scale = source_mass > max_allowed_source_mass

    source_scale[must_scale] = (
        max_allowed_source_mass[must_scale]
        / source_mass[must_scale].clamp_min(1e-12)
    )

    # Scale only source -> target incoming edges.
    global_target_source_mask = (
        (rows >= num_source)
        & (cols < num_source)
    )

    local_target_rows = (
        rows[global_target_source_mask] - num_source
    )

    values[global_target_source_mask] *= (
        source_scale[local_target_rows]
    )

    return torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=adj.shape,
        dtype=adj.dtype,
        device=adj.device,
    ).coalesce()

def normalize_sparse_adjacency(adj):

    adj = adj.coalesce()

    indices = adj.indices()
    values = adj.values()

    rows = indices[0]
    cols = indices[1]

    num_nodes = adj.shape[0]

    degree = torch.zeros(
        num_nodes,
        dtype=values.dtype,
        device=values.device,
    )

    degree.scatter_add_(
        dim=0,
        index=rows,
        src=values,
    )

    degree_inv_sqrt = degree.clamp_min(1e-12).pow(-0.5)

    normalized_values = (
        values
        * degree_inv_sqrt[rows]
        * degree_inv_sqrt[cols]
    )

    return torch.sparse_coo_tensor(
        indices=indices,
        values=normalized_values,
        size=adj.shape,
        dtype=adj.dtype,
        device=adj.device,
    ).coalesce()
    
def gcn(train_x_new,val_x_new,train_y_new,budget,epoch):
    adj = build_source_target_adjacency(source_features=train_x_new,target_features=val_x_new)
    
    adj = cap_target_source_mass(
        adj=adj,
        num_source=len(train_x_new) - budget*epoch,
        max_source_ratio=0.4,
    )

    # 3. Normalize sau khi cap
    adj = normalize_sparse_adjacency(adj)
    

    result = train_and_select_with_gcn(
        source_features=train_x_new,
        target_features=val_x_new,
        labels=train_y_new,
        adj=adj,
        budget=budget,
        device=device,
        epoch=epoch)
    
    return torch.tensor(result)
    
def active_learning(train_x, train_y, val_x, val_y, one):

    #enhance the performance for a specific category
    train_y = np.where(train_y == one, 1, 0)
    val_y = np.where(val_y == one, 1, 0)
    one_train = train_y==1
    one_idx = val_y==1
    
    L2_WEIGHT = 1e-3
    C = 1 / (train_x[0].shape[0] * L2_WEIGHT)
    
    src_acc = []
    acc, acc_one, f1 = [], [], []
    aa, sa, fa, model = train_and_eval(train_x, train_y, val_x, val_y, one_idx, C)
    ori_pred = model.model.predict(val_x[one_idx])
    ori_one = sa
    
    src_acc.append(model.model.score(train_x[one_train], train_y[one_train]))
    
    # n = int(val_x.shape[0]*0.01)
    n = 100
    print("Budget: ", n)

    pred_ones = (model.model.predict(val_x)==1).astype(int)
    one_ratio = 1.0 
    qones = min(int(n*one_ratio), pred_ones.sum())

    if qones == 0:
        one_idxs = np.argpartition(model.model.predict_proba(val_x)[:, 1], -int(n*one_ratio))[-int(n*one_ratio):]
    else:
        one_idxs = np.random.choice(range(len(val_x)), qones, replace=False, p=pred_ones/np.sum(pred_ones))
    
    rest_idxs = np.setdiff1d(range(len(val_x)), one_idxs)
    rest_idxs = np.random.choice(rest_idxs, n-one_idxs.shape[0], replace=False)
    q_idxs = np.concatenate((one_idxs, rest_idxs))
    
    train_x_new, val_x_new, train_y_new, val_y_new, valid_x, valid_y = update_datasets(train_x, val_x, train_y, val_y, q_idxs)
    

    # no weights the first round 
    Ns = train_x.shape[0]
    n_t_l = q_idxs.shape[0]
    weight_BAL = None   
    aa, sa, fa, model = retrain_model(train_x_new, train_y_new, val_x, val_y, one_idx, C)
    acc.append(aa)
    acc_one.append(sa)
    f1.append(fa)
    src_acc.append(model.model.score(train_x[one_train], train_y[one_train]))
    
    
    selected_idx = []
    
    for i in range(4):
        q_idxs = gcn(train_x_new=train_x_new, val_x_new=val_x_new, train_y_new=train_y_new,budget=n,epoch=i)
        # q_idxs = np.argpartition(tar_infl, -n)[-n:]
        n_t_l += q_idxs.shape[0]
        weight_BAL = np.r_[np.ones(Ns), Ns/n_t_l*np.ones(n_t_l)] 
         
        train_x_new, val_x_new, train_y_new, val_y_new, valid_x, valid_y = update_datasets(train_x_new, val_x_new, train_y_new, val_y_new, q_idxs)      
        aa, sa, fa, model = retrain_model(train_x_new, train_y_new, val_x, val_y, one_idx, C, weights=weight_BAL)
        
        acc.append(aa)
        acc_one.append(sa)
        f1.append(fa)
        src_acc.append(model.model.score(train_x[one_train], train_y[one_train]))
    

    selected_labels = train_y_new[-n*5:]    
    pred = model.model.predict(val_x[one_idx])
    label = val_y[one_idx]

    sel_y_train = train_y_new[-n*5:]
    sel_x_train = train_x_new[-n*5:]

    ori_y_train = train_y_new[:-n*5]
    ori_x_train = train_x_new[:-n*5]

    sel_none = sel_y_train==0
    sel_y_train = sel_y_train[sel_none]
    sel_x_train = sel_x_train[sel_none]

    train_x_o = np.concatenate((ori_x_train, sel_x_train))
    train_y_o = np.concatenate((ori_y_train, sel_y_train))

    n_t_l = sel_y_train.shape[0]
    weight_BAL = np.r_[np.ones(Ns), Ns/n_t_l*np.ones(n_t_l)]
    return src_acc, acc, ori_one, acc_one, f1, ori_pred, pred, label, selected_labels, selected_idx

if __name__ == "__main__":

    seed = 0
    np.random.seed(seed)
    random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False


    train_x = np.load('./data_repr/'+tsk+'/source_emb.npy')
    train_y = np.load('./data_repr/'+tsk+'/source_lab.npy')
    val_x = np.load('./data_repr/'+tsk+'/target_emb.npy')
    val_y = np.load('./data_repr/'+tsk+'/target_lab.npy')
    
    
    logger.info("Load data succesful")

    ori_ones = []
    acc_ones = []
    src_accs = []
    selected_labels = []
    selected_idxs = []
    
    start = time.time()
    start_idx = 0
    for j in range(12):
        src_acc, acc, ori_one, acc_one, f1, ori_pred, pred, label, lbs, idxs = active_learning(train_x, train_y, val_x, val_y, j)
        ori_ones.append(ori_one)
        acc_ones.append(acc_one)
        src_accs.append(src_acc)
        selected_labels.append(lbs)
        selected_idxs.append(idxs)

        if j == start_idx:
            ori_preds = ori_pred
            preds = pred
            labels = label
        else:
            ori_preds = np.concatenate((ori_preds, ori_pred))
            preds = np.concatenate((preds, pred))
            labels = np.concatenate((labels, label))

        logging.info(f"Domain adaptation for class {j} successful")
        logging.info("Original accuracy %4f - Adaptation accuracy %4f", (ori_pred == label).sum() / len(ori_pred), (pred == label).sum() / len(pred))

    np.save("./log/" + tsk + "/herding_selected_idxs_" + timestamp + ".npy", selected_idxs)
    np.save("./log/" + tsk + "/ori_preds_" + timestamp + ".npy", ori_preds)
    np.save("./log/" + tsk + "/preds_" + timestamp + ".npy", preds)
    np.save("./log/" + tsk + "/labels_" + timestamp + ".npy", labels)
    logging.info("same start 1.0: %s", tsk)
    logging.info("Original accuracy %4f - Adaptation accuracy %4f", (ori_preds == labels).sum() / len(preds), (preds == labels).sum() / len(preds))
    logging.info("Total time: %4f", time.time()-start)
