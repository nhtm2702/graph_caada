import numpy as np
from utils import *
from graph_model import *
import copy


def train_binary_graph(graph,source_features,target_features,labels,num_labeled_target,adj,avg_degree,device):

    source_features = torch.as_tensor(source_features,dtype=torch.float32,device=device)
    target_features = torch.as_tensor(target_features,dtype=torch.float32,device=device)
    labels = torch.as_tensor(labels,dtype=torch.float32,device=device).view(-1)

    num_source = source_features.shape[0] - num_labeled_target
    
    print("Shape của source domain sau khi adding budget:", source_features.shape)
    print("Số lượng của source domain ban đầu:", num_source)
    print("Shape của target domain sau khi delete budget:", target_features.shape)
    print("Số lượng sample có nhãn sau khi adding budget:", labels.shape)


    adj = adj.to(device)
    if adj.is_sparse:
        adj = adj.coalesce()

    node_features = torch.cat([source_features, target_features],dim=0)
    node_features = F.normalize(node_features,p=2,dim=1,eps=1e-12)

    source_labels = labels[:num_source]
    labeled_target_labels = labels[num_source:]
    source_global_indices = torch.arange(num_source,dtype=torch.long,device=device)
    labeled_target_global_indices = torch.arange(num_source,num_source+num_labeled_target,dtype=torch.long,device=device)

    if graph == "gcn":
        model = BinaryGCN(nfeat=node_features.shape[1],nhid=128,dropout=0.5).to(device)
    if graph == "graphsage":
        model = BinaryGraphSAGE(nfeat=node_features.shape[1],nhid=128,dropout=0.5).to(device)
    if graph == "pna":
        model = BinaryPNA(nfeat=node_features.shape[1],nhid=128,dropout=0.5,avg_degree=avg_degree).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=5e-4)

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    history = {"loss": [],"source_loss": [],"target_loss": []}

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()

        logits, _ = model(node_features, adj)

        source_loss = balanced_binary_loss(logits[source_global_indices],source_labels)

        if num_labeled_target > 0:
            target_loss = balanced_binary_loss(
                logits[labeled_target_global_indices],
                labeled_target_labels,
            )

            loss = (0.3 * source_loss + 2.0 * target_loss)
        else:
            target_loss = torch.zeros((),dtype=source_loss.dtype,device=device)

            loss = 0.3 * source_loss

        loss.backward()

        optimizer.step()

        current_loss = float(loss.detach().cpu())

        history["loss"].append(current_loss)
        history["source_loss"].append(float(source_loss.detach().cpu()))
        history["target_loss"].append(float(target_loss.detach().cpu()))

        if current_loss < best_loss - 1e-7:
            best_loss = current_loss
            best_state = copy.deepcopy(model.state_dict())
        if epoch%100==0:
            print("Epoch", epoch, "Loss: ", loss.item())

    model.load_state_dict(best_state)

    metadata = {
        "node_features": node_features,
        "history": history,
        "best_loss": best_loss,
        "epochs_trained": len(history["loss"])
    }

    return model, metadata


@torch.no_grad()
def infer_unlabeled_target(model,node_features,adj,num_source):

    model.eval()
    logits, graph_embeddings = model(node_features,adj)

    target_logits = logits[num_source:]

    target_probabilities = torch.sigmoid(target_logits)
    target_uncertainty = binary_entropy(target_probabilities)
    target_graph_embeddings = graph_embeddings[num_source:]
    print("Shape của uncertainty target: ", target_uncertainty.shape)
    print("Shape của embedding target: ", target_graph_embeddings.shape)

    ranking_order = torch.argsort(target_uncertainty, descending=True)

    return {
        # Target indices ranked from most to least uncertain
        "ranked_target_indices":
            ranking_order.detach().cpu(),

        # Keep on current device for diversity selection
        "target_graph_embeddings":
            target_graph_embeddings,

        "target_uncertainty_device":
            target_uncertainty,
    }

@torch.no_grad()
def select_target_batch(target_uncertainty,target_graph_embeddings,budget,num_labeled_target,device):
    
    num_target = target_graph_embeddings.shape[0]
    all_indices = torch.arange(num_target,dtype=torch.long,device=device)

    available_mask = torch.ones(num_target,dtype=torch.bool,device=device)
    available_mask[:num_labeled_target] = False

    covered_indices = all_indices[:num_labeled_target]
    selected_indices = []
    selected_scores = []

    for _ in range(budget):
        candidate_indices = all_indices[available_mask]
        candidate_embeddings = target_graph_embeddings[candidate_indices]
        candidate_uncertainty = target_uncertainty[candidate_indices]

        if covered_indices.numel() > 0:
            covered_embeddings = target_graph_embeddings[covered_indices]
            distances = torch.cdist(candidate_embeddings,covered_embeddings,p=2)

            min_distance = distances.min(dim=1).values

            distance_min = min_distance.min()
            distance_max = min_distance.max()

            normalized_distance = (min_distance - distance_min) / (distance_max - distance_min).clamp_min(1e-8)

            # Avoid making the smallest-distance sample exactly zero.
            normalized_distance = (0.1 + 0.9 * normalized_distance)
        else:
            normalized_distance = torch.ones_like(candidate_uncertainty)

        acquisition_score = (candidate_uncertainty * normalized_distance.pow(1.0))
        best_local_index = torch.argmax(acquisition_score)
        best_target_index = candidate_indices[best_local_index]

        selected_indices.append(int(best_target_index.item()  - num_labeled_target))
        selected_scores.append(float(acquisition_score[best_local_index].item()))
        available_mask[best_target_index] = False
        covered_indices = torch.cat([covered_indices,best_target_index.view(1)],dim=0)

    # return {
    #     "selected_target_indices": selected_indices,
    #     "selected_acquisition_scores": selected_scores,
    # }
    return selected_indices

def train_and_select_with_gcn(graph,source_features,target_features,labels,adj,avg_degree,budget,device,epoch):
    
    print("Active round ", epoch)

    num_labeled_target = budget*epoch
    print("Số lượng target có nhãn: ", num_labeled_target)
    model, metadata = train_binary_graph(graph,source_features,target_features,labels,num_labeled_target,adj,avg_degree,device)
    print("Done training graph")
    
    num_source = len(source_features) - num_labeled_target
    inference = infer_unlabeled_target(
        model=model,
        node_features=metadata["node_features"],
        adj=adj.to(device),
        num_source=num_source
    )

    selection = select_target_batch(target_uncertainty=(inference["target_uncertainty_device"]),target_graph_embeddings=(inference["target_graph_embeddings"]),budget=budget,num_labeled_target=num_labeled_target,device=device)
    return selection