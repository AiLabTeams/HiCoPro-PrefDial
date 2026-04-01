import argparse
import json
import random
from collections import defaultdict
from typing import List, Dict
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
import pandas as pd
import os

# 常量
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def set_global_seed():
    """
    设置全局随机种子，尽可能保证实验可复现
    """
    # 1. Python
    random.seed(SEED)
    # 2. NumPy
    np.random.seed(SEED)
    # 3. PyTorch (CPU)
    torch.manual_seed(SEED)
    # 4. PyTorch (GPU)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # 5. cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 6. 一些额外的确定性设置
    os.environ["PYTHONHASHSEED"] = str(SEED)


class LayerCurriculumScheduler:
    def __init__(
        self,
        base_layer_weights: List[float],
        start_epochs: List[int],
        ramp_epochs: int = 0,
    ):
        assert len(base_layer_weights) == len(
            start_epochs
        ), "base_layer_weights 和 start_epochs 长度必须一致（=层数）"
        self.base_layer_weights = [float(w) for w in base_layer_weights]
        self.start_epochs = [max(1, int(e)) for e in start_epochs]
        self.num_layers = len(self.base_layer_weights)
        self.ramp_epochs = int(ramp_epochs)

        for i in range(1, self.num_layers):
            if self.start_epochs[i] < self.start_epochs[i - 1]:
                print(
                    f"[LayerCurriculum] WARNING: 第 {i} 层的 start_epoch({self.start_epochs[i]}) "
                    f"早于上一层({self.start_epochs[i - 1]})，自动调整为后者。"
                )
                self.start_epochs[i] = self.start_epochs[i - 1]

    def get_layer_weights(self, epoch: int) -> List[float]:
        e = max(1, int(epoch))
        out = []
        for layer in range(self.num_layers):
            base_w = self.base_layer_weights[layer]
            start_e = self.start_epochs[layer]

            if e < start_e:
                factor = 0.0
            else:
                if self.ramp_epochs <= 0:
                    factor = 1.0
                else:
                    progress = (e - start_e + 1) / float(self.ramp_epochs)
                    factor = max(0.0, min(1.0, progress))
            out.append(base_w * factor)
        return out


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha=None,
        reduction: str = "mean",
        eps: float = 1e-9,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.eps = float(eps)
        self.ignore_index = ignore_index

        if alpha is None:
            self.alpha = None
        else:
            if isinstance(alpha, torch.Tensor):
                self.alpha = alpha.clone().detach()
            elif isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(list(alpha), dtype=torch.float32)
            else:
                self.alpha = torch.tensor(float(alpha), dtype=torch.float32)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.dim() != 2:
            raise ValueError("FocalLoss expects inputs with shape (B, C) logits.")

        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index

        logpt = F.log_softmax(inputs, dim=-1)
        pt = logpt.exp()

        targets = targets.long()
        logpt_t = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt_t = pt.gather(1, targets.unsqueeze(1)).squeeze(1)

        pt_t = pt_t.clamp(min=self.eps, max=1.0 - self.eps)
        logpt_t = logpt_t.clamp(max=-self.eps)

        mod_factor = (1.0 - pt_t).pow(self.gamma).detach()
        loss = -mod_factor * logpt_t

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            if alpha.dim() == 0:
                loss = alpha * loss
            else:
                at = alpha.gather(0, targets)
                loss = at * loss

        if valid_mask is not None:
            valid_mask_f = valid_mask.to(loss.dtype)
            loss = loss * valid_mask_f

        if self.reduction == "mean":
            if valid_mask is None:
                return loss.mean()
            else:
                denom = valid_mask.sum().clamp_min(1).to(loss.dtype)
                return loss.sum() / denom
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError("reduction must be 'mean','sum' or 'none'")


class GCNLayer(nn.Module):
    """Simple graph convolution layer: H' = D^{-1} A H W (with self-loop added)."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, H, adj):
        deg = adj.sum(dim=1)
        deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
        A_norm = adj * deg_inv.unsqueeze(1)
        out = A_norm @ H
        out = self.linear(out)
        return out


def read_graph(graph_path: str):
    with open(graph_path, "r", encoding="utf-8") as f:
        g = json.load(f)

    nodes = g.get("nodes", [])
    edges = g.get("edges", [])

    layer_nodes = defaultdict(list)
    node_name = {}
    node_layer = {}
    for n in nodes:
        nid = str(n["id"])
        layer = int(n["layer"])
        node_name[nid] = n.get("name", nid)
        node_layer[nid] = layer
        layer_nodes[layer].append(nid)

    for layer in layer_nodes:
        layer_nodes[layer] = sorted(layer_nodes[layer])

    layer_id2idx = {}
    layer_idx2id = {}
    for layer, ids in layer_nodes.items():
        layer_id2idx[layer] = {nid: i for i, nid in enumerate(ids)}
        layer_idx2id[layer] = {i: nid for i, nid in enumerate(ids)}

    adj = defaultdict(set)
    for src, dst in edges:
        adj[str(src)].add(str(dst))

    graph = {
        "node_name": node_name,
        "node_layer": node_layer,
        "layer_nodes": layer_nodes,
        "layer_id2idx": layer_id2idx,
        "layer_idx2id": layer_idx2id,
        "adj": adj,
    }
    return graph


def load_dataset_file(path: str, graph: Dict):
    data = []
    ext = Path(path).suffix.lower()
    if ext in [".jsonl", ".ndjson", ".json"]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj["text"]
                path_nodes = [str(x) for x in obj["path"]]
                data.append((text, path_nodes))
    else:
        import csv

        delim = "," if ext == ".csv" else "\t"
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delim)
            for row in reader:
                text = row.get("text") or row.get("Text")
                if "path" in row and row["path"]:
                    path_nodes = json.loads(row["path"])
                else:
                    path_nodes = [
                        row.get("path0"),
                        row.get("path1"),
                        row.get("path2"),
                        row.get("path3"),
                    ]

                # 转成字符串并替换空格为下划线，同时去掉空值
                path_nodes = [str(x).replace(" ", "_") for x in path_nodes if x]
                data.append((text, path_nodes))

    converted = []
    for text, path_nodes in data:
        if len(path_nodes) != 4:
            raise ValueError(
                f"Each path must have 4 node ids (got {len(path_nodes)}): {path_nodes}"
            )
        labels = []
        for layer, nid in enumerate(path_nodes):
            if nid not in graph["layer_id2idx"].get(layer, {}):
                raise KeyError(f"Node id {nid} not found in graph layer {layer}")
            labels.append(graph["layer_id2idx"][layer][nid])
        converted.append({"text": text, "labels": labels, "raw_path": path_nodes})
    return converted


class GraphAwareDataset(Dataset):
    def __init__(self, examples, tokenizer: AutoTokenizer, max_length: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        t = ex["text"]
        enc = self.tokenizer(
            t,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex["labels"], dtype=torch.long),
            "raw_path": ex["raw_path"],
        }
        return item


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    raw_paths = [b["raw_path"] for b in batch]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "raw_paths": raw_paths,
    }


class GraphAwareEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model: str,
        graph: Dict,
        hidden_dropout_prob: float = 0.1,
        gnn_layers: int = 1,
        gnn_init_from: str = "onehot",
        gnn_seed: int = SEED,
        use_gnn_in_forward: bool = True,
        use_node_embeddings: bool = True,  # <- 新增开关
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        self.config = self.encoder.config
        self.hidden_size = self.config.hidden_size

        self.layer_nodes = graph["layer_nodes"]
        self.num_classes_per_layer = {
            layer: len(nodes) for layer, nodes in self.layer_nodes.items()
        }
        self.heads = nn.ModuleDict()
        for layer, numc in self.num_classes_per_layer.items():
            head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(hidden_dropout_prob),
                nn.Linear(self.hidden_size, numc),
            )
            self.heads[str(layer)] = head

        # graph retention
        self.graph = graph

        # Node embeddings control
        self.use_node_embeddings = bool(use_node_embeddings)
        # if node embeddings are disabled, we cannot compute sim term -> disable gnn-in-forward
        if not self.use_node_embeddings and use_gnn_in_forward:
            print(
                "[GraphAwareEncoder] WARNING: use_gnn_in_forward requested but use_node_embeddings is False. Disabling use_gnn_in_forward."
            )
            use_gnn_in_forward = False

        # Node embeddings (per-layer) - only create when enabled
        if self.use_node_embeddings:
            self.node_embeddings = nn.ModuleDict()
            for layer, numc in self.num_classes_per_layer.items():
                emb = nn.Embedding(numc, self.hidden_size)
                nn.init.xavier_uniform_(emb.weight)
                self.node_embeddings[str(layer)] = emb

            # learnable scale for combining sim with head logits per layer
            self.node_emb_scales = nn.ParameterDict()
            for layer in self.num_classes_per_layer.keys():
                p = nn.Parameter(torch.tensor(1.0))
                self.node_emb_scales[str(layer)] = p

        # whether to run GNN in forward (only meaningful when node embeddings enabled)
        self.use_gnn_in_forward = bool(use_gnn_in_forward) and self.use_node_embeddings

        # only build global index / GNN modules if node embeddings enabled
        if self.use_node_embeddings:
            self._build_global_node_index_and_adj()
            gnn_hidden = self.hidden_size
            self._gnn_layers = nn.ModuleList(
                [
                    GCNLayer(self.hidden_size if i == 0 else gnn_hidden, gnn_hidden)
                    for i in range(gnn_layers)
                ]
            )
            # initialize node embeddings via small GNN forward
            self._initialize_node_embeddings_with_gnn(
                gnn_layers, gnn_init_from, gnn_seed
            )
        else:
            # set placeholders to avoid attribute errors elsewhere; training code will check hasattr
            # do NOT create _gnn_layers or node_embeddings when disabled
            pass

    def _compute_dynamic_node_embeddings(self):
        """Only called when node embeddings enabled."""
        device = next(self.parameters()).device
        ordered_layers = sorted(self.layer_nodes.keys())
        H_list = []
        for layer in ordered_layers:
            emb = self.node_embeddings[str(layer)]
            H_list.append(emb.weight.to(device))
        if len(H_list) == 0:
            return {}
        H = torch.cat(H_list, dim=0)
        for layer_module in self._gnn_layers:
            if isinstance(layer_module, GCNLayer):
                H = F.relu(layer_module(H, self._global_adj))
            else:
                H = F.relu(layer_module(H, self._global_edge_index))
        out = {}
        for layer_idx in ordered_layers:
            base = self.layer_base_idx[layer_idx]
            n = len(self.layer_nodes[layer_idx])
            out[layer_idx] = H[base : base + n]
        return out

    def _build_global_node_index_and_adj(self):
        layers_sorted = sorted(self.layer_nodes.keys())
        self.global_node_list = []
        self.layer_base_idx = {}
        base = 0
        for layer in layers_sorted:
            self.layer_base_idx[layer] = base
            ids = list(self.layer_nodes[layer])
            self.global_node_list.extend(ids)
            base += len(ids)
        self.num_global_nodes = len(self.global_node_list)
        self.nodeid_to_globalidx = {
            nid: i for i, nid in enumerate(self.global_node_list)
        }

        adj = torch.zeros(
            (self.num_global_nodes, self.num_global_nodes), dtype=torch.float32
        )
        for src, children in self.graph["adj"].items():
            if src not in self.nodeid_to_globalidx:
                continue
            si = self.nodeid_to_globalidx[src]
            for child in children:
                if child not in self.nodeid_to_globalidx:
                    continue
                ci = self.nodeid_to_globalidx[child]
                adj[si, ci] = 1.0
                adj[ci, si] = 1.0
        for i in range(self.num_global_nodes):
            adj[i, i] = 1.0
        self.register_buffer("_global_adj", adj)
        srcs, dsts = (adj > 0).nonzero(as_tuple=True)
        edge_index = torch.stack([srcs, dsts], dim=0)
        self.register_buffer("_global_edge_index", edge_index)

    def _initialize_node_embeddings_with_gnn(self, gnn_layers, gnn_init_from, gnn_seed):
        device = next(self.parameters()).device
        torch.manual_seed(gnn_seed)
        N = self.num_global_nodes
        if gnn_init_from == "onehot":
            H = torch.eye(N, device=device, dtype=torch.float32)
            if N != self.hidden_size:
                proj = nn.Linear(N, self.hidden_size).to(device)
                nn.init.xavier_uniform_(proj.weight)
                H = proj(H)
        else:
            H = torch.randn((N, self.hidden_size), device=device)
        H = H.to(device)
        for i, layer in enumerate(self._gnn_layers):
            if isinstance(layer, GCNLayer):
                H = F.relu(layer(H, self._global_adj))
            else:
                H = F.relu(layer(H, self._global_edge_index))
        for layer_idx, ids in self.layer_nodes.items():
            base = self.layer_base_idx[layer_idx]
            n = len(ids)
            sub = H[base : base + n]
            emb = self.node_embeddings[str(layer_idx)]
            with torch.no_grad():
                if sub.size(1) != emb.weight.size(1):
                    proj2 = nn.Linear(sub.size(1), emb.weight.size(1)).to(device)
                    nn.init.xavier_uniform_(proj2.weight)
                    sub = proj2(sub)
                emb.weight.data.copy_(sub)

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
        summed = (last_hidden * mask).sum(1)
        lens = mask.sum(1).clamp(min=1e-9)
        pooled = summed / lens
        return pooled

    def forward_heads(self, pooled):
        logits = {}
        # only compute dynamic embeddings if node embeddings enabled and gnn_in_forward set
        if self.use_node_embeddings and self.use_gnn_in_forward:
            dynamic_embs = self._compute_dynamic_node_embeddings()
        else:
            dynamic_embs = None

        for layer, head in self.heads.items():
            layer_i = int(layer)
            base_logit = head(pooled)  # (B, num_classes)
            if not self.use_node_embeddings:
                # node embeddings disabled -> return head logits only
                logits[layer_i] = base_logit
                continue

            if dynamic_embs is not None:
                W = dynamic_embs[layer_i]
            else:
                emb = self.node_embeddings[layer]
                W = emb.weight
            sim = pooled @ W.T
            scale = self.node_emb_scales[layer]
            logits[layer_i] = base_logit + scale * sim
        return logits


# train_one_epoch / get_allowed_mask_batch_by_gold_parents / evaluate / get_allowed_mask_batch_by_predicted_parents
# （这些函数保持不变，直接复制原逻辑 -- 在此省略重贴以节省篇幅）
# 为完整性我把它们放回去（代码中保持与原始版本一致），这里只在实际文件中保留完整实现。


def train_one_epoch(
    model: GraphAwareEncoder,
    dataloader: DataLoader,
    optimizer,
    scheduler=None,
    layer_weights=None,
    graph=None,
    focal_criterion=None,
):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        pooled = model.encode(input_ids, attention_mask)
        logits = model.forward_heads(pooled)

        losses = []
        for layer in range(4):
            logit = logits[layer]
            target = labels[:, layer]

            if layer > 0:
                parent_layer = layer - 1
                mask = get_allowed_mask_batch_by_gold_parents(
                    batch["raw_paths"], graph, parent_layer, layer
                )
                inf_mask = (~mask).to(logit.dtype) * (-1e9)
                logit = logit + inf_mask

                gold_logits = logit.gather(1, target.unsqueeze(1)).squeeze(1)
                masked_gold = gold_logits < -1e6

                if masked_gold.any():
                    target = target.clone()
                    target[masked_gold] = -100

            if target.min() < 0 or target.max() >= logit.size(1):
                print("🔥 INVALID TARGET DETECTED!")
                print("min target:", target.min().item())
                print("max target:", target.max().item())
                print("num_classes:", logit.size(1))
                bad_idx = (target < 0) | (target >= logit.size(1))
                print("bad targets:", target[bad_idx])
                raise ValueError("Target index out of range")

            if focal_criterion is not None:
                loss = focal_criterion(logit, target)
            else:
                loss = F.cross_entropy(logit, target)

            if layer_weights is not None:
                loss = loss * layer_weights[layer]
            losses.append(loss)

        loss = sum(losses)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def get_allowed_mask_batch_by_gold_parents(
    raw_paths: List[List[str]], graph: Dict, parent_layer: int, child_layer: int
):
    batch_size = len(raw_paths)
    num_child = len(graph["layer_nodes"].get(child_layer, []))
    mask = torch.zeros((batch_size, num_child), dtype=torch.bool, device=DEVICE)
    for i, raw_path in enumerate(raw_paths):
        parent_id = raw_path[parent_layer]
        allowed_children = graph["adj"].get(parent_id, set())
        for j, child_id in enumerate(graph["layer_nodes"].get(child_layer, [])):
            if child_id in allowed_children:
                mask[i, j] = True
    for i in range(batch_size):
        if not mask[i].any():
            mask[i] = True
    return mask


def evaluate(
    model, dataloader: DataLoader, graph: Dict, save_predictions_path: str = None
):
    model.eval()
    total = 0
    exact_match = 0
    all_preds = []

    preds_all_layers = [[] for _ in range(4)]
    labels_all_layers = [[] for _ in range(4)]
    hierarchical_consistencies = []

    eval_total_loss = 0.0
    eval_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            raw_paths = batch["raw_paths"]

            pooled = model.encode(input_ids, attention_mask)
            logits = model.forward_heads(pooled)

            losses = []
            for layer in range(4):
                logit = logits[layer]
                target = labels[:, layer]

                if layer > 0:
                    parent_layer = layer - 1
                    mask = get_allowed_mask_batch_by_gold_parents(
                        raw_paths, graph, parent_layer, layer
                    )
                    inf_mask = (~mask).to(logit.dtype) * (-1e9)
                    logit = logit + inf_mask

                loss = F.cross_entropy(logit, target)
                losses.append(loss)
            loss = sum(losses)
            eval_total_loss += loss.item()
            eval_batches += 1
            eval_avg_loss = eval_total_loss / max(1, eval_batches)

            batch_size = input_ids.size(0)
            preds = torch.zeros((batch_size, 4), dtype=torch.long, device=DEVICE)

            p0 = logits[0].argmax(dim=-1)
            preds[:, 0] = p0

            for layer in range(1, 4):
                logit = logits[layer]
                mask = get_allowed_mask_batch_by_predicted_parents(
                    preds[:, layer - 1], graph, layer - 1, layer
                )
                inf_mask = (~mask).to(logit.dtype) * (-1e9)
                logit = logit + inf_mask
                p = logit.argmax(dim=-1)
                preds[:, layer] = p

            total += batch_size
            exact_match += (preds == labels).all(dim=1).sum().item()

            for layer in range(4):
                preds_all_layers[layer].extend(preds[:, layer].cpu().tolist())
                labels_all_layers[layer].extend(labels[:, layer].cpu().tolist())

            for i in range(batch_size):
                pred_path = [
                    graph["layer_idx2id"][l][int(preds[i, l].item())] for l in range(4)
                ]
                gold_path = raw_paths[i]

                consistent_layers = 0
                for l in range(min(len(pred_path), len(gold_path))):
                    if pred_path[: l + 1] == gold_path[: l + 1]:
                        consistent_layers += 1
                hierarchical_consistencies.append(consistent_layers / 4)
                all_preds.append({"pred": pred_path, "gold": gold_path})

    layer_metrics = []
    for layer in range(4):
        y_true = labels_all_layers[layer]
        y_pred = preds_all_layers[layer]
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        layer_metrics.append((p, r, f1))

    exact_match_rate = exact_match / total if total > 0 else 0.0
    hierarchical_consistency = (
        np.mean(hierarchical_consistencies) if hierarchical_consistencies else 0.0
    )

    y_true_paths = [tuple(x["gold"]) for x in all_preds]
    y_pred_paths = [tuple(x["pred"]) for x in all_preds]

    if len(y_true_paths) == 0:
        path_macro_p = path_macro_r = path_macro_f1 = 0.0
    else:
        unique_paths = list({p for p in y_true_paths} | {p for p in y_pred_paths})
        path2id = {p: i for i, p in enumerate(unique_paths)}
        y_true_ids = [path2id[p] for p in y_true_paths]
        y_pred_ids = [path2id[p] for p in y_pred_paths]

        path_macro_p, path_macro_r, path_macro_f1, _ = precision_recall_fscore_support(
            y_true_ids, y_pred_ids, average="macro", zero_division=0
        )

    # ---------- 保存预测与标签（可选） ----------
    if save_predictions_path is not None:
        # 保存 per-layer y_true/y_pred (indices), 以及所有的 gold/pred paths (label ids)
        np.savez_compressed(
            save_predictions_path,
            preds_all_layers=preds_all_layers,
            labels_all_layers=labels_all_layers,
            all_paths=all_preds,  # list of dicts with 'pred' and 'gold'
        )
        # 也写一个 human-readable json copy（可选）
        try:
            with open(save_predictions_path + ".json", "w", encoding="utf-8") as jf:
                json.dump(all_preds, jf, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return (
        eval_avg_loss,
        path_macro_f1,
        exact_match_rate,
        hierarchical_consistency,
        layer_metrics,
    )


def get_allowed_mask_batch_by_predicted_parents(
    pred_parent_indices: torch.Tensor,
    graph: Dict,
    parent_layer: int,
    child_layer: int,
):
    batch_size = pred_parent_indices.size(0)
    num_child = len(graph["layer_nodes"].get(child_layer, []))
    mask = torch.zeros((batch_size, num_child), dtype=torch.bool, device=DEVICE)
    for i in range(batch_size):
        pidx = int(pred_parent_indices[i].item())
        parent_id = graph["layer_idx2id"][parent_layer][pidx]
        allowed_children = graph["adj"].get(parent_id, set())
        for j, child_id in enumerate(graph["layer_nodes"].get(child_layer, [])):
            if child_id in allowed_children:
                mask[i, j] = True
    for i in range(batch_size):
        if not mask[i].any():
            mask[i] = True
    return mask


def parse_layer_weights(layer_weights: str):
    parts = [float(x) for x in layer_weights.split(",")]
    if len(parts) != 4:
        raise ValueError("layer-weights must have 4 comma-separated floats")
    return parts


def train_model(args, model, tokenizer, graph):
    if not args.train_file or not args.val_file:
        raise ValueError("train_file and val_file are required for training")

    focal_criterion = None
    if getattr(args, "use_focal", False):
        alpha_arg = getattr(args, "focal_alpha", None)
        if alpha_arg is None or (
            isinstance(alpha_arg, str) and alpha_arg.lower() == "none"
        ):
            alpha_val = None
        else:
            parts = [p.strip() for p in alpha_arg.split(",")]
            if len(parts) == 1:
                alpha_val = float(parts[0])
            else:
                alpha_val = [float(p) for p in parts]
        focal_criterion = FocalLoss(
            gamma=args.focal_gamma, alpha=alpha_val, reduction="mean"
        )
        print(f"Using FocalLoss(gamma={args.focal_gamma}, alpha={alpha_val})")

    train_examples = load_dataset_file(args.train_file, graph)
    train_ds = GraphAwareDataset(train_examples, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    eval_loader = None
    base_layer_weights = parse_layer_weights(args.layer_weights)

    if args.val_file:
        eval_examples = load_dataset_file(args.val_file, graph)
        eval_ds = GraphAwareDataset(
            eval_examples, tokenizer, max_length=args.max_length
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        encoder_lr = float(args.encoder_lr)
        heads_lr = float(args.heads_lr)
        gnn_lr = (
            float(args.gnn_lr)
            if getattr(args, "gnn_lr", None) is not None
            else heads_lr
        )

        # 动态构建优化器参数组：仅在模型提供相关属性时加入
        optimizer_grouped_parameters = [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.heads.parameters(), "lr": heads_lr},
        ]
        # node_embeddings (if present)
        if getattr(model, "use_node_embeddings", False) and hasattr(
            model, "node_embeddings"
        ):
            optimizer_grouped_parameters.append(
                {"params": model.node_embeddings.parameters(), "lr": heads_lr}
            )
        # gnn layers (if present)
        if hasattr(model, "_gnn_layers"):
            optimizer_grouped_parameters.append(
                {"params": model._gnn_layers.parameters(), "lr": gnn_lr}
            )
        # node_emb_scales (if present)
        if getattr(model, "use_node_embeddings", False) and hasattr(
            model, "node_emb_scales"
        ):
            optimizer_grouped_parameters.append(
                {"params": list(model.node_emb_scales.values()), "lr": heads_lr}
            )

        optimizer = AdamW(optimizer_grouped_parameters)

        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.06 * total_steps),
            num_training_steps=total_steps,
        )

        layer_curriculum = None
        if getattr(args, "use_layer_curriculum", False):
            num_layers = len(base_layer_weights)
            if args.layer_curriculum_starts is not None:
                starts = [int(x) for x in args.layer_curriculum_starts.split(",")]
                if len(starts) != num_layers:
                    raise ValueError(
                        f"--layer_curriculum_starts 需要提供 {num_layers} 个数字，"
                        f"当前为 {len(starts)}: {args.layer_curriculum_starts}"
                    )
            else:
                gap = max(1, args.epochs // num_layers)
                starts = [1 + i * gap for i in range(num_layers)]
                starts = [min(s, args.epochs) for s in starts]

            layer_curriculum = LayerCurriculumScheduler(
                base_layer_weights=base_layer_weights,
                start_epochs=starts,
                ramp_epochs=args.layer_curriculum_ramp_epochs,
            )

            print(
                "[LayerCurriculum] 已启用层级式 curriculum：\n"
                f"  base_layer_weights = {base_layer_weights}\n"
                f"  start_epochs       = {starts}\n"
                f"  ramp_epochs        = {args.layer_curriculum_ramp_epochs}"
            )
        else:
            print("[LayerCurriculum] 未启用层级式 curriculum，使用固定 layer_weights。")

        history = []

        # 训练循环
        for epoch in range(1, args.epochs + 1):
            if layer_curriculum is not None:
                curr_layer_weights = layer_curriculum.get_layer_weights(epoch)
            else:
                curr_layer_weights = base_layer_weights

            print(
                f"[LayerCurriculum] Epoch {epoch}: layer_weights = {curr_layer_weights}"
            )

            train_avg_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler=scheduler,
                layer_weights=curr_layer_weights,
                graph=graph,
                focal_criterion=focal_criterion,
            )

            # 如果存在评估数据集，则进行评估
            if eval_loader is not None:
                (
                    eval_avg_loss,
                    path_macro_f1,
                    exact_match_rate,
                    hierarchical_consistency,
                    layer_metrics,
                ) = evaluate(model, eval_loader, graph)

                # # ⭐ 只在最好的 epoch 时保存预测结果
                # if epoch == 46:
                #     evaluate(
                #         model,
                #         eval_loader,
                #         graph,
                #         save_predictions_path="val_preds.npz",
                #     )

                print("\n========== 训练日志 ==========")
                print(f"Epoch {epoch}/{args.epochs}")
                print(f"Train Loss: {train_avg_loss:.4f}")
                print(f"Eval Loss: {eval_avg_loss:.4f}")
                print("========== 评估结果 ==========")
                print(f"整体路径 Macro F1: {path_macro_f1 * 100:.2f}%")
                print(f"精确匹配率 (Exact Match Rate): {exact_match_rate * 100:.2f}%")
                print(
                    f"层级一致性 (Hierarchical Consistency): {hierarchical_consistency * 100:.2f}%"
                )
                print("每层指标 (Per-Layer Macro F1):")
                for i, (p, r, f1) in enumerate(layer_metrics, 1):
                    print(f"  第{i}层 -> Macro F1: {f1 * 100:.2f}%")
                print("==============================")

                entry = {
                    "Epoch": epoch,
                    "Train Loss": round(train_avg_loss, 4),
                    "Eval Loss": round(eval_avg_loss, 4),
                    "Macro F1": round(path_macro_f1 * 100, 2),
                    "Exact Match": round(exact_match_rate * 100, 2),
                    "HC": round(hierarchical_consistency * 100, 2),
                }

                for i, (_, _, f1) in enumerate(layer_metrics, 1):
                    entry[f"Layer{i} F1"] = round(f1 * 100, 2)

                history.append(entry)
                print(history)

        df = pd.DataFrame(history)
        df.to_excel(
            f"{args.save_file_name}.xlsx",
            index=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train"], required=True)
    parser.add_argument("--pretrained", type=str, default="facebook/bart-base")
    parser.add_argument("--graph_file", type=str, required=True)
    parser.add_argument("--train_file", type=str, default="data/train.jsonl")
    parser.add_argument("--val_file", type=str, default="data/eval.jsonl")
    parser.add_argument(
        "--encoder_lr", type=str, default=6e-5, help="BART encoder学习率"
    )
    parser.add_argument(
        "--save_file_name", type=str, default="HiCoPro", help="保存文件名"
    )
    parser.add_argument("--heads_lr", type=str, default=6e-4, help="分类头学习率")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument(
        "--layer_weights", type=str, default="1.0, 1.0, 1.0, 1.0", help="每层权重"
    )
    parser.add_argument(
        "--gnn_layers",
        type=int,
        default=1,
        help="GNN 层数",
    )
    parser.add_argument(
        "--gnn_lr",
        type=str,
        default=None,
        help="GNN 和 node_embeddings 的学习率（若未指定则使用 heads_lr）",
    )

    # ===== Focal Loss 相关参数 =====
    parser.add_argument(
        "--use_focal",
        action="store_true",
        help="如果设置，将对每层损失使用 Focal Loss，而不是交叉熵。",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal Loss 的 gamma 参数（用于调节聚焦程度）。",
    )
    parser.add_argument(
        "--focal_alpha",
        type=str,
        default=None,
        help="Focal Loss 的 alpha 参数。可以传入标量（如 '0.25'）或按类逗号分隔的值（如 '0.1,0.2,0.7'）。默认值为 None。",
    )

    # ===== 层级式 Curriculum Learning 相关参数 =====
    parser.add_argument(
        "--use_layer_curriculum",
        action="store_true",
        help="是否启用“先训练浅层，再逐渐加入深层”的层级式 curriculum learning。",
    )
    parser.add_argument(
        "--layer_curriculum_starts",
        type=str,
        default=None,
        help=("每一层开始参与训练的 epoch（1-based），逗号分隔。"),
    )
    parser.add_argument(
        "--layer_curriculum_ramp_epochs",
        type=int,
        default=0,
        help=("某一层从开始参与训练到达到完整权重的线性爬坡轮数。"),
    )

    # ===== Node Embeddings 开关（新增） =====
    parser.add_argument(
        "--use_node_embeddings",
        dest="use_node_embeddings",
        action="store_true",
        help="启用 per-layer learnable node embeddings（默认启用）。",
    )
    parser.add_argument(
        "--no_node_embeddings",
        dest="use_node_embeddings",
        action="store_false",
        help="禁用 per-layer learnable node embeddings（只使用 head logits，不加 sim）。",
    )
    parser.set_defaults(use_node_embeddings=True)

    args = parser.parse_args()

    # 可复现性函数
    set_global_seed()

    graph = read_graph(args.graph_file)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    model = GraphAwareEncoder(
        args.pretrained,
        graph,
        gnn_layers=args.gnn_layers,
        use_gnn_in_forward=True,  # keep default behavior, GraphAwareEncoder 内部会与 use_node_embeddings 协调
        use_node_embeddings=args.use_node_embeddings,
    )
    model.to(DEVICE)

    if args.mode == "train":
        train_model(args, model, tokenizer, graph)
    else:
        raise ValueError("unknown mode")
