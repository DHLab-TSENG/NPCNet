import math
import numpy as np
import pandas as pd
from pandas import Series
from collections import Counter
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from utils import seed_everything, args

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        pe = torch.zeros(max_len, d_model)
        """ [[0, 1, 2, 3, ...]] """
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
        
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward = 2048, dropout = 0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = d_model)
        self.cat_embeddings = nn.ModuleList([nn.Embedding(2, d_model) for _ in range(31)] + 
                                            [nn.Embedding(10, d_model)])
        self.positional_encoding = PositionalEncoding(d_model, dropout = 0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model, nhead = nhead, dim_feedforward = dim_feedforward, dropout = dropout, batch_first = True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)

    def forward(self, src, src_key_padding_mask, static_var):
        x = self.embedding(src)
        x = self.positional_encoding(x)

        gender, comorbidities, age = static_var[:, 0], static_var[:, 1:-1], static_var[:, -1]

        auxiliary_embedding = (self.cat_embeddings[0](gender) + sum(
            e(comorbidities[:, i]) for i, e in enumerate(self.cat_embeddings[1:-1])
        ) + self.cat_embeddings[-1](age)) / (1 + 31 + 1)

        weight = 0.7
        x = weight * x + (1 - weight) * auxiliary_embedding.unsqueeze(1)
        memory = self.encoder(x, src_key_padding_mask = src_key_padding_mask)
        return memory

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward = 2048, dropout = 0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = d_model)
        self.cat_embeddings = nn.ModuleList([nn.Embedding(2, d_model) for _ in range(31)] + 
                                            [nn.Embedding(10, d_model)])
        self.positional_encoding = PositionalEncoding(d_model, dropout = 0)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model = d_model, nhead = nhead, dim_feedforward = dim_feedforward, dropout = dropout, batch_first = True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers = num_layers)
        self.predictor = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_key_padding_mask, static_var):
        x = self.embedding(tgt)
        x = self.positional_encoding(x)

        gender, comorbidities, age = static_var[:, 0], static_var[:, 1:-1], static_var[:, -1]

        auxiliary_embedding = (self.cat_embeddings[0](gender) + sum(
            e(comorbidities[:, i]) for i, e in enumerate(self.cat_embeddings[1:-1])
        ) + self.cat_embeddings[-1](age)) / (1 + 31 + 1)

        weight = 0.7
        x = weight * x + (1 - weight) * auxiliary_embedding.unsqueeze(1)
        output = self.decoder(tgt = x, memory = memory, tgt_key_padding_mask = tgt_key_padding_mask)
        return output

class ClusteringAlgorithm(object):
    def __init__(self):
        self.n_features = args["latent_dim"]
        self.n_clusters = args["n_clusters"]
        self.clusters = np.zeros((self.n_clusters, self.n_features))
        self.counts = 100 * np.ones((self.n_clusters))

    def ComputeDist(self, X):
        diff = X[:,np.newaxis,:] - self.clusters[np.newaxis,:,:]
        dist_matrix = np.linalg.norm(diff, axis = 2)
        return dist_matrix

    def InitCluster(self, X):
        model = KMeans(n_clusters = self.n_clusters, n_init = 20)
        model.fit(X)
        self.clusters = model.cluster_centers_

        cluster_counts = Counter(model.labels_)
        for i, count in enumerate(cluster_counts.items()):
            print(f"Cluster {i}: {count} samples")

    def UpdateCluster(self, X, cluster_idx):
        for i in range(X.shape[0]):
            self.counts[cluster_idx] += 1
            eta = 1.0 / self.counts[cluster_idx]
            self.clusters[cluster_idx] = eta * X[i] + (1 - eta) * self.clusters[cluster_idx]

    def UpdateAssignment(self, X):
        dist_matrix = self.ComputeDist(X)

        return np.argmin(dist_matrix, axis = 1)

class FocalLoss(nn.Module):
    def __init__(self, weight, gamma = 2):
        super(FocalLoss,self).__init__()
        self.gamma = gamma
        self.weight = weight.to(args["device"])

    def forward(self, preds, labels):
        preds = F.softmax(preds, dim = 1)
        eps = 1e-7

        target = self.OneHot(preds.size(1), labels).to(args["device"])

        ce = -1 * torch.log(preds + eps) * target
        floss = torch.pow((1 - preds), self.gamma) * ce
        floss = torch.mul(floss, self.weight)
        floss = torch.sum(floss, dim = 1)
        return torch.mean(floss)

    def OneHot(self, num, labels):
        one = torch.zeros((labels.size(0), num))
        one[range(labels.size(0)), labels] = 1
        return one

class Linear(nn.Module):
    def __init__(self, embedding_dim):
        super(Linear, self).__init__()
        self.fc = nn.Linear(embedding_dim, 2)

    def forward(self, x):
        x = self.fc(x)
        return x

def evaluator(preds, probs, gts):
    preds = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
    gts = gts.cpu().numpy() if isinstance(gts, torch.Tensor) else gts
    acc = accuracy_score(preds, gts)
    auc = roc_auc_score(gts, probs[:, 1].cpu().detach().numpy())
    f1 = f1_score(preds, gts)

    return acc, auc, f1

class ClusteringModule(nn.Module):
    def __init__(self, args):
        super(ClusteringModule, self).__init__()
        self.clustering = ClusteringAlgorithm()
        self.encoder = Encoder(vocab_size = args["vocab_size"], d_model = args["latent_dim"], nhead = 4, num_layers = 2).to(args["device"])
        self.decoder = Decoder(vocab_size = args["vocab_size"], d_model = args["latent_dim"], nhead = 4, num_layers = 2).to(args["device"])
        self.fc = Linear(embedding_dim = args["latent_dim"]).to(args["device"])
        self.criterion2rec = nn.CrossEntropyLoss(ignore_index = 0)
        self.criterion2prob = FocalLoss(torch.tensor([0.2, 0.8]).to(args["device"]))
        self.criterion2dist = torch.nn.TripletMarginLoss(margin = 1.0, p = 2)
        self.optimizer = torch.optim.Adam(self.parameters(), lr = args["lr"], weight_decay = args["wd"])

    def LossFunction(self, text, attention_mask, static_var, positive_text, positive_attention_mask, positive_static_var, negative_text, negative_attention_mask, negative_static_var, y, cluster_idx):
        batch_size = text.size()[0]
        latent_X = self.encoder(text, attention_mask, static_var)

        tgt = text[:, :-1]
        tgt_y = text[:, 1:]
        tgt_attention_mask = attention_mask[:, :-1]
        text_rec = self.decoder(tgt, latent_X, tgt_attention_mask, static_var)
        text_rec = self.decoder.predictor(text_rec)

        """ reconstruction loss """
        rec_loss = args["lambda1"] * self.criterion2rec(text_rec.contiguous().view(-1, text_rec.size(-1)), tgt_y.contiguous().view(-1))

        latent_X2 = latent_X.mean(dim = 1)
        """ clustering loss """
        clustering_loss = torch.tensor(0.).to(args["device"])
        clusters = torch.FloatTensor(self.clustering.clusters).to(args["device"])
        for i in range(batch_size):
            cont_diff = latent_X2[i] - clusters[cluster_idx[i]]
            batch_clustering_loss = torch.matmul(cont_diff.view(1, -1), cont_diff.view(-1, 1))
            clustering_loss += args["lambda2"] * 0.5 * torch.squeeze(batch_clustering_loss)
        
        """ probability loss """
        pred_y = self.fc(latent_X2)
        prob_loss = args["lambda3"] * args["kappa1"] * self.criterion2prob(pred_y, y)

        """ distance loss """
        positive_X = self.encoder(positive_text, positive_attention_mask, positive_static_var)
        positive_X = positive_X.mean(dim = 1)
        negative_X = self.encoder(negative_text, negative_attention_mask, negative_static_var)
        negative_X = negative_X.mean(dim = 1)
        triplet_loss = args["lambda3"] * args["kappa2"] * self.criterion2dist(latent_X2, positive_X, negative_X)

        return (rec_loss + clustering_loss + prob_loss + triplet_loss,
                rec_loss.detach().cpu().numpy(),
                clustering_loss.detach().cpu().numpy(),
                prob_loss.detach().cpu().numpy(),
                triplet_loss.detach().cpu().numpy(),
                torch.argmax(pred_y, dim = 1),
                pred_y)

    def pretrain(self, pretraining_epoch, train_loader):
        self.train()
        for epoch in range(pretraining_epoch):
            for batch_idx, (text, attention_mask, static_var, _, _, _, _, _, _, _) in enumerate(train_loader):
                batch_size = text.size()[0]
                text = text.to(args["device"])
                attention_mask = attention_mask.to(args["device"])
                static_var = static_var.to(args["device"])

                latent_X = self.encoder(text, attention_mask, static_var)

                tgt = text[:, :-1]
                tgt_y = text[:, 1:]
                tgt_attention_mask = attention_mask[:, :-1]
                text_rec = self.decoder(tgt, latent_X, tgt_attention_mask, static_var)
                text_rec = self.decoder.predictor(text_rec)

                loss = self.criterion2rec(text_rec.contiguous().view(-1, text_rec.size(-1)), tgt_y.contiguous().view(-1))

                if batch_idx % 100 == 0:
                    print("Epoch: %02d | Batch: %03d | RecLoss: %.3f" % (epoch, batch_idx, loss.detach().cpu().numpy()))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                                        
        self.eval()
        """ initialize clusters after pretraining """
        batch_X = []
        for batch_idx, (text, attention_mask, static_var, _, _, _, _, _, _, _) in enumerate(train_loader):
            batch_size = text.size()[0]
            text = text.to(args["device"])
            attention_mask = attention_mask.to(args["device"])
            static_var = static_var.to(args["device"])

            latent_X = self.encoder(text, attention_mask, static_var)
            latent_X = latent_X.mean(dim = 1)
            batch_X.append(latent_X.detach().cpu().numpy())

        batch_X = np.vstack(batch_X)
        self.clustering.InitCluster(batch_X)

    def fit(self, epoch, train_loader):
        train_loss = .0
        preds, probs, gts = [], [], []
        for batch_idx, (text, attention_mask, static_var, positive_text, positive_attention_mask, positive_static_var, negative_text, negative_attention_mask, negative_static_var, y) in enumerate(train_loader):
            batch_size = text.size()[0]
            text = text.to(args["device"])
            attention_mask = attention_mask.to(args["device"])
            static_var = static_var.to(args["device"])
            positive_text = positive_text.to(args["device"])
            positive_attention_mask = positive_attention_mask.to(args["device"])
            positive_static_var = positive_static_var.to(args["device"])
            negative_text = negative_text.to(args["device"])
            negative_attention_mask = negative_attention_mask.to(args["device"])
            negative_static_var = negative_static_var.to(args["device"])
            y = y.to(args["device"])

            latent_X = self.encoder(text, attention_mask, static_var)
            latent_X = latent_X.mean(dim = 1)
            latent_X = latent_X.cpu().detach().numpy()

            cluster_idx = self.clustering.UpdateAssignment(latent_X)

            elem_count = np.bincount(cluster_idx, minlength = args["n_clusters"])

            for k in range(args["n_clusters"]):
                if elem_count[k] == 0:
                    continue
                self.clustering.UpdateCluster(latent_X[cluster_idx == k], k)
            
            loss, rec_loss, clustering_loss, prob_loss, triplet_loss, pred_y, prob = self.LossFunction(text, attention_mask, static_var, positive_text, positive_attention_mask, positive_static_var, negative_text, negative_attention_mask, negative_static_var, y, cluster_idx)
            train_loss += prob_loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            preds.append(pred_y)
            probs.append(prob)
            gts.append(y)

            if batch_idx % 100 == 0:
                print("Epoch: %02d | Batch: %03d | Loss: %.3f | RecLoss: %.3f | ClusteringLoss: %.3f | ProbLoss: %.3f | DistLoss: %.3f" % (epoch, batch_idx, loss.detach().cpu().numpy(), rec_loss, clustering_loss, prob_loss, triplet_loss))

        train_loss /= len(train_loader)
        preds, probs, gts = torch.cat(preds), torch.cat(probs), torch.cat(gts)
        train_acc, train_auc, train_f1 = evaluator(preds, probs, gts)
        print("ProbLoss: %.4f | ACC: %.2f% | AUC: %.2f% | F1 score: %.2f%" % (train_loss, 100*train_acc, 100*train_auc, 100*train_f1))

        return self.clustering.clusters

def result(inputs, masks, statics, selected_idx, single_points, model, cluster_centroids):
    model.eval()
    
    column_names = ["subject_id","event"]
    for i in range(args["latent_dim"]): column_names.append("embedding" + str(i+1))
    output = pd.DataFrame(data = None, columns = column_names)
    
    dist_matrix = np.zeros((len(inputs), 4))
    for i in range(len(inputs)):
        text = inputs[i].to(args["device"])
        mask = masks[i].to(args["device"])
        cat = statics[i].to(args["device"])
        feat = model.encoder(torch.unsqueeze(text, 0), torch.unsqueeze(mask, 0), torch.unsqueeze(cat, 0))
        feat = feat.mean(dim = 1)
        feat = feat.flatten().cpu().detach().numpy()
        output.loc[str(i)] = np.append(single_points.loc[selected_idx[i]][["subject_id", "event"]].values, feat)
        for j in range(4):
            dist_matrix[i, j] += np.sqrt(np.sum((feat - cluster_centroids[j]) ** 2, axis = 0))
    
    output["cluster"] = np.argmin(dist_matrix, axis = 1)
    print(Series(np.argmin(dist_matrix, axis = 1)).value_counts())

    return output