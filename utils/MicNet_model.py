import math
import torch
from torch import nn
import torch.utils
import torch.utils.data
import torch.nn.functional as F
import time
from scipy.stats import spearmanr
from alias_multinomial import AliasMethod
from utils_eval import AverageMeter
from alias_multinomial import AliasMethod
from utils_data import normalize, augmentor
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import skimage
import skimage.transform


class MicNet(torch.nn.Module):
    def __init__(self, image_shape, n_genes, n_out_features, 
                 imagenet='resnet101', 
                 genenet=[500, 100]):
        """
        Args:
            image_shape: w/h of input image
            n_genes: number of genes to consider
            n_out_features: number of output features
            imagenet: architecture to encode image features
            genenet: architecture to encode genetic features: list of #hidden nodes
        """
        super(MicNet, self).__init__()

        if imagenet == 'resnet101':
            # Require image size at least 224
            self.imagenet = torch.hub.load('pytorch/vision:v0.4.0', imagenet, pretrained=True)
            self.imagenet.float()
            self.imagenet.layer4[2].relu = torch.nn.ReLU6()
            self.imagenet.fc = torch.nn.Linear(self.imagenet.fc.in_features, n_out_features)
            input_size = image_shape
        elif imagenet == 'inception_v3':
            # Require image input size = 299, has auxillary output
            self.imagenet = torch.hub.load('pytorch/vision:v0.4.0', imagenet, pretrained=True)
            self.imagenet.float()
            raise("Incomplete model")
        else:
            raise("Invalid model name")

        genenet_fcs = []
        genenet_in_shape = n_genes
        for i, genenet_n_hidden_nodes in enumerate(genenet):
            genenet_fcs.append(torch.nn.Linear(genenet_in_shape, genenet_n_hidden_nodes))
            genenet_fcs.append(torch.nn.BatchNorm1d(genenet_n_hidden_nodes))
            genenet_fcs.append(torch.nn.ReLU6())
            genenet_in_shape = genenet_n_hidden_nodes
        genenet_fcs.append(torch.nn.Linear(genenet_in_shape, n_out_features))
        self.genenet_fcs = torch.nn.ModuleList(genenet_fcs)

    def forward(self, data):
        """
        Args:
            data: a dictionary
        """
        image, gene = data['image'], data['gene']
        f_image = self.imagenet(image)
        
        f_gene = gene
        for layer in self.genenet_fcs:
            f_gene = layer(f_gene)
        
        return f_image, f_gene


    
# Get image patch
def get_image_patch(image, coords, patch_size):
    """
    Args:
        coords: center of image; x, y (col, row) in pixels
    """
    row_start = int(coords[1] - patch_size/2)
    col_start = int(coords[0] - patch_size/2)
    return image[row_start:row_start + patch_size, col_start:col_start + patch_size]

class Dataset(torch.utils.data.Dataset):
    __initialized = False
    def __init__(self, indexes, image, count_df, meta_df, augmentation=False):
        """
        Args:
            indexes: index used for both count_df and meta_df
        """
        self.indexes = indexes
        self.image = image
        self.count_df = count_df
        self.meta_df = meta_df
        self.augmentation = augmentation
        self.__initialized = True

    def __len__(self):
        """Denotes the number of samples"""
        return len(self.indexes)
    
    def __getitem__(self, index):
        """Generate one batch of data.
        
        Returns:
            index: indexes of samples (long)
        """
        # Generate data
        data = self.__data_generation(self.indexes[index])

        return data, index
    
    def __data_generation(self, indexes):
        """Generates data containing batch_size samples.
        
        Returns:
            data: a dictionary with data.image in [b, ch, h, w]; data.gene in [b, n_genes]
        """
        PATCH_SIZE=256
        image = get_image_patch(self.image, 
                                [self.meta_df.loc[indexes, "Y"], self.meta_df.loc[indexes, "X"]], 
                                PATCH_SIZE)  # meta_df mistakes X/Y
        if self.augmentation:
            image = augmentor(normalize(image))
        else:
            image = normalize(image)
        
        data = dict()
        data['image'] = torch.tensor(np.transpose(image, (2, 0, 1)).astype(float))
        data['gene'] = torch.tensor(self.count_df.loc[indexes, :].values)
        
        return data 


class NCEAverage(nn.Module):

    def __init__(self, input_size, output_size, K, T=0.07, momentum=0.5, use_softmax=False, 
                 device="cuda: 0"):
        """
        Args:
            input_size: n_features
            output_size: n_samples
            K: number of negatives to constrast for each positive
            T: temperature that modulates the distribution
        """
        super(NCEAverage, self).__init__()
        self.output_size = output_size
        self.unigrams = torch.ones(self.output_size)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.to(device)
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(input_size / 3)
        self.register_buffer('memory_image', torch.rand(output_size, input_size).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_gene', torch.rand(output_size, input_size).mul_(2 * stdv).add_(-stdv))

    def forward(self, image, gene, index, idx=None):
        """
        Args:
            image: out_features for image
            gene: out_features for gene
            index: torch.long for data ids
        """
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_image = self.params[2].item()  # Normalization constant for image output
        Z_gene = self.params[3].item()  # Normalization constant for gene output

        momentum = self.params[4].item()
        batch_size = image.size(0)
        output_size = self.memory_image.size(0)
        input_size = self.memory_image.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batch_size * (self.K + 1)).view(batch_size, -1)
            idx.select(1, 0).copy_(index.data)
        # sample
        weight_gene = torch.index_select(self.memory_gene, 0, idx.view(-1)).detach()
        weight_gene = weight_gene.view(batch_size, K + 1, input_size)
        out_image = torch.bmm(weight_gene, image.view(batch_size, input_size, 1))
        # sample
        weight_image = torch.index_select(self.memory_image, 0, idx.view(-1)).detach()
        weight_image = weight_image.view(batch_size, K + 1, input_size)
        # Batchwise matrix multiplication
        # weight_image: [batch_size, K + 1, n_out_features]
        # gene: [batch_size, n_out_features]
        # out_gene:[batch_size, K + 1, 1]
        out_gene = torch.bmm(weight_image, gene.view(batch_size, input_size, 1))

        if self.use_softmax:
            out_image = torch.div(out_image, T)
            out_gene = torch.div(out_gene, T)
            out_image = out_image.contiguous()
            out_gene = out_gene.contiguous()
        else:
            out_image_e = torch.exp(out_image - torch.max(out_image, dim=1, keepdim=True)[0])
            out_image_s = torch.sum(out_image_e, dim=1, keepdim=True)
            out_image = torch.div(out_image_e, out_image_s)
            
            out_gene_e = torch.exp(out_gene - torch.max(out_gene, dim=1, keepdim=True)[0])
            out_gene_s = torch.sum(out_gene_e, dim=1, keepdim=True)
            out_gene = torch.div(out_gene_e, out_gene_s)
            """
            out_image = torch.exp(torch.div(out_image, T))
            out_gene = torch.exp(torch.div(out_gene, T))
            # set Z_0 if haven't been set yet,
            # Z_0 is used as a constant approximation of Z, to scale the probs
            if Z_image < 0:
               self.params[2] = out_image.mean() * output_size
                Z_image = self.params[2].clone().detach().item()
                print("normalization constant Z_image is set to {:.1f}".format(Z_image))
            if Z_gene < 0:
                self.params[3] = out_gene.mean() * output_size
                Z_gene = self.params[3].clone().detach().item()
                print("normalization constant Z_gene is set to {:.1f}".format(Z_gene))
            # compute out_image, out_gene
            out_image = torch.div(out_image, Z_image).contiguous()
            out_gene = torch.div(out_gene, Z_gene).contiguous()
            """

        # # update memory
        with torch.no_grad():
            image_pos = torch.index_select(self.memory_image, 0, index.view(-1))
            image_pos.mul_(momentum)
            image_pos.add_(torch.mul(image, 1 - momentum))
            image_norm = image_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_image = image_pos.div(image_norm)
            self.memory_image.index_copy_(0, index, updated_image)

            gene_pos = torch.index_select(self.memory_gene, 0, index.view(-1))
            gene_pos.mul_(momentum)
            gene_pos.add_(torch.mul(gene, 1 - momentum))
            gene_norm = gene_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_gene = gene_pos.div(gene_norm)
            self.memory_gene.index_copy_(0, index, updated_gene)

        return out_image, out_gene

import time
from scipy.stats import spearmanr

from utils_eval import AverageMeter

def train(epoch, train_loader, model, contrast, criterion_image, criterion_gene, optimizer, 
          gradient_clip=10, print_freq=1, device=torch.device("cuda:0")):
    """
    One epoch training
    """
    model.train()
    contrast.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    image_loss_meter = AverageMeter()
    gene_loss_meter = AverageMeter()
    image_prob_meter = AverageMeter()
    gene_prob_meter = AverageMeter()

    end = time.time()
    for idx, (data, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        batch_size = data['image'].size(0)
        index = index.to(device)
        for _ in data.keys():
            data[_] = data[_].float().to(device)

        # ===================forward=====================
        feat_image, feat_gene = model(data)
        out_image, out_gene = contrast(feat_image, feat_gene, index)
        
        # print("features:", feat_image, feat_gene, "\n")
        # print("outs: ", out_image, out_gene, "\n")
        
        image_loss = criterion_image(out_image)
        gene_loss = criterion_gene(out_gene)
        image_prob = out_image[:, 0].mean()
        gene_prob = out_gene[:, 0].mean()

        loss = image_loss + gene_loss

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        torch.nn.utils.clip_grad_norm_(contrast.parameters(), gradient_clip)
        optimizer.step()

        # ===================meters=====================
        losses.update(loss.item(), batch_size)
        image_loss_meter.update(image_loss.item(), batch_size)
        image_prob_meter.update(image_prob.item(), batch_size)
        gene_loss_meter.update(gene_loss.item(), batch_size)
        gene_prob_meter.update(gene_prob.item(), batch_size)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  # 'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  # 'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'image_p {image_probs.val:.3f} ({image_probs.avg:.3f})\t'
                  'gene_p {gene_probs.val:.3f} ({gene_probs.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, 
                   loss=losses, 
                   image_probs=image_prob_meter,
                   gene_probs=gene_prob_meter))
            # print(out_image.shape)
            sys.stdout.flush()
            
        # ===================debug======================
        if np.isnan(image_prob_meter.val):
            print(list(model.parameters()))
            print(feat_image)
            print(feat_gene)
            print(out_image)
            
            raise Exception("Nan detected")
            break

    return losses.avg, image_loss_meter.avg, image_prob_meter.avg, gene_loss_meter.avg, gene_prob_meter.avg


def test(epoch, test_loader, model, contrast, criterion_image, criterion_gene, optimizer, 
          gradient_clip=10, print_freq=1, device=torch.device("cuda:0")):
    """Testing"""
    model.eval()
    contrast.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    for idx, (data, index) in enumerate(test_loader):
        data_time.update(time.time() - end)

        batch_size = data['image'].size(0)
        index = index.to(device)
        for _ in data.keys():
            data[_] = data[_].float().to(device)

        # ===================forward=====================
        with torch.no_grad():
            feat_image, feat_gene = model(data)
        
        # Append
        if idx == 0:
            feat_images = feat_image.cpu().numpy()
            feat_genes = feat_gene.cpu().numpy()
        else:
            feat_images = np.concatenate([feat_images, feat_image.cpu().numpy()])
            feat_genes = np.concatenate([feat_genes, feat_gene.cpu().numpy()])
        
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()
        
    # Normalize & calculate correlation
    feat_images = feat_images / np.sum(feat_images ** 2, axis=1, keepdims=True) ** 0.5
    feat_genes = feat_genes / np.sum(feat_genes ** 2, axis=1, keepdims=True) ** 0.5
    corr = []
    for i in range(feat_images.shape[1]):
        corr.append(spearmanr(feat_images[:, i], feat_genes[:, i]).correlation)
    if epoch % 100 == 0:
        plt.hist(corr, bins=30)
        plt.show()
    print("Val epoch {}, average corr {}".format(epoch, np.average(corr)))

    return np.average(corr)