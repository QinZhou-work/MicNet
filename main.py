import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
import os
import tifffile
import skimage
import tifffile
import torch
import torch.hub
import torch.nn
from utils.MicNet_model import *
from torch.nn.modules import MSELoss
from utils.utils_eval import NCESoftmaxLoss, NCECriterion
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from scipy.optimize import linear_sum_assignment

def data_check_preprocessing(args):
    """
    Perform data checking and preprocessing for spatial transcriptomics data.

    Steps:
    1. Load image, count, and transformation files.
    2. Visualize spatial coordinates over the image.
    3. Remove genes with all-zero expression.
    4. Normalize count data using CPM (Counts Per Million).
    5. Compute summary statistics for each gene.
    6. Save processed data and statistics to CSV files.

    Args:
        args: An object containing file paths:
            - args.image_file: Path to the TIFF image file.
            - args.count_file: Path to the gene count matrix file.
            - args.transformation_file: Path to metadata or coordinate file (CSV or TSV).
    """

    # --- Load input data ---
    # Load count data (genes x spots)
    pd_count = pd.read_table(args.count_file, index_col=0)

    print("Original count size: {}".format(np.shape(pd_count)))

    # --- Remove genes with all-zero expression ---
    gene_sum = np.sum(pd_count, axis=0)
    gene_names = gene_sum.index.values[gene_sum > 0]  # Keep only genes with nonzero counts
    pd_count_norm = pd_count.loc[:, gene_names]

    # --- CPM (Counts Per Million) normalization ---
    # Each cell/spot’s gene expression is normalized by its total counts and scaled to 1e6
    for n_row in range(len(pd_count_norm)):
        pd_count_norm.iloc[n_row, :] = (
            pd_count_norm.iloc[n_row, :] / np.sum(pd_count_norm.iloc[n_row, :]) * 1_000_000
        )

    # --- Compute statistics for each gene ---
    count_stat = pd.DataFrame(columns=pd_count_norm.columns.values,
                              index=["mean", "std", "tile_95", "non_zero_p", "max"])
    count_stat.loc['mean', :] = np.average(pd_count_norm, axis=0)
    count_stat.loc['std', :] = np.std(pd_count_norm, axis=0)
    count_stat.loc['tile_95', :] = np.quantile(pd_count_norm, 0.95, axis=0)
    count_stat.loc['non_zero_p', :] = np.sum(np.array(pd_count_norm) > 0, axis=0) / pd_count_norm.shape[0]
    count_stat.loc['max', :] = np.max(pd_count_norm, axis=0)

    # --- Save output statistics ---
    if not os.path.exists("./output"):
        os.makedirs("./output")
    count_stat.to_csv("./output/1_count_stat_breast_cancer_FFPE.csv")

    # --- Log-transform and normalize counts ---
    count = pd_count_norm.copy()
    count = np.log2(count + 1)  # Log2 transformation for normalization

    # Normalize by dividing by the log2(max+1) of each gene
    count_max = np.log2(count_stat.loc['max', :].astype(float).values + 1)
    count_max[count_max < 1] = 1  # Prevent division by small values
    count = count / count_max

    # --- Save normalized counts ---
    count.to_csv("./output/1_pd_count_norm_breast_cancer_FFPE.csv")


def micnet_train(args):
    """
    Train the MicNet model for spatial transcriptomics analysis.

    Steps:
    1. Load preprocessed data (gene statistics and normalized counts).
    2. Load spatial image and metadata (cell coordinates).
    3. Split dataset into training and testing sets.
    4. Visualize data split on the image.
    5. Initialize datasets and data loaders.
    6. Configure model, optimizer, and contrastive learning components.
    7. Train model with contrastive learning and evaluate correlation.
    8. Save checkpoints and the best-performing model.

    Args:
        args: Argument object containing required parameters and paths:
            - image_file (str): Path to the input image.
            - transformation_file (str): Path to metadata file (CSV/TSV).
            - model_dir (str): Directory to save trained model checkpoints.
            - device (torch.device): Device for computation (e.g., 'cuda' or 'cpu').
    """

    # --- Load preprocessed data ---
    count_stat = pd.read_csv("./output/1_count_stat_breast_cancer_FFPE.csv", index_col=0)
    count = pd.read_csv("./output/1_pd_count_norm_breast_cancer_FFPE.csv", index_col=0)

    # Load input image and metadata (coordinate transformation)
    image = tifffile.imread(os.path.join(args.image_file))
    image_norm = image.copy()  # Keep a normalized copy for visualization
    pd_meta = pd.read_table(args.transformation_file, index_col=0, sep=",")

    # --- Split data into training and testing sets (80/20 split) ---
    np.random.seed(82321)  # For reproducibility
    indexes_all = pd_meta.index.values
    training_indexes = np.random.choice(indexes_all, int(len(indexes_all) * 0.8), replace=False)
    testing_indexes = [_ for _ in indexes_all if _ not in training_indexes]
    print("#Training: {}, #Testing: {}".format(len(training_indexes), len(testing_indexes)))

    # Sort indices to maintain consistent ordering
    training_indexes.sort()
    testing_indexes.sort()

    # --- Quick data loader check (visual sanity test) ---
    np.random.seed()  # Reset seed for random augmentation
    train_set = Dataset(training_indexes, image_norm, count, pd_meta, augmentation=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1)

    # Visualize a single training example with label
    for x, y in train_loader:
        plt.imshow(normalize(np.transpose(x['image'][0].numpy(), (1, 2, 0)), inverse=True))
        plt.show()
        print(y)
        break

    # --- Display augmentation example ---
    image = normalize(np.transpose(x['image'][0].numpy(), (1, 2, 0)), inverse=True)

    # --- Training configuration ---
    device = args.device
    image_shape = 256           # Input image patch size
    n_genes = count_stat.shape[1]  # Number of genes (output dimensions)
    n_data = len(training_indexes)
    n_out_features = 100        # Output feature embedding dimension
    imagenet = 'resnet101'      # Backbone model for feature extraction

    # --- Contrastive learning (NCE) parameters ---
    nce_k = 100    # Number of negative samples
    nce_t = 0.07   # Temperature for contrastive loss
    nce_m = 0.9    # Momentum coefficient for contrastive memory bank
    softmax = False  # Whether to use softmax NCE variant

    # --- Training hyperparameters ---
    lr = 0.01           # Learning rate
    momentum = 0.9
    weight_decay = 0.0001
    gradient_clip = 5   # Prevent exploding gradients

    # --- Create dataset loaders ---
    train_set = Dataset(training_indexes, image_norm, count, pd_meta, augmentation=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=20, shuffle=True, num_workers=1)

    test_set = Dataset(testing_indexes, image_norm, count, pd_meta, augmentation=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=20, shuffle=False, num_workers=1)

    # --- Initialize model and contrastive modules ---
    model = MicNet(image_shape=image_shape, n_genes=n_genes,
                   n_out_features=n_out_features, imagenet=imagenet,
                   genenet=[500, 100]).to(device)

    contrast = NCEAverage(n_out_features, n_data, nce_k, nce_t, nce_m, softmax, device=device).to(device)

    # Define contrastive loss functions for image and gene embeddings
    criterion_image = NCESoftmaxLoss().to(device) if softmax else NCECriterion(n_data).to(device)
    criterion_gene = NCESoftmaxLoss().to(device) if softmax else NCECriterion(n_data).to(device)

    # --- Optimizer configuration ---
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    # --- Prepare model directory for saving checkpoints ---
    model_dir = args.trained_save_path
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # --- Optional: Load from existing checkpoint ---
    model_file = None
    if model_file is not None:
        print(f"=> loading checkpoint '{model_file}'")
        checkpoint = torch.load(model_file, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        contrast.load_state_dict(checkpoint['contrast'])
        del checkpoint  # Free memory

    # --- Training loop ---
    hist = {
        'loss': [],
        'corr_val': [],
        'image_loss': [],
        'image_prob': [],
        'gene_loss': [],
        'gene_prob': []
    }

    hist_best = dict()
    best_corr = -float("inf")
    save_freq = 1  # Save every epoch

    for epoch in range(args.count_trained):
        # --- Train for one epoch ---
        loss, image_loss, image_prob, gene_loss, gene_prob = train(
            epoch, train_loader, model, contrast,
            criterion_image, criterion_gene, optimizer,
            gradient_clip=gradient_clip, print_freq=10, device=device
        )

        # Log metrics
        hist['loss'].append(loss)
        hist['image_loss'].append(image_loss)
        hist['image_prob'].append(image_prob)
        hist['gene_loss'].append(gene_loss)
        hist['gene_prob'].append(gene_prob)

        # --- Evaluate correlation on test set ---
        corr = test(
            epoch, test_loader, model, contrast,
            criterion_image, criterion_gene, optimizer,
            gradient_clip=gradient_clip, print_freq=10, device=device
        )
        hist['corr_val'].append(corr)

        # --- Save best model based on correlation metric ---
        if corr > best_corr:
            best_corr = corr
            hist_best = {
                'model': model.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'corr_val': corr
            }
            torch.save(hist_best, os.path.join(model_dir, "epoch_best_breast.pt"))
            print(f"New best corr_val: {best_corr:.4f} at epoch {epoch}, model saved.")

    # Regular save by frequency
    if args.is_save_trained == 1 and epoch % save_freq == 0 and epoch != 0:
        print("Saving model checkpoint...")
        state = { 'model': model.state_dict(),
                  'contrast': contrast.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'epoch': epoch }
        torch.save(state, os.path.join(model_dir, f"epoch_{epoch}.pt"))

    # --- Save full training history for post-analysis ---
    torch.save(hist, os.path.join(model_dir, "hist.pt"))


def micnet_inference(args):
    """
    Perform inference using a trained MicNet model to extract image and gene feature embeddings.

    Steps:
    1. Load preprocessed count data and metadata.
    2. Initialize model and contrastive modules with saved weights.
    3. Run inference over all spatial locations to extract image and gene embeddings.
    4. Compute correlation between the learned feature spaces.
    5. Save extracted features to disk for downstream analysis (e.g., visualization or clustering).

    Args:
        args: Argument object containing file paths and parameters:
            - image_file (str): Path to the input image file (.tif).
            - transformation_file (str): Path to metadata file (CSV/TSV).
            - trained_save_path (str): Directory containing saved model checkpoints.
            - device (str or torch.device): Device for computation ('cuda' or 'cpu').
    """

    # --- Load preprocessed count data and statistics ---
    count_stat = pd.read_csv("./output/1_count_stat_breast_cancer_FFPE.csv", index_col=0)
    count = pd.read_csv("./output/1_pd_count_norm_breast_cancer_FFPE.csv", index_col=0)

    # Load input image
    image = tifffile.imread(os.path.join(args.image_file))
    image_norm = image.copy()

    # Load metadata (containing spatial coordinates)
    # NOTE: Possible typo in "trainsformation_file" — should be "transformation_file"
    pd_meta = pd.read_table(args.transformation_file, index_col=0, sep=",")
    
    # Split training and testing
    np.random.seed(82321)
    indexes_all = pd_meta.index.values.copy()
    training_indexes = np.random.choice(indexes_all, int(len(indexes_all) * 0.8), replace=False)
    testing_indexes = [_ for _ in indexes_all if _ not in training_indexes]
    print("#Training: {}, #Testing: {}".format(len(training_indexes), len(testing_indexes)))
    training_indexes.sort()
    testing_indexes.sort()

    # --- Define model and inference parameters ---
    device = args.device  # Use GPU for inference if available
    image_shape = 256                # Input patch size
    n_genes = count_stat.shape[1]    # Number of gene expression features
    n_data = len(training_indexes)   # Total data points (should be defined earlier)
    n_out_features = 100             # Feature embedding dimension
    imagenet = 'resnet101'           # CNN backbone for feature extraction

    # --- Contrastive learning (NCE) parameters ---
    nce_k = 100     # Number of negative samples
    nce_t = 0.07    # Temperature parameter for contrastive loss
    nce_m = 0.9     # Momentum coefficient for memory update
    softmax = False # Use classic NCE instead of softmax variant

    # --- Optimizer / training hyperparameters (used for loading state) ---
    lr = 0.01
    momentum = 0.9
    weight_decay = 0.0001
    gradient_clip = 5

    # --- Build dataset and dataloaders ---
    # These use the same Dataset class as training to load spatial patches
    train_set = Dataset(training_indexes, image_norm, count, pd_meta, augmentation=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=20, shuffle=True, num_workers=1)

    test_set = Dataset(testing_indexes, image_norm, count, pd_meta, augmentation=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=20, shuffle=False, num_workers=1)

    # --- Initialize model and contrastive modules ---
    model = MicNet(image_shape=image_shape, n_genes=n_genes, n_out_features=n_out_features,
                   imagenet=imagenet, genenet=[500, 100]).to(device)
    contrast = NCEAverage(n_out_features, n_data, nce_k, nce_t, nce_m, softmax, device=device).to(device)

    # Define loss functions (used in training but required to restore checkpoint)
    criterion_image = NCESoftmaxLoss().to(device) if softmax else NCECriterion(n_data).to(device)
    criterion_gene = NCESoftmaxLoss().to(device) if softmax else NCECriterion(n_data).to(device)
    criterion_rec = MSELoss().to(device)

    # --- Optimizer (to restore state) ---
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    # --- Load the trained model checkpoint ---
    model_file = os.path.join(args.trained_save_path, "epoch_best_breast.pt")
    if model_file is not None:
        print(f"=> Loading checkpoint '{model_file}'")
        checkpoint = torch.load(model_file, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        contrast.load_state_dict(checkpoint['contrast'])
        del checkpoint  # Free memory

    # --- Prepare full test dataset for inference (all spatial positions) ---
    test_set = Dataset(indexes_all, image_norm, count, pd_meta, augmentation=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=20, shuffle=False, num_workers=1)

    # --- Switch model to evaluation mode ---
    model.eval()

    # --- Run inference ---
    with torch.no_grad():
        for idx, (data, index) in enumerate(test_loader):
            batch_size = data['image'].size(0)
            index = index.to(device)
            for _ in data.keys():
                data[_] = data[_].float().to(device)

            # Forward pass: extract image and gene features
            feat_image, feat_gene = model(data)

            # Accumulate batch outputs into full arrays
            if idx == 0:
                feat_images = feat_image.cpu().numpy()
                feat_genes = feat_gene.cpu().numpy()
            else:
                feat_images = np.concatenate([feat_images, feat_image.cpu().numpy()])
                feat_genes = np.concatenate([feat_genes, feat_gene.cpu().numpy()])

    # --- Normalize feature embeddings (L2 normalization) ---
    feat_images = feat_images / np.sum(feat_images ** 2, axis=1, keepdims=True) ** 0.5
    feat_genes = feat_genes / np.sum(feat_genes ** 2, axis=1, keepdims=True) ** 0.5

    # --- Compute Spearman correlation between image and gene features ---
    corr = []
    for i in range(feat_images.shape[1]):
        corr.append(spearmanr(feat_images[:, i], feat_genes[:, i]).correlation)

    print(f"save the final result to {os.path.join(args.final_result, 'features.pt')}")
    # --- Save extracted features ---
    if not os.path.exists(args.final_result):
        os.makedirs(args.final_result)
    torch.save([feat_images, feat_genes],
               os.path.join(args.final_result, "features.pt"))


def main(args):

    # step 1: check data for the spatial transcriptome dataset
    data_check_preprocessing(args)

    # step 2: train the input data
    micnet_train(args)

    # step 3: inference Micnet model
    micnet_inference(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from slides')

    parser.add_argument('--image_file', default='./example_data/Visium_FFPE_Human_Breast_Cancer_image.tif', type=str,
                        help="image file location'")
    parser.add_argument('--count_file', default='./example_data/Counts.txt', type=str,
                        help="count file location'")
    parser.add_argument('--transformation_file', default='./example_data/Spot_metadata.csv', type=str,
                        help="spot meta data location'")
    parser.add_argument('--trained_save_path', default='./output/trained_models', type=str,
                        help="the path to save the intermediate trained models'")
    parser.add_argument('--count_trained', default=10, type=int, help='the number of the trained')
    parser.add_argument('--is_save_trained', default=0, choices=[0, 1], type=int, help='whether or not to save the trained models. Default is 0 (no)')
    parser.add_argument('--final_result', default='./final_result', type=str,
                        help="the output result of feature extraction")
    parser.add_argument('--device', default='cuda:0', type=str, help='Only GPU card supported.')

    args = parser.parse_args()
    main(args)



