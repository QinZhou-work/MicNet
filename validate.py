import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
import tifffile
import skimage
import tifffile
import torch
import torch.hub
import torch.nn
from utils.MicNet_model import *
from torch.nn.modules import MSELoss
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from scipy.optimize import linear_sum_assignment

def run_clustering(method, embedding, index=None, label_name="label", fill_value=-1):
    """
    Run clustering with NaN-safe masking and return a DataFrame with cluster labels.

    Parameters
    ----------
    method : sklearn-like clustering object
        Must implement .fit() and provide .labels_ after fitting.
    embedding : np.ndarray or pd.DataFrame
        2D array of shape (n_samples, n_features). May contain NaNs.
    index : pd.Index or list-like, optional
        Index to assign to the returned DataFrame. If None, defaults to 0..n-1.
    label_name : str, optional
        Column name for cluster labels in the returned DataFrame.
    fill_value : int, optional
        Value to assign for rows with NaN embeddings (default=-1).

    Returns
    -------
    df : pd.DataFrame
        DataFrame with one column containing cluster labels as integers.
        Invalid rows get `fill_value`. Index matches `index` if provided.
    """
    # Mask for valid samples (no NaNs across features)
    mask_valid = ~np.any(np.isnan(embedding), axis=1)

    # Allocate integer array with fill_value
    Y = np.full(embedding.shape[0], fill_value, dtype=int)

    # Fit method only on valid rows
    Cluster = method.fit(embedding[mask_valid])

    # Assign cluster labels back
    Y[mask_valid] = Cluster.labels_.astype(int)

    # Return DataFrame with correct index
    if index is None:
        index = range(embedding.shape[0])
    return pd.DataFrame({label_name: Y}, index=index)


def relabel_to_gt(pred_series, gt_series):
    """
    Relabel predicted clusters to match ground truth (GT) labels
    using the Hungarian algorithm for optimal one-to-one mapping.

    This function finds the best correspondence between predicted cluster IDs
    and true class labels, maximizing the overlap between them.
    It is commonly used in unsupervised clustering evaluation to align
    predicted cluster indices with known ground truth labels.

    Args:
        pred_series (pd.Series): Series of predicted cluster labels for each sample.
        gt_series (pd.Series): Series of ground truth (true) labels for each sample.

    Returns:
        pd.Series: A new Series where predicted clusters are relabeled
                   to match ground truth classes based on optimal assignment.
    """

    # Combine predicted and ground truth labels into a single DataFrame.
    # Drop any rows with missing values to ensure alignment.
    df = pd.DataFrame({"pred": pred_series, "gt": gt_series}).dropna()

    # Extract the unique cluster IDs and class labels.
    pred_clusters = sorted(df["pred"].unique())
    gt_classes = sorted(df["gt"].unique())

    # Initialize a confusion matrix to record overlaps
    # between each predicted cluster and each ground truth class.
    conf_mat = np.zeros((len(pred_clusters), len(gt_classes)))

    # Compute the confusion matrix:
    # Each entry (i, j) counts how many samples in predicted cluster i
    # belong to ground truth class j.
    for i, pc in enumerate(pred_clusters):
        mask = df["pred"] == pc  # Boolean mask for samples in cluster 'pc'
        for j, gc in enumerate(gt_classes):
            conf_mat[i, j] = np.sum(df.loc[mask, "gt"] == gc)

    # Apply Hungarian (linear sum assignment) algorithm to find
    # the best one-to-one mapping that maximizes total agreement.
    # Since the algorithm minimizes cost, use negative counts to maximize.
    row_ind, col_ind = linear_sum_assignment(-conf_mat)

    # Build a mapping dictionary:
    # predicted cluster → corresponding ground truth label
    mapping = {pred_clusters[i]: gt_classes[j] for i, j in zip(row_ind, col_ind)}

    # Map the predicted clusters to new GT-aligned labels
    return pred_series.map(mapping)

def micnet_clustering(args):
    """
    Perform clustering on MicNet embeddings and compare results to ground-truth annotations.

    Steps:
    1. Load ground truth (GT) tissue annotations and learned features.
    2. Concatenate image and gene embeddings into a joint MicNet representation.
    3. Run K-means clustering to assign each spatial spot to a cluster.
    4. Align predicted clusters with GT labels using Hungarian matching.
    5. Compute clustering quality metrics (ARI, AMI).
    6. Visualize GT vs predicted cluster maps over histology images.

    Args:
        args: Argument object containing file paths and configuration, including:
            - meta_data_annotation (str): Path to metadata file with GT labels.
            - model_save_path (str): Directory containing `features.pt` file.
            - image_path (str): Path to the corresponding histology image (.tif).
            - transformation_file (str): Metadata file with spot coordinates (CSV/TSV).
    """

    # --- Load ground truth (GT) tissue annotations ---
    GT_raw = pd.read_csv(args.meta_data_annotation, index_col=0)
    GT_raw = GT_raw[['annotation']]  # Keep only annotation column

    # --- Load precomputed feature embeddings (from trained MicNet model) ---
    feat_images, feat_genes = torch.load(args.final_result + "/features.pt")
    image_feature = feat_images
    gene_feature = feat_genes

    # --- Concatenate image and gene features into a joint embedding ---
    MicNet_embedding_array = np.concatenate((image_feature, gene_feature), axis=1)

    # Convert the embedding to a DataFrame and align with GT sample indices
    MicNet_embedding = pd.DataFrame(MicNet_embedding_array, index=GT_raw.index)

    # --- Perform clustering ---
    n_clusters = 5  # Number of clusters (predefined for this dataset)
    method = KMeans(init='k-means++', n_clusters=n_clusters, random_state=1)

    # Run clustering on MicNet embeddings
    MicNet_df = run_clustering(method, MicNet_embedding.to_numpy(),
                               MicNet_embedding.index, "label")

    # --- Retrieve ground truth labels ---
    gt = GT_raw['annotation']

    # --- Dictionary of predictions for consistency with other methods ---
    pred_dfs = {"MicNet": MicNet_df}

    # --- Initialize results storage ---
    scores = {}

    # --- Evaluate clustering results ---
    for name, df in pred_dfs.items():
        # Extract predicted labels from the DataFrame
        if "label" in df.columns:
            Y = df['label'].to_numpy()
        elif "cluster" in df.columns:
            Y = df['cluster'].to_numpy()
        else:
            raise ValueError(f"{name} dataframe missing label column")

        # Masks: keep only samples with both predictions and GT
        mask_pred = pd.notna(Y)
        mask_gt = gt.notna().to_numpy()
        mask_common = mask_pred & mask_gt

        y_true = gt.to_numpy()[mask_common]
        y_pred = Y[mask_common]

        # Encode string labels into integer IDs for metric computation
        le_true = LabelEncoder()
        y_true_enc = le_true.fit_transform(y_true)

        le_pred = LabelEncoder()
        y_pred_enc = le_pred.fit_transform(y_pred)

        # Compute clustering evaluation metrics:
        # - ARI: Adjusted Rand Index (corrects for chance)
        # - AMI: Adjusted Mutual Information
        ari = adjusted_rand_score(y_true_enc, y_pred_enc)
        ami = adjusted_mutual_info_score(y_true_enc, y_pred_enc)

        # Store results
        scores[name] = {"ARI": ari, "AMI": ami, "n_samples": mask_common.sum()}

    # --- Print evaluation results ---
    for name, score in scores.items():
        print(f"{name}: ARI={score['ARI']:.4f}, AMI={score['AMI']:.4f} (n={score['n_samples']})")

    # --- Load the original histology image and metadata ---
    image = tifffile.imread(args.image_file)
    image_norm = image.copy()
    pd_meta = pd.read_table(args.transformation_file, index_col=0, sep=",")

    # --- Define visualization layout ---
    methods = ["GT_annotation", "MicNet"]  # What to visualize
    n_rows, n_cols = 1, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()

    coords_x = pd_meta["Y"].values
    coords_y = pd_meta["X"].values

    # --- Define a fixed color palette for GT tissue types ---
    gt_labels = ['necrosis', 'immune', 'invasive', 'fat', 'fibrous', 'undefined']
    gt_palette = {
        'necrosis': sns.color_palette("tab10")[0],
        'immune': sns.color_palette("tab10")[1],
        'invasive': sns.color_palette("tab10")[2],
        'fat': sns.color_palette("tab10")[3],
        'fibrous': sns.color_palette("tab10")[4],
        'undefined': (0.7, 0.7, 0.7)
    }

    # --- Plot GT and predicted clustering maps side-by-side ---
    for i, method in enumerate(methods):
        ax = axes[i]

        if method == "GT_annotation":
            # Show ground truth tissue types
            hue = GT_raw['annotation'].fillna("undefined").astype(str)
            title = "Ground Truth"
        else:
            # Show MicNet clustering results, relabeled using Hungarian matching
            df = pred_dfs[method]
            relabeled = relabel_to_gt(df["label"], GT_raw['annotation'])
            hue = relabeled.fillna("undefined").astype(str)
            title = method

        sns.scatterplot(
            x=coords_x, y=coords_y,
            s=12,
            hue=hue,
            palette=gt_palette,     # Fixed GT color scheme for visual consistency
            hue_order=gt_labels,
            edgecolor=None,
            linewidth=0,
            ax=ax,
            legend=False
        )

        # Overlay scatterplot on the histology image
        ax.imshow(image_norm)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(np.min(coords_x) - 200, np.max(coords_x) + 200)
        ax.set_ylim(np.max(coords_y) + 200, np.min(coords_y) - 200)  # Flip y-axis
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide any unused subplots
    for j in range(len(methods), n_rows * n_cols):
        axes[j].axis("off")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    print("The validation figure is saved to ./validation.png")
    plt.savefig("validation.png")
    plt.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from slides')

    parser.add_argument('--image_file', default='example_data/Visium_FFPE_Human_Breast_Cancer_image.tif', type=str,
                        help="image file location'")
    parser.add_argument('--transformation_file', default='example_data/Spot_metadata.csv', type=str,
                        help="meta data annotation file location'")
    parser.add_argument('--meta_data_annotation', default='example_data/meta_data_with_annotation.csv', type=str,
                        help="meta data with annotation file location'")
    parser.add_argument('--final_result', default='./final_result', type=str,
                        help="the output result of feature extraction")

    args = parser.parse_args()
    micnet_clustering(args)