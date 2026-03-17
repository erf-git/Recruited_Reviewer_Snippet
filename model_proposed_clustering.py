'''
File: model_proposed_clustering.py
Author: Ethan Feldman (erf6575@rit.edu)

Applies clustering algorithms to model predictions from a trained Graph Neural Network (GNN) used to detect fake (recruited) Amazon reviewers.

Pipeline overview:
  1. Load reviewer-pair edges and per-reviewer features.
  2. Assign ground-truth labels (Genuine / Recruited / Gray-area).
  3. Stratified train / val / test split (gray-area reviewers are excluded from
     training but included in inference as the "unsure" set).
  4. Scale features, build edge tensors, and move data to GPU/CPU.
  5. Define and load the saved ChebConv GNN model.
  6. Run inference → probabilities + hidden-layer embeddings.
  7. Visualise high-confidence predictions with feature histograms.
  8. Reduce embeddings to 2-D with t-SNE at several perplexity values.
  9. Cluster the t-SNE projections with DBSCAN and export per-cluster summaries.
 10. Drill down into individual clusters with a second DBSCAN pass.
 11. Overlay model probabilities as a heatmap on every cluster.
'''

# Standard library 
import os
import sys
import json
import time
from datetime import datetime

# Third-party 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------------

def setup_seed(seed: int) -> None:
    """
    Fix random seeds for PyTorch (CPU + all GPUs) and NumPy so that results
    are reproducible across runs.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

setup_seed(42)


# ---------------------------------------------------------------------------
# PLOT STYLE
# ---------------------------------------------------------------------------

plt.style.use('ggplot')

# Colorblind-friendly palette used consistently across all figures.
custom_colors = [
    "#0072B2",  # Dark Blue
    "#E69F00",  # Orange
    "#56B4E9",  # Light Blue
    "#ED6050",  # Red
    "#8EBA42",  # Green
    "#C77CFF",  # Purple
    "#006600",  # Dark Green
    "#F479C5",  # Pink
    "#BC0D0D",  # Dark Red
    "#7C0786",  # Dark Purple
    "#EAD56A",  # Yellow
    "#57CA9A",  # Mint
]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

# Root directory for all data, model artefacts, and results.
PATH = '/home/erf6575/Desktop/fake_reviewer_publish/'


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

# --- Knowledge-graph edges ---------------------------------------------------
# Each row represents a pair of reviewers who have both reviewed at least one common product.  `common_products_count` is used as the edge weight.
df_pairs = pd.read_csv(PATH + 'data/reviewer_pairs.csv')

# --- Per-reviewer features ---------------------------------------------------
# Sorted and indexed by reviewer_id for consistent positional alignment with PyTorch tensors and boolean masks constructed later.
df_features = pd.read_csv(PATH + 'data/features.csv')
df_features.sort_values(by=['reviewer_id'], inplace=True)
df_features.set_index('reviewer_id', drop=False, inplace=True)


# ---------------------------------------------------------------------------
# LABEL ASSIGNMENT  (Y)
# ---------------------------------------------------------------------------
# Three-class labelling scheme:
#   0   – Genuine reviewer (no reviews on focal products at all)
#   1   – Recruited / fake reviewer (>1 reviews on focal products during the
#         campaign window, suggesting coordinated behaviour)
#   0.5 – Gray-area (ambiguous; used for inference but excluded from training)

df_features['fake_reviewer'] = 0.5  # Default: gray-area

# Reviewers with more than one campaign-window review on focal products are treated as recruited (fake).
df_features.loc[df_features.n_of_reviews_during_campaign > 1, 'fake_reviewer'] = 1

# Reviewers with zero reviews on focal products are treated as genuine.
df_features.loc[df_features.n_reviews_to_focals == 0, 'fake_reviewer'] = 0

print("Label distribution:\n", df_features['fake_reviewer'].value_counts())


# ---------------------------------------------------------------------------
# STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# ---------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

def split_stratified_into_train_val_test(
        df_input,
        stratify_colname,
        frac_train=0.7,
        frac_val=0.15,
        frac_test=0.15,
        random_state=42):
    '''
    Split a DataFrame into stratified train, validation, and test subsets.

    Stratification ensures that the relative class frequencies in `stratify_colname` are preserved in every split.  
    The split is performed in two consecutive calls to sklearn's train_test_split:
        1. full data  →  train  +  temp
        2. temp        →  val    +  test

    Parameters
    ----------
    df_input : pd.DataFrame
        The full input dataset to be split.
    stratify_colname : str
        Column used for stratification (typically the label column).
    frac_train : float
        Fraction of data allocated to training (default 0.70).
    frac_val : float
        Fraction of data allocated to validation (default 0.15).
    frac_test : float
        Fraction of data allocated to testing (default 0.15).
    random_state : int
        Seed passed to train_test_split for reproducibility.

    Returns
    -------
    df_train, df_test, df_val : pd.DataFrame
        The three data subsets.
    y_train, y_test, y_val : pd.DataFrame
        The corresponding label columns.

    Raises
    ------
    ValueError
        If the three fractions do not sum to 1.0, or if `stratify_colname`
        is absent from `df_input`.
    '''
    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError(
            'fractions %f, %f, %f do not add up to 1.0'
            % (frac_train, frac_val, frac_test)
        )
    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % stratify_colname)

    X = df_input
    y = df_input[[stratify_colname]]

    # First split: isolate the training set.
    df_train, df_temp, y_train, y_temp = train_test_split(
        X, y,
        stratify=y,
        test_size=(1.0 - frac_train),
        random_state=random_state
    )

    # Second split: divide the remaining data into val and test.
    # `relative_frac_test` re-expresses frac_test as a fraction of the temp set.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(
        df_temp, y_temp,
        stratify=y_temp,
        test_size=relative_frac_test,
        random_state=random_state
    )

    # Sanity check: no samples should be lost or duplicated.
    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_test, df_val, y_train, y_test, y_val


# Split only the labeled (non-gray) examples; gray-area reviewers are predicted later as the "unsure" set.
labeled_df = df_features.loc[df_features.fake_reviewer != 0.5]
train_set, test_set, val_set, y_train, y_test, y_val = \
    split_stratified_into_train_val_test(labeled_df, 'fake_reviewer')


# ---------------------------------------------------------------------------
# BOOLEAN MASKS
# ---------------------------------------------------------------------------
# Masks index into the full df_features tensor (one entry per reviewer).
# Positions belonging to train / val / test are True; the rest are False.

train_mask = torch.zeros(len(df_features), dtype=torch.bool)
train_mask[train_set.index.values] = True

test_mask = torch.zeros(len(df_features), dtype=torch.bool)
test_mask[test_set.index.values] = True

val_mask = torch.zeros(len(df_features), dtype=torch.bool)
val_mask[val_set.index.values] = True

# `test_unsure_mask` selects all reviewers who are *not* in train or val, i.e. the held-out test set AND the gray-area reviewers.  
# This is the population on which we run the final clustering analysis.
test_unsure_mask = torch.logical_not(train_mask + val_mask).cpu().numpy()


# ---------------------------------------------------------------------------
# DEVICE
# ---------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ---------------------------------------------------------------------------
# FEATURE MATRIX  (X)
# ---------------------------------------------------------------------------

# Drop columns that are either identifiers, raw count proxies for the label, or graph-structural features not used by this model variant.
cols_to_drop = [
    'reviewer_id',
    'n_reviews_to_focals',           # directly encodes label
    'n_reviews_removed_by_Amazon',   # directly encodes label
    'n_of_reviews_during_campaign',  # directly encodes label
    'review_count',                  # raw count — label-leaking proxy
    'fake_reviewer',                 # the label itself
    'w_degree',                      # graph structural feature (excluded here)
    'eigenvector_centrality',        # graph structural feature (excluded here)
    'clustering_coefficient',        # graph structural feature (excluded here)
]
x_df = df_features.drop(cols_to_drop, axis=1)

# Fit the StandardScaler on training data only to prevent data leakage.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_df[train_mask.numpy()])

# Scaled tensor fed to the model.
x_tensor = torch.tensor(scaler.transform(x_df), dtype=torch.float32).to(device)

# Unscaled tensor kept for reference / interpretability plots.
x_tensor_raw = torch.tensor(x_df.to_numpy(), dtype=torch.float32).to(device)


# ---------------------------------------------------------------------------
# LABEL TENSOR  (Y)
# ---------------------------------------------------------------------------

y_df = df_features['fake_reviewer']
y_tensor = torch.tensor(y_df.values, dtype=torch.float32).to(device)


# ---------------------------------------------------------------------------
# MOVE MASKS TO DEVICE
# ---------------------------------------------------------------------------

train_mask = train_mask.to(device)
test_mask  = test_mask.to(device)
val_mask   = val_mask.to(device)


# ---------------------------------------------------------------------------
# EDGE TENSORS
# ---------------------------------------------------------------------------

# edge_index shape: (2, num_edges) — standard PyG convention.
edge_index = torch.tensor(
    df_pairs[['reviewer_id_x', 'reviewer_id_y']].values.T,
    dtype=torch.long
).to(device)

# Edge weight = number of products both reviewers reviewed in common.
edge_weight = torch.tensor(
    df_pairs['common_products_count'].values,
    dtype=torch.float32
).to(device)


# ---------------------------------------------------------------------------
# FEATURE COLUMN LISTS
# ---------------------------------------------------------------------------

# All columns used as model input.
feature_columns = x_df.columns.tolist()

# Subset of behavioural / metadata features used for cluster characterisation and distribution plots.  
# These correspond to reviewer writing and timing patterns rather than graph-structural signals.
metadata_columns = [
    'avg_review_rating',
    'std_review_rating',
    'avg_review_length',
    'std_review_length',
    'avg_time_between_reviews',
    'std_time_between_reviews',
    'min_time_between_reviews',
    'max_time_between_reviews',
    'share_photos',
    'share_helpful',
    'share_5star',
    'share_1star',
    'share_of_nonextreme_ratings',
    'cosine_sim',
]


# ---------------------------------------------------------------------------
# GNN MODEL DEFINITION
# ---------------------------------------------------------------------------

from torch_geometric.nn import ChebConv

layer = "ChebConv"  # Used as a string key when loading saved parameters/weights.

class MultimodalModel(nn.Module):
    '''
    Single-layer Chebyshev Graph Convolutional Network (ChebConv) with a
    linear classifier head for binary fake-reviewer detection.

    Architecture
    ------------
    Input features  (input_dim)
        └─ ChebConv(K=k_1)  →  ReLU  →  Dropout(p)   [hidden_dim_1 units]
            └─ Linear(1)  →  Sigmoid                   [scalar probability]

    The forward pass returns three outputs so that the intermediate hidden
    representation (x) can be inspected for t-SNE / clustering analysis.

    Parameters
    ----------
    input_dim : int
        Number of input features per node.
    p : float
        Dropout probability applied after the graph convolution.
    hidden_dim_1 : int
        Output dimensionality of the ChebConv layer.
    k_1 : int
        Chebyshev filter order (polynomial degree).
    '''

    def __init__(self, input_dim: int, p: float, hidden_dim_1: int, k_1: int):
        super(MultimodalModel, self).__init__()

        self.p = p
        # Graph convolutional layer: aggregates neighbour information using
        # Chebyshev polynomials up to degree k_1.
        self.conv1 = ChebConv(input_dim, hidden_dim_1, K=k_1)
        # Final linear layer maps the hidden representation to a single logit.
        self.classifier = nn.Linear(hidden_dim_1, 1)

    def forward(self, x, e_index, e_weights):
        '''
        Forward pass through the GNN.

        Parameters
        ----------
        x : torch.Tensor, shape (N, input_dim)
            Node feature matrix (standardised).
        e_index : torch.Tensor, shape (2, E)
            Edge list in COO format.
        e_weights : torch.Tensor, shape (E,)
            Scalar edge weights (common product counts).

        Returns
        -------
        out_probs : torch.Tensor, shape (N,)
            Sigmoid-activated probability that each reviewer is recruited.
        x : torch.Tensor, shape (N, hidden_dim_1)
            Post-activation hidden-layer embedding used for t-SNE analysis.
        out : torch.Tensor, shape (N, 1)
            Raw (pre-sigmoid) logits.
        '''
        # Graph convolution + ReLU activation + dropout for regularisation.
        x = F.dropout(
            F.relu(self.conv1(x, e_index, e_weights)),
            training=self.training,
            p=self.p
        )
        out = self.classifier(x)

        # Sigmoid converts the logit to a probability in [0, 1].
        out_probs = torch.sigmoid(out)

        return out_probs.squeeze(), x, out


# ---------------------------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------------------------

# Timestamp string that identifies which saved model / result files to use.
# Update this to point to a different training run if needed.
NOW = "2024-Nov-19_12-29-41"

# Load the hyperparameters recorded during training.
with open(f'{PATH}model_json/proposed_{layer}_params.json') as json_data:
    params = json.load(json_data)

# Instantiate the model with the saved hyperparameters.
model = MultimodalModel(
    input_dim=x_df.shape[1],
    hidden_dim_1=params['hidden_dim_1'],
    k_1=params['k_1'],
    p=params['p']
).to(device)

# Load the trained weights; weights_only=True avoids arbitrary code execution.
model.load_state_dict(
    torch.load(PATH + f"results/{NOW}_model", weights_only=True)
)
model.eval()  # Disable dropout for deterministic inference.


# ---------------------------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------------------------

# Run the full dataset through the trained model (no gradient needed).
with torch.no_grad():
    PROBS, X_RAW, OUT_RAW = model(x_tensor, edge_index, edge_weight)

# Hard predictions: threshold probability at 0.5.
PREDS = torch.where(PROBS < 0.5, 0.0, 1.0)

# Restrict to the test + unsure population for all downstream analysis.
PROBS  = PROBS[test_unsure_mask].cpu().detach().numpy()  # shape: (M,)
X_RAW  = X_RAW[test_unsure_mask].cpu().detach().numpy()  # shape: (M, hidden_dim_1)
PREDS  = PREDS[test_unsure_mask].cpu().detach().numpy()  # shape: (M,)


# ---------------------------------------------------------------------------
# FEATURE DISTRIBUTION PLOTS FOR HIGH-CONFIDENCE PREDICTIONS
# ---------------------------------------------------------------------------

# Plot the distribution of each metadata feature for reviewers where the model is highly confident (prob ≤ 0.25 → likely genuine; prob ≥ 0.75 → likely fake).

# High-confidence GENUINE predictions (prob ≤ 0.25) 
os.makedirs(PATH + "results/pred_genuine_plots", exist_ok=True)
for column in metadata_columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(x_df[test_unsure_mask][PROBS <= 0.25][column], kde=True)
    plt.title(f'Model Prediction GENUINE Distribution of {column}')
    plt.savefig(PATH + f"results/pred_genuine_plots/pred_genuine_{column}_plot.png")
    plt.close()

# High-confidence RECRUITED predictions (prob ≥ 0.75) 
os.makedirs(PATH + "results/pred_fake_plots", exist_ok=True)
for column in metadata_columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(x_df[test_unsure_mask][PROBS >= 0.75][column], kde=True)
    plt.title(f'Model Prediction RECRUITED Distribution of {column}')
    plt.savefig(PATH + f"results/pred_fake_plots/pred_fake_{column}_plot.png")
    plt.close()

# Export descriptive statistics for the two high-confidence cohorts.
x_df[test_unsure_mask][PROBS <= 0.25][metadata_columns].describe().T \
    .to_csv(PATH + f"results/genuine_summary_{NOW}.csv")

x_df[test_unsure_mask][PROBS >= 0.75][metadata_columns].describe().T \
    .to_csv(PATH + f"results/fake_summary_{NOW}.csv")


# ---------------------------------------------------------------------------
# T-SNE DIMENSIONALITY REDUCTION
# ---------------------------------------------------------------------------

# Project the GNN's hidden-layer embeddings (X_RAW) into 2-D using t-SNE.
# We experiment with several perplexity values; higher perplexity captures more global structure at the cost of local neighbourhood fidelity.

from sklearn.manifold import TSNE

def get_tsne(r: int):
    '''
    Fit a 2-D t-SNE on the GNN hidden-layer embeddings and produce three plots:
      1. Original ground-truth labels coloured by Genuine / Recruited / Unsure.
      2. Model predictions coloured by Likely Genuine / Likely Recruited / Unsure.
      3. Continuous probability heatmap (coolwarm colormap).

    Results (plot images + CSV of 2-D coordinates) are saved under
    PATH/results/t-SNE_p{r}_group_plots/.

    Parameters
    ----------
    r : int
        t-SNE perplexity value.  Typical range: 5–500.

    Returns
    -------
    tsne_x_raw : np.ndarray, shape (M, 2)
        Raw 2-D t-SNE coordinates.
    df_tsne : pd.DataFrame
        DataFrame with columns x, y, label, pred, prob.
    '''
    print(f"Running t-SNE with perplexity={r} …")
    os.makedirs(PATH + f"results/t-SNE_p{r}_group_plots", exist_ok=True)

    # Fit t-SNE on the hidden-layer embeddings.
    tsne = TSNE(n_components=2, perplexity=r, n_iter=1000,
                random_state=42, n_jobs=-1)
    tsne_x_raw = tsne.fit_transform(X_RAW)

    df_tsne = pd.DataFrame(tsne_x_raw, columns=['x', 'y'])

    # Plot 1: ground-truth labels 
    df_tsne['label'] = y_df[test_unsure_mask].values
    fig, ax = plt.subplots(figsize=(10, 6))
    label_style = {
        0.0: ('#71b8dfff', 'Genuine'),
        1.0: ('#e06666',   'Recruited'),
        0.5: ('#808080',   'Unsure'),
    }
    for u in df_tsne['label'].unique():
        color, text = label_style[u]
        ax.scatter(
            x=df_tsne.loc[df_tsne['label'] == u, 'x'],
            y=df_tsne.loc[df_tsne['label'] == u, 'y'],
            c=color, label=text, s=10, alpha=0.8
        )
    ax.legend(title='Original Labels', loc='upper right',
              title_fontsize=12, markerscale=3)
    plt.title(f't-SNE(p={r}) of GNN Output after Layer 1')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(PATH + f"results/t-SNE_p{r}_group_plots/"
                f"t-SNE_p{r}_GNN_originial_{NOW}.png")
    plt.show()

    # Plot 2: model predictions 
    df_tsne['pred'] = PREDS
    fig, ax = plt.subplots(figsize=(10, 6))
    pred_style = {
        0.0: ('#71b8dfff', 'Likely Genuine'),
        1.0: ('#e06666',   'Likely Recruited'),
        0.5: ('#808080',   'Unsure'),
    }
    for u in df_tsne['pred'].unique():
        color, text = pred_style[u]
        ax.scatter(
            x=df_tsne.loc[df_tsne['pred'] == u, 'x'],
            y=df_tsne.loc[df_tsne['pred'] == u, 'y'],
            c=color, label=text, s=10, alpha=0.8
        )
    ax.legend(title='Model Predictions', loc='upper right',
              title_fontsize=12, markerscale=3)
    plt.title(f't-SNE(p={r}) of GNN Output after Layer 1')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(PATH + f"results/t-SNE_p{r}_group_plots/"
                f"t-SNE_p{r}_GNN_predictions_{NOW}.png")
    plt.show()

    # Plot 3: continuous probability heatmap 
    df_tsne['prob'] = PROBS
    fig, ax = plt.subplots(figsize=(11, 6))
    scatter = ax.scatter(
        x=df_tsne['x'], y=df_tsne['y'],
        c=df_tsne['prob'], cmap='coolwarm', s=10, alpha=0.8
    )
    plt.colorbar(scatter, label='Probability Reviewer is Recruited')
    plt.title(f't-SNE(p={r}) of GNN Output after Layer 1')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(PATH + f"results/t-SNE_p{r}_group_plots/"
                f"t-SNE_p{r}_GNN_probabilities_{NOW}.png")
    plt.show()

    # Save the 2-D coordinates together with labels, predictions, and probs.
    df_tsne.to_csv(
        PATH + f"results/t-SNE_p{r}_group_plots/t-SNE_p{r}_{NOW}.csv"
    )

    return tsne_x_raw, df_tsne


# Run t-SNE at four perplexity values to assess structural stability.
# (Higher values commented out to save time; uncomment as needed.)
tsne_50,  df_tsne_50  = get_tsne(50)
tsne_100, df_tsne_100 = get_tsne(100)
# tsne_150 = get_tsne(150)
# tsne_200 = get_tsne(200)
tsne_250 = get_tsne(250)
# tsne_300 = get_tsne(300)
tsne_500 = get_tsne(500)
# tsne_750 = get_tsne(750)
# tsne_1000 = get_tsne(1000)


# Reload a previously saved t-SNE result (e.g. for downstream clustering without re-running the expensive embedding step).
r = 100
df_tsne_100 = pd.read_csv(
    PATH + f"results/t-SNE_p{r}_group_plots/t-SNE_p100_2024-Nov-19_12-29-41.csv"
)


# ---------------------------------------------------------------------------
# DBSCAN — EPSILON SELECTION VIA K-DISTANCE GRAPH
# ---------------------------------------------------------------------------

from sklearn.neighbors import NearestNeighbors

def plot_k_distance_graph(X: np.ndarray, k: int) -> None:
    '''
    Plot the sorted k-th nearest-neighbour distance for each point in X.

    This is a standard diagnostic for choosing the DBSCAN epsilon parameter:
    the optimal epsilon corresponds to the first pronounced "elbow" (inflection
    point pointing upward) in the resulting curve.

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Input data points.
    k : int
        Neighbourhood size (should match the min_samples used in DBSCAN).
    '''
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)

    # Sort k-th neighbour distances in ascending order for the elbow plot.
    distances = np.sort(distances[:, k - 1])

    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel('Points (sorted by distance)')
    plt.ylabel(f'{k}-th nearest neighbour distance')
    plt.title('K-distance Graph  —  choose ε at the first upward elbow')
    plt.show()


# ---------------------------------------------------------------------------
# DBSCAN — GLOBAL CLUSTERING
# ---------------------------------------------------------------------------

from sklearn.cluster import DBSCAN

def get_DBSCAN(tsne_: np.ndarray, r: int, n_neighbors: int, e_size: float):
    '''
    Fit DBSCAN on 2-D t-SNE coordinates and plot the resulting clusters.

    Points assigned label -1 by DBSCAN are noise (no cluster membership).
    All other labels correspond to dense clusters in the embedding space.

    Parameters
    ----------
    tsne_ : np.ndarray, shape (N, 2)
        2-D t-SNE coordinates to cluster.
    r : int
        Perplexity used when generating tsne_ (for plot titles / filenames).
    n_neighbors : int
        DBSCAN `min_samples` — minimum points to form a core point.
    e_size : float
        DBSCAN `eps` — neighbourhood radius.

    Returns
    -------
    d_labels : np.ndarray, shape (N,)
        Integer cluster label per point (-1 = noise).
    '''
    dbscan = DBSCAN(eps=e_size, min_samples=n_neighbors)
    dbscan.fit(tsne_)
    d_labels = dbscan.labels_

    plt.figure(figsize=(11, 6))
    unique_labels = np.unique(d_labels)
    for label in unique_labels:
        if label == -1:
            # Noise points are rendered in grey and labelled separately.
            plt.scatter(
                tsne_[d_labels == label, 0], tsne_[d_labels == label, 1],
                color='gray', label='Noise', s=10, alpha=0.8
            )
        else:
            plt.scatter(
                tsne_[d_labels == label, 0], tsne_[d_labels == label, 1],
                label=f'Cluster {label}', s=10, alpha=0.8
            )
    plt.title(f't-SNE Clusters  DBSCAN(eps={e_size}, min_samples={n_neighbors})')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(markerscale=3, bbox_to_anchor=(1.04, 0.5),
               loc="center left", borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()

    return d_labels


# Select epsilon from the k-distance graph elbow, then run global DBSCAN.
r = 100
n_neighbors = 150
plot_k_distance_graph(df_tsne_100[['x', 'y']].values, k=n_neighbors)

e_size  = 2.98  # Chosen from the first upward elbow in the k-distance plot.
d_labels = get_DBSCAN(df_tsne_100[['x', 'y']].values, r, n_neighbors, e_size)


# Export a descriptive-statistics summary for each cluster so that the behavioural profile of each group can be inspected.
unique_labels = np.unique(d_labels)
for group in unique_labels:
    x_df[test_unsure_mask][d_labels == group][metadata_columns] \
        .describe().T \
        .to_csv(PATH + f"results/t-SNE_p{r}_group_plots/"
                       f"cluster_{group}_summary_{NOW}.csv")


# ---------------------------------------------------------------------------
# DBSCAN — SUB-CLUSTER ANALYSIS (drill-down into individual clusters)
# ---------------------------------------------------------------------------

def get_group_DBSCAN(
        tsne_: np.ndarray,
        r: int,
        n_neighbors: int,
        e_size: float,
        group: int,
        data: pd.DataFrame):
    '''
    Run a second DBSCAN pass on the points belonging to a single top-level
    cluster to discover finer sub-structure within it.

    Sub-clusters are labelled with letters (A, B, C, …) to distinguish them
    from the numeric labels of the global clustering.  A summary CSV is saved
    for each sub-cluster.

    Parameters
    ----------
    tsne_ : np.ndarray, shape (M, 2)
        2-D t-SNE coordinates for the *subset* of points in this cluster.
    r : int
        Perplexity of the parent t-SNE (for plot titles / filenames).
    n_neighbors : int
        DBSCAN min_samples for this local pass.
    e_size : float
        DBSCAN eps for this local pass.
    group : int
        Top-level cluster index (used in titles and filenames).
    data : pd.DataFrame
        Feature rows corresponding to the points in tsne_ (same row order).

    Returns
    -------
    c_labels : np.ndarray, shape (M,)
        Integer sub-cluster label per point (-1 = noise).
    '''
    dbscan = DBSCAN(eps=e_size, min_samples=n_neighbors)
    dbscan.fit(tsne_)
    c_labels = dbscan.labels_

    plt.figure(figsize=(7, 4))
    unique_labels = np.unique(c_labels)
    letters = ["A", "B", "C", "D", "E", "F"]  # Up to 6 sub-clusters supported.

    for label in unique_labels:
        if label == -1:
            plt.scatter(
                tsne_[c_labels == label, 0], tsne_[c_labels == label, 1],
                color='gray', label='Noise', s=25, alpha=0.8
            )
        else:
            plt.scatter(
                tsne_[c_labels == label, 0], tsne_[c_labels == label, 1],
                label=f'Cluster {letters[label]}', s=25, alpha=0.8
            )
    plt.title(
        f't-SNE Cluster {group}  '
        f'DBSCAN(eps={e_size}, min_samples={n_neighbors})'
    )
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(markerscale=3, bbox_to_anchor=(1.04, 0.5),
               loc="center left", borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(
        PATH + f"results/t-SNE_p{r}_group_plots/"
               f"t-SNE_p{r}_Cluster_{group}_DBSCAN_{NOW}.png"
    )
    plt.show()

    # Save a feature summary for each sub-cluster.
    for label in unique_labels:
        data[c_labels == label][metadata_columns].describe().T \
            .to_csv(PATH + f"results/t-SNE_p{r}_group_plots/"
                           f"cluster_{group}{letters[label]}_summary_{NOW}.csv")

    return c_labels


# Drill-down into clusters 0, 2, and 5 
# Parameters (n, e) were tuned per-cluster using the k-distance graph.

group = 0
n, e = 75, 2.032
group_0_labels = get_group_DBSCAN(
    df_tsne_100[d_labels == group][['x', 'y']].values,
    r, n, e, group,
    x_df[test_unsure_mask][d_labels == group]
)

group = 2
n, e = 60, 2.37
group_1_labels = get_group_DBSCAN(
    tsne_50[d_labels == group],
    r, n, e, group,
    x_df[test_unsure_mask][d_labels == group]
)

group = 5
n, e = 50, 2.2
group_3_labels = get_group_DBSCAN(
    tsne_50[d_labels == group],
    r, n, e, group,
    x_df[test_unsure_mask][d_labels == group]
)


# ---------------------------------------------------------------------------
# PER-CLUSTER PROBABILITY HEATMAPS
# ---------------------------------------------------------------------------

def get_group_heatmap(tsne_: pd.DataFrame, group: int) -> None:
    '''
    Scatter plot of a single cluster's t-SNE embedding coloured by the model's
    predicted recruitment probability (coolwarm: blue = genuine, red = fake).

    Helps visually confirm whether DBSCAN sub-clusters align with the model's
    confidence gradient within a given top-level cluster.

    Parameters
    ----------
    tsne_ : pd.DataFrame
        Subset of the t-SNE DataFrame for this cluster.
        Must contain columns 'x', 'y', and 'prob'.
    group : int
        Top-level cluster index (used in the figure title and filename).
    '''
    fig, ax = plt.subplots(figsize=(7, 4))
    scatter = ax.scatter(
        x=tsne_['x'], y=tsne_['y'],
        c=tsne_['prob'], cmap='coolwarm', s=25, alpha=0.8
    )
    plt.colorbar(scatter, label='Probability Reviewer is Recruited')
    plt.title(f't-SNE(p={r}) of GNN Output after Layer 1  —  Cluster {group}')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig(
        PATH + f"results/t-SNE_p{r}_group_plots/"
               f"t-SNE_p{r}_Cluster_{group}_Heatmap_{NOW}.png"
    )
    plt.show()


# Generate a heatmap for every top-level DBSCAN cluster.
for group in np.unique(d_labels):
    get_group_heatmap(df_tsne_100[d_labels == group], group)
