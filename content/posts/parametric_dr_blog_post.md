---
title: "From One Algorithm to Five: Rebuilding Parametric t-SNE as Parametric DR"
date: 2026-03-11
draft: false
tags:
  - dimensionality-reduction
  - machine-learning
  - open-source
summary: "Rebuilding parametric_tsne from scratch as parametric_dr: a PyTorch library providing parametric neural network implementations of five dimensionality reduction methods under a single API."
---

A few years ago I published [parametric_tsne](https://github.com/jsilter/parametric_tsne), a small Python package that trained a neural network to learn t-SNE embeddings. It did one thing: minimize KL divergence between Gaussian affinities in high-dimensional space and Student-t similarities in the embedding. It was built on TensorFlow and Keras, it supported custom layer architectures and multi-scale perplexities, and it accumulated 162 stars on GitHub. People used it, which was great.

But the landscape of dimensionality reduction has changed significantly since then. UMAP, PaCMAP, TriMap, and CEBRA have each introduced different inductive biases for learning low-dimensional representations; from fuzzy simplicial sets to triplet constraints to contrastive temporal learning. Meanwhile, PyTorch has become the dominant framework for research code, rather than TensorFlow.

So I rebuilt it from scratch. The result is [parametric_dr](https://github.com/jsilter/parametric_dr): a PyTorch-based library that provides parametric (neural network) implementations of five dimensionality reduction methods under a single, consistent API.

## What "parametric" means here

Standard implementations of t-SNE, UMAP, and friends directly optimize the positions of points in the embedding space. This works well for a fixed dataset, but if new data arrives, you have to re-run the entire optimization from scratch.

Parametric versions instead train a neural network encoder that *learns the mapping* from high-dimensional input to low-dimensional output. Once trained, the encoder can transform new data in a single forward pass; no re-optimization needed. This is the key practical advantage: you get a reusable model.

## What changed from parametric_tsne to parametric_dr

The old package was three files:

```
parametric_tSNE/
    __init__.py
    core.py
    utils.py
```

The new one is substantially larger and more capable:

```
parametric_dr/
    __init__.py
    _base.py          # shared base class
    core.py           # Parametric_tSNE
    umap.py           # Parametric_UMAP (fully self-contained, no umap-learn dep)
    pacmap.py         # Parametric_PaCMAP
    trimap.py         # Parametric_TriMap
    cebra.py          # Parametric_CEBRA
    temporal.py       # TemporalMixin for time-series regularization
    loss.py           # PyTorch loss functions
    utils.py          # NumPy utilities (KNN, perplexity search, etc.)
    metrics.py        # trustworthiness, continuity, neighborhood preservation, Shepard correlation
    analysis.py       # sweep_dims(), intrinsic dimensionality estimation
```

Here is a summary of the major changes:

|                             | parametric_tsne       | parametric_dr                          |
| --------------------------- | --------------------- | -------------------------------------- |
| **Framework**               | TensorFlow / Keras    | PyTorch                                |
| **Algorithms**              | t-SNE only            | t-SNE, UMAP, PaCMAP, TriMap, CEBRA     |
| **API**                     | Custom                | sklearn-compatible (fit/transform)     |
| **Model persistence**       | Not built-in          | save_model() / restore_model()         |
| **Quality metrics**         | None                  | 4 metrics (T, C, N, S)                 |
| **Temporal regularization** | None                  | Composable TemporalMixin               |
| **Dimensionality analysis** | None                  | sweep_dims(), intrinsic dim estimation |
| **PCA preprocessing**       | Manual recommendation | Built-in (automatic, configurable)     |
| **Custom encoders**         | Keras layers          | Any nn.Module                          |

Some things stayed the same. The default encoder architecture is still the 500-500-2000 dense network from van der Maaten 2009. The linear output layer (rather than ReLU, which occasionally produced degenerate all-zero dimensions) carries over. You can still pass in a custom network, if desired. Multi-scale perplexity support, where you pass a list like `[10, 20, 30, 50, 100, 200]`, is still there. 

## The architecture

Every method inherits from a single base class (`ParametricDR`) that handles the boring-but-important stuff: encoder initialization, PCA preprocessing, the training loop skeleton, model serialization, and the sklearn estimator interface.

```
ParametricDR (sklearn BaseEstimator + TransformerMixin)
    |-- Parametric_tSNE     # KL divergence on Gaussian/Student-t
    |-- Parametric_UMAP     # fuzzy simplicial set + cross-entropy
    |-- Parametric_PaCMAP   # pairwise controlled manifold approximation
    |-- Parametric_TriMap   # triplet constraints
    |-- Parametric_CEBRA    # contrastive learning with temporal structure
```

Each subclass only needs to implement `fit()` with its method-specific loss function. Everything else (transform, save, restore) is inherited. Adding a new DR method means writing one class with one method.

The `TemporalMixin` is a separate composable piece. If you have time-series data and want smooth trajectories in the embedding, combine it via multiple inheritance:

```python
class Temporal_tSNE(TemporalMixin, Parametric_tSNE):
    pass
```

This adds a temporal smoothness penalty (mean squared distance between consecutive timepoints in the embedding) to whatever loss the base method computes. It works with any of the five methods.

## The metrics

The package includes five quality metrics, all computed in pure NumPy so they work with any DR method (not just parametric ones):

- **Trustworthiness**: Are the nearest neighbors in the embedding actually neighbors in the original space? Penalizes "false neighbors" that appear close in the embedding but were distant in the input. Higher is better.

- **Continuity**: Are the nearest neighbors in the original space still neighbors in the embedding? Penalizes "missing neighbors." Higher is better.

- **Neighborhood Preservation**: What fraction of k-nearest neighbors are shared between the two spaces? A simple, interpretable overlap measure.

- **Shepard Correlation**: Pearson correlation between all pairwise distances in the input and the embedding. Measures global structure preservation. Important for catching methods that distort large-scale geometry.

- **Trajectory Smoothness** (temporal data only): Mean squared step size between consecutive timepoints in the embedding, normalized by the mean squared pairwise distance across the embedding. This normalization makes the metric scale-invariant, so methods that produce larger or smaller embeddings can be compared fairly. Lower is smoother. Only meaningful for data with a natural temporal ordering.

The first four are computed together via `compute_metrics()`, which shares distance matrices for efficiency. Trajectory smoothness is called separately via `trajectory_smoothness()` since not all data has temporal structure.

## How the methods compare: no single winner

The interesting finding from benchmarking across multiple real datasets is that there is no single best method. Each algorithm's loss function encodes different assumptions about what matters, and those assumptions show up clearly in the metrics.

### Handwritten digits (64 features, 10 classes, 1797 samples)

3-layer encoder (256 hidden units), 50 epochs, no PCA. Metrics on training data, k=10.

| Method          | Trustworthiness | Continuity | Nbr Preservation | Shepard Corr | Fit Time (s) |
| --------------- | :-------------: | :--------: | :--------------: | :----------: | :----------: |
| t-SNE (perp=30) |      0.967      |   0.984    |      0.399       |  **0.600**   |     11.9     |
| PaCMAP          |      0.971      |   0.985    |      0.394       |    0.463     |     20.7     |
| UMAP            |    **0.980**    | **0.986**  |    **0.447**     |    0.493     |    166.4     |
| TriMap          |      0.971      |   0.983    |      0.399       |    0.554     |     43.3     |
| PCA (baseline)  |      0.831      |   0.948    |      0.134       |    0.599     |     0.02     |

UMAP leads on local structure (T, C, N). t-SNE leads on Shepard correlation. It is by far the slowest, but of course that's fitting time. All the neural network methods (everything besides PCA) will take the same time at inference. Notice that PCA, despite being a linear method, has a Shepard correlation (0.599) nearly matching t-SNE (0.600); it preserves global geometry well but fails at local structure. This is the tradeoff that nonlinear methods are navigating.

### Olivetti faces (4096 features, 40 subjects, 400 images)

3-layer encoder (128 hidden units), 50 epochs, PCA to 50 dims. Metrics on full dataset, k=5.

| Method          | Trustworthiness | Continuity | Nbr Preservation | Shepard Corr | Fit Time (s) |
| --------------- | :-------------: | :--------: | :--------------: | :----------: | :----------: |
| t-SNE (perp=10) |      0.926      |   0.947    |      0.531       |    0.786     |     2.8      |
| PaCMAP          |      0.921      |   0.960    |      0.472       |    0.653     |     2.7      |
| UMAP            |      0.946      |   0.961    |      0.581       |    0.623     |     27.1     |
| TriMap          |    **0.950**    |   0.955    |    **0.593**     |    0.579     |     21.4     |
| PCA (baseline)  |      0.856      |   0.924    |      0.252       |  **0.850**   |     0.1      |

The same pattern holds on a completely different data type: face images at much higher dimensionality. t-SNE's Shepard correlation (0.786) is 20% higher than the next-best nonlinear method (PaCMAP at 0.653). Meanwhile TriMap edges out UMAP on trustworthiness here (0.950 vs 0.946). And again, PCA's Shepard correlation (0.850) is the highest of all; it just cannot do the nonlinear folding needed for local structure.

### Lorenz attractor (20 dimensions, 1500 timepoints)

This is where temporal methods shine. The Lorenz system has smooth dynamics, so embeddings should produce smooth trajectories.

| Method             |   Trust   |   Cont    | Nbr Pres  |  Shepard  | Traj Smoothness | Time (s) |
| ------------------ | :-------: | :-------: | :-------: | :-------: | :-------------: | :------: |
| t-SNE              |   0.996   |   0.999   | **0.782** | **0.962** |     0.0058      |   24.9   |
| PaCMAP             |   0.996   |   0.999   |   0.734   |   0.943   |     0.0052      |   40.2   |
| UMAP               | **0.997** |   0.999   |   0.753   |   0.948   |     0.0048      |  263.2   |
| CEBRA              |   0.899   |   0.975   |   0.269   |   0.713   |     0.0087      |   98.4   |
| Temporal t-SNE     |   0.997   |   0.999   |   0.781   |   0.955   |     0.0051      |   26.8   |
| Temporal UMAP      |   0.996   |   0.999   |   0.778   |   0.928   |   **0.0040**    |  197.3   |

Trajectory smoothness here is normalized (divided by the mean pairwise distance in the embedding; see the metrics documentation), so the values are scale-invariant and comparable across methods. Lower is smoother.

Two things stand out. First, all methods do well on this smooth manifold; t-SNE achieves the best Shepard correlation (0.962) and neighborhood preservation (0.782), with its temporal variant close behind. The temporal variants improve trajectory smoothness over their static counterparts (e.g., Temporal UMAP at 0.0040 vs static UMAP at 0.0048), and Temporal UMAP achieves the smoothest trajectories overall.

Second, on this well-structured continuous manifold, PaCMAP and UMAP both achieve excellent Shepard correlations (0.94+), much closer to t-SNE than on the discrete cluster datasets. The difference between methods shrinks when the underlying data has smooth, continuous structure rather than separated clusters.

### Neural place cells (50 neurons, 1500 timepoints, circular track)

Simulated hippocampal place cells with a known 1D circular ground truth.

| Method | Trust | Cont | Nbr Pres | Shepard | Traj Smoothness | Time (s) |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|
| t-SNE | 0.828 | 0.892 | 0.059 | 0.437 | 0.129 | 19.7 |
| PaCMAP | 0.842 | 0.904 | 0.072 | 0.423 | 0.081 | 8.7 |
| UMAP | **0.853** | 0.901 | **0.094** | 0.401 | 0.082 | 138.0 |
| CEBRA | 0.681 | 0.842 | 0.031 | 0.399 | 0.143 | 7.4 |
| Temporal t-SNE | 0.842 | 0.899 | 0.059 | **0.485** | 0.075 | 17.1 |
| Temporal UMAP | 0.852 | 0.894 | 0.080 | 0.381 | **0.020** | 141.9 |

This is a harder problem; all methods have lower scores. But the relative pattern persists: UMAP best on local metrics, Temporal t-SNE best on Shepard. Temporal UMAP achieves the smoothest trajectory (0.020), and both temporal variants improve over their static counterparts on smoothness.

### What explains the pattern?

The consistent Shepard correlation advantage of t-SNE is not an accident; it follows directly from how each loss function works.

**t-SNE** minimizes KL divergence over a full pairwise probability distribution. Every pair of points contributes to the loss, including distant ones. The KL divergence asymmetrically penalizes cases where the model places points far apart when they should be close (high P, low Q), which means it fights to preserve meaningful distance relationships at all scales.

**UMAP, PaCMAP, and TriMap** are graph-based methods that explicitly construct neighbor sets (edges, pairs, or triplets) and only optimize over those local relationships. Long-range distances are effectively discarded during graph construction. This gives them an edge on local metrics (trustworthiness, continuity, neighborhood preservation) because that is exactly what their loss functions are designed to optimize. But it means they have no direct signal pushing them to preserve global geometry.

This is not a flaw in the graph-based methods; it is a design choice. For many visualization tasks, local structure (are nearby points actually similar?) matters more than global faithfulness (is cluster A the right distance from cluster B?). But if your downstream analysis depends on distance relationships across the full dataset (e.g., trajectory analysis, interpolation, or clustering in the embedding space), the Shepard correlation gap matters.

The Lorenz results hint at an interesting middle ground: when the underlying manifold is smooth and continuous, the graph-based methods close much of the Shepard gap. Their local optimization effectively chains together into global structure. It is on data with discrete clusters and voids that global information gets lost.

## Practical usage

The API is deliberately simple:

```python
from parametric_dr import Parametric_UMAP

model = Parametric_UMAP(num_inputs=64, num_outputs=2, n_neighbors=15)
model.fit(X_train, epochs=50)

# Transform new data without re-training
X_test_emb = model.transform(X_test)

# Save and reload
model.save_model("my_umap.pt")
model.restore_model("my_umap.pt")
```

Custom encoders are supported (pass any `nn.Module`), PCA preprocessing is on by default (automatically skipped when input dimensions are already low), and all hyperparameters have sensible defaults.

The `analysis` module includes `sweep_dims()`, which trains models at multiple output dimensionalities, evaluates metrics at each, and finds the elbow point. Combined with `estimate_intrinsic_dim()` (which wraps the TwoNN estimator from skdim), this gives you a principled way to choose how many dimensions your embedding should have. For visualizations we almost always choose 2 or 3 but it's nice to have options.

## What is next

The package currently includes six examples covering synthetic data, handwritten digits, face images, Lorenz attractors, simulated neural recordings, and single-cell RNA-seq. The test suite covers all methods and metrics thoroughly. All the low-hanging fruit has been picked.

It should be clear from the evaluation that the best method is highly dependent on the nature of the underlying data. I'm thinking more about problems which are real enough to be important but still have some ground-truth for evaluation.  

The code is at [github.com/jsilter/parametric_dr](https://github.com/jsilter/parametric_dr). MIT licensed. Feedback and contributions welcome.
