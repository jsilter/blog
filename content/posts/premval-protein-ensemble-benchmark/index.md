---
title: Benchmarking Protein Ensemble Generators
date: 2026-05-31
draft: true
tags:
  - structural-biology
  - machine-learning
  - protein-structure
  - benchmarks
summary: Generative models for protein conformational ensembles are arriving monthly, but the field has no neutral cross-model comparison. A tour of why ensemble evaluation is hard and where the methods landscape sits today.
---
## Introducing Premval

Premval (PRotein ENseMble eVALuation) is a neutral, openly published leaderboard for generative models that predict protein conformational ensembles. It scores every model against the same reference data (the ATLAS molecular dynamics set), through the same fixed panel of metrics, and labels which models were trained on that data, so you can tell genuine generalization from a model grading its own homework. The aim is to make a stream of monthly new methods comparable, and to answer a question the field currently can't: given a protein, which generator should you actually use?

## Beyond a Single Fold

Not too long ago, AlphaFold changed everything. And then AlphaFold2 changed everything again. And then AlphaFold3 changed everything again. Now with just a protein sequence, a database of MSAs, and a pile of GPUs we can cure all disease. Except not really.

AlphaFold3 is excellent at solving fixed structures for well-behaved protein monomers which have many homologs so we can build a deep multiple-sequence alignment (MSA). This still leaves some gaps.

1. Proteins without many relatives
2. Protein complexes (multimers, protein-protein interactions).
3. Proteins with some flexibility in the structure

The first two are an issue with training data. Presumably if we had enough data from "orphan" proteins and from protein complexes we could train models (both single and ensemble) to predict such things.

Ensemble generation explicitly addresses point #3; flexibility and motion. Instead of limiting ourselves to a single structure we identify the distribution. That is, generate hundreds or thousands of different structures which (hopefully) cover the distribution. Note that these methods do *not* predict explicit dynamics, as one gets from a molecular dynamics simulation. Rather they predict the set of structures we expect a protein to inhabit. 

![Meme contrasting generating a single most-likely structure against generating from the full distribution](single-vs-distribution.png)

## A Sample of Ensemble Generators

These have received less press (and certainly no Nobel prizes...yet) but I think this family will be much more important in the long run. These models attempt a much harder task: instead of just sampling some elements, we wish to cover the entire distribution. 

| Method                          | Approach                                     | Reference                                                                          |
| ------------------------------- | -------------------------------------------- | ---------------------------------------------------------------------------------- |
| AlphaFlow / ESMFlow             | Flow matching on AlphaFold2 / ESMFold        | [Jing et al., ICML 2024](https://arxiv.org/abs/2402.04845)                         |
| ConfDiff                        | Diffusion with force guidance                | [Wang et al., ICML 2024](https://arxiv.org/abs/2403.14088)                         |
| Str2Str                         | Zero-shot diffusion                          | [Lu et al., ICLR 2024](https://arxiv.org/abs/2306.03117)                           |
| ESMDiff                         | Masked diffusion over structural tokens      | [Lu et al., ICLR 2025](https://arxiv.org/abs/2410.18403)                           |
| Distributional Graphormer (DiG) | Diffusion (Graphormer backbone)              | [Zheng et al., Nature MI 2024](https://www.nature.com/articles/s42256-024-00837-3) |
| BioEmu                          | Diffusion (large-scale equilibrium emulator) | [Lewis et al., Science 2025](https://www.science.org/doi/10.1126/science.adv9817)  |

Several of the methods above incorporate molecular dynamics (MD) training datasets by simply treating them as independent possible outputs. That is, a protein with 300 structures calculated by molecular dynamics would be 300 possible target structures for the same input sequence. The model never sees the time-ordering; the frames are treated as independent samples from the protein's structure distribution. 

## Why Premval

Six methods, six papers, six tables. Each one reports its own numbers on its own chosen proteins with its own favorite metrics. One scores a handful of well-studied proteins, another scores a few hundred; one likes RMSF correlation, another likes Wasserstein distances. So "state of the art" usually just means "best in the table the authors built." Not very satisfying.

It gets worse. Several of these models trained on molecular dynamics data and then got evaluated on molecular dynamics data from the same place. When AlphaFlow-MD is scored on ATLAS proteins it was fine-tuned on, a nice number is partly memorization, and nothing in the paper tells you how much.

So Premval is the neutral ground. Every model runs against the same proteins (ATLAS), through the same metrics, with a label on each one for whether it saw that data in training. One leaderboard, same rules for everybody, and "graded its own homework" is a column instead of something you have to dig out of a methods section. The rest of this post is about the metrics, because that is where the real work happens.

## Metrics

Individual structure prediction methods compare one single structure to another. In the simplest case we have a single protein which is a linear string of standard amino acids, each of which has standard notations for every atom. We align the two via rotation and translation to minimize the root-mean-squared deviation (RMSD), and start comparing. RMSD itself is the simplest and most obvious metric. There are more to choose from. The lDDT (local distance-difference test) which operates on the distogram level, and thus does not require a global alignment. Also the much more general template modelling score (TM-score), which *is* a global alignment, and allows us to compare non-equivalent proteins.

Ensembles have their own difficulties. Thankfully we have decades of mathematics to pull from.

### Wasserstein Distance

By far the most popular tool for comparing distributions is the [Wasserstein Metric](https://en.wikipedia.org/wiki/Wasserstein_metric), also called the Earth Mover's Distance. $W_1$ asks what is the least amount of "mass" we would need to move from one place to another to convert one distribution $p_1$ to another $p_2$. It is defined in any number of dimensions, although defined does not mean easy to calculate. However, the $W_2$ distance between Gaussian distributions has a nice closed form:
$$
W^2_2 (X_0, X_1) = (μ_0 − μ_1)^2 + (σ_0 − σ_1)^2 
$$
The above formula is for a single dimension. In many dimensions it gets a bit more complicated, but basically the scalars turn into matrices:
$$
W^2_2 (X_0, X_1) = ||\vec{μ_0} − \vec{μ_1}||^2 + Tr(\Sigma_0 + \Sigma_1 - 2(\Sigma_0^{1/2} \Sigma_1 \Sigma_0^{1/2})^{1/2} )
$$
AlphaFlow introduced a metric based on this formula which the authors called the *root mean Wasserstein distance* (RMWD), which consists of a few pieces:
1. Let $\mathcal{N}[X_i]$ be the set of Gaussians fit to the positional distribution of each atom $i$ in one ensemble $X$, and say we are comparing to another ensemble $Y$.
2. $RMWD ^2 = (1/N)\sum_{i}^N W_2^2(\mathcal{N}[X_i], \mathcal{N}[Y_i])$

So basically we fit the atoms to a Gaussian distribution and take the mean distance between all of those distributions. 

### MD-PCA Wasserstein Distance

This one is a bit more complicated. First, a PCA is performed on the $C\alpha$ positions of the protein as determined by MD (ground truth). This sets the basis vectors for motion. Next we project the positions of the candidate and reference structures onto the top PCA vectors, and calculate the Wasserstein distance between these distributions. This captures the dominant global collective motion modes. 

### Atomic Contacts

We are dealing with biological molecules here, so it makes sense to consider the actual physical interactions. AlphaFlow introduced "weak and transient contacts":

> defined as those Cα pairs which are in contact (respectively, not in contact) in the crystal structure but which dissociate (respectively, associate) in > 10% of ensemble structures, with a 8 Å threshold

Source: [Jing et al., ICML 2024](https://arxiv.org/abs/2402.04845) (AlphaFlow)

### Root-Mean-Squared Fluctuation Correlation

The root-mean-squared fluctuation (RMSF) is a measure of motion. We simply measure the root-mean-squared distance of an atom from its mean. The metric is the Pearson correlation between the fluctuation numbers of the candidate ensemble vs the reference (either all-heavy-atom or $C\alpha$). 

### Thermodynamic and kinetic readout

BioEmu reported thermodynamic stability metrics as well: folding ΔG, mutation ΔΔG, cryptic-pocket recovery rates. These need reference data (experimental stability, kinetics-resolved MD) that ATLAS doesn't provide, so they aren't present in most published methods, and they are not present in Premval.

## So Which Generator Should You Use?

Here is the uncomfortable answer the whole metrics section was building toward: no single number tells you whether an ensemble is any good. A model can nail RMSF correlation (every residue wiggles by the right amount) and still put the entire distribution in the wrong place. Wasserstein distance can look great while half the structures have atoms clashing straight through each other. Contacts catch the functional motions and miss the global shape. Pick your favorite metric and some model is the "winner." That is exactly how we ended up with six papers, each on top of its own table.

So Premval doesn't report a score. It reports the whole panel, side by side, the same proteins for everybody. And next to every model sits the column that actually matters: did it train on the data it is being scored on? A model can top the ATLAS leaderboard by quietly reciting trajectories it already memorized, and without that label you would never know.

So, which generator should you use? It depends on what you need (raw flexibility? the right global ensemble? physically valid frames?), and the point is that you can finally look instead of taking an author's word for it. The board is here: [the Premval leaderboard](LEADERBOARD_URL). Sort by the metric you care about, check the contamination badge, and weight the in-distribution numbers accordingly.

One honest caveat. Everything above lives on ATLAS, which means equilibrium fluctuations around a folded state. The things ATLAS can't see (folding free energies, mutation effects, slow kinetics) are exactly the things Premval can't score yet, and they are where the next generation of these models is heading. That is the v2 problem. For now, if you want to know which ensemble generator to actually run, this is the first place that will give you a straight answer.

