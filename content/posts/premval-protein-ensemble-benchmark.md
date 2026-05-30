---
title: "Benchmarking Protein Ensemble Generators"
date: 2026-05-25
draft: true
tags: ["structural-biology", "machine-learning", "protein-structure", "benchmarks"]
summary: "Generative models for protein conformational ensembles are arriving monthly, but the field has no neutral cross-model comparison. A tour of why ensemble evaluation is hard and where the methods landscape sits today."
---

<!-- TODO: hook / opening section (why ensembles matter; what AlphaFold didn't solve) -->
## One Structure Isn't Everything

![[Pasted image 20260525200646.png|399]]

Classical regression models select a single best answer, often the maximum-likelihood estimate. For many situations this is exactly what we want, or at least good enough. But for many complex outputs it very much is not. Some of the first image generators were variational autoencoders and often had this same failure mode:

![[Pasted image 20260525201031.png|187]]

By minimizing the average error against multiple datapoints we end up with something unrealistic. Diffusion models handle this issue more elegantly:

![[Pasted image 20260525201425.png|273]]

(Image credit [DiffDock](https://www.youtube.com/watch?v=_KBqVh6YbgI)) For image generators we can settle for realistic generation. While we definitely care about mode collapse ("smiling face" shouldn't just generate 20-year-old American white guys, there should be a range) so some diversity is required, we don't need to cover the complete distribution of all possible images for a given caption.

Proteins are different. If we want to understand the full spectrum of activity for a given protein, we need to know the full spectrum of conformations. <!--That rare binding pocket I think I heard about?-->  Models like AlphaFold3 and Boltz are diffusion based, so the architecture supports sampling multiple structures. However, these models were trained against target structures from the PDB. Not ensembles, single structures. So while the architecture might support

## Ensemble Generators

These have received less press (and certainly no Nobel prizes...yet) but I think this family will be much more important in the long run. These models attempt a much harder task: instead of just sampling some elements, we wish to cover the entire distribution. 

A typical diffusion method involves learning to reverse added noise by learning a score function. The score function is typically a deep neural network, using some combination of convolution and attention layers.

![[Pasted image 20260530140113.png]]

Source: Score-Based Generative Modeling through Stochastic Differential Equations https://arxiv.org/abs/2011.13456

 So basically guess and check. We need a function capable of the "check" step, which is no easy task, but conceptually we just pick random directions and our function says "hot" or "cold". 

Flow matching is an alternative to diffusion. Instead of learning a score function, we learn an entire vector field capable of transitioning a sample from one distribution (ie noisy random) to another (ie our prediction). As stated, learning such a function is intractable, but with some clever mathematics we can approximate it.

<!--But that's not really the advantage; diffusion models could be trained against ensemble models as well. The special advantage was in the training data; AlphaFlow was trained against [ATLAS](https://www.dsimb.inserm.fr/ATLAS/about.html), a dataset of protein structures calculate from molecular dynamics. Ensemble prediction methods lose the explicit timing dynamics from MD and instead treat -->

## State of the Field: Metrics

Individual structure prediction methods compare one single structure to another. In the simplest case we have a single protein which is a linear string of standard amino acids, each of which has standard notations for every atom. We align the two via rotation and translation to minimize the root-mean-squared deviation (RMSD), and start comparing. RMSD itself is the simplest and most obvious metric. There are more to choose from. The LDDT (local distance-difference test) which operates on the distogram level, and thus does not require a global alignment. Also the much more general template modelling score (TM-score), which *is* a global alignment, and allows us to compare non-equivalent proteins .

Ensembles have their own difficulties. Thankfully we have decades of mathematics to pull from.

### Wasserstein Distance

By far the most popular tool for comparing distributions is the [Wasserstein Metric](https://en.wikipedia.org/wiki/Wasserstein_metric), also called the Earth Move's Distance. $W_1$ asks what is the least amount of "mass" we would need to move from one place to another to convert one distribution $p_1$ to another $p_2$. It is defined in any number of dimensions, although defined does not mean easy to calculate. However, the $W_2$ distance between Gaussian distributions has a nice closed form:
$$
W^2_2 (X_1, X_2) = (μ_0 − μ_1)^2 + (σ_1 − σ_0)^2 
$$
The above formula is for a single dimension. In many dimensions it gets a bit more complicated, but basically the scalars turn into matrices:
$$
W^2_2 (X_1, X_2) = ||\vec{μ_0} − \vec{μ_1}||^2 + Tr(\Sigma_1 + \Sigma_2 - 2(\Sigma_1^{1/2} \Sigma_2 \Sigma_1^{1/2})^{1/2} 
$$
AlphaFlow introduced a metric based on this formula which the authors called the *root mean Wasserstein distance* (RMWD), which consists of a few pieces:
1. Let $\mathcal{N}[X_i]$ be the set of Gaussians fit to the positional distribution of each atom $i$ in the ensemble
2. $RMWD ^2 = (1/N)\sum_{i}^N W_2^2(\mathcal{N}[X_i], \mathcal{N}[Y_i])$

So basically we fit the atoms to a Gaussian distribution and take the mean distance between all of those distributions. 

### MD-PCA Wasserstein Distance

This one is a bit more complicated. First, a PCA is performed on the $C\alpha$ positions of the protein as determined by MD (ground truth). This sets the basis vectors for motion. Next we project the positions of the candidate and reference structures onto the top PCA vectors, and calculate the Wasserstein distance between these distributions. This captures the dominant global collective motion modes. 

### Physical Validity

We are dealing with biological molecules here, so it makes sense to consider the actual physical interactions. AlphaFlow introduced "weak and transient contacts":
> defined as those Cα pairs which are in contact (respectively, not in contact) in the crystal structure but which dissociate (respectively, associate) in > 10% of ensemble structures, with a 8 ˚A threshold

Source: Bowen Jing, Bonnie Berger, and Tommi Jaakkola. "AlphaFold Meets Flow Matching for Generating Protein Ensembles." In Forty-first International Conference on Machine Learning (ICML), 2024. https://arxiv.org/abs/2402.04845

<!--AI Generated
The metric landscape sorts into four buckets that measure different things.

**Flexibility.** Per-residue RMSF correlation (Pearson r between model and MD) is the single most-reported number in the field. It tells you whether your model is wiggling the right residues, but it's invariant to whether those wiggles have the right *shape* in 3D.

**Distributional accuracy.** AlphaFlow's contribution to the field's metric vocabulary is the Root Mean Wasserstein Distance (RMWD), a per-atom 2-Wasserstein between Gaussian fits, plus the MD-PCA Wasserstein-2 (project both ensembles into a PCA basis fit to MD, take EMD). These actually measure whether the *distribution* of conformations matches, not just per-residue variance. ConfDiff and ESMDiff prefer Jensen-Shannon divergences on pairwise Cα distances, Rg, and TIC components; broadly the same idea in a different basis.

**Ensemble observables.** AlphaFlow added the weak/transient contact Jaccard (which crystal contacts dissociate? which non-crystal contacts form?) and the exposed-residue Jaccard. These do real work: they separate collapsed-into-the-crystal models from models that actually sample apo/holo or fold-switching behavior.

**Physical validity.** Per-frame realism: clash rate, peptide-bond geometry, Ramachandran allowed-region fraction. ProteinBench surfaces these prominently; the alignment-on-top methods (EBA, EPO) treat them as a reward signal.

**Thermodynamic and kinetic readouts.** BioEmu pushes into territory the AlphaFlow panel doesn't touch: folding ΔG, mutation ΔΔG, cryptic-pocket recovery rates. These need reference data (experimental stability, kinetics-resolved MD) that ATLAS doesn't provide.

ProteinBench (Wang et al., 2024) attempts a holistic survey across all of these, but it's a survey rather than a head-to-head: it doesn't run every model on identical inputs and report a single ranking. The AlphaFlow panel is the closest thing the field has to a shared standard, and even that is essentially one paper's choice that the next two papers adopted.

## The Problem: Generators Outpacing Evaluation

The protein conformational ensemble field is in a stampede. AlphaFlow, ESMFlow, BioEmu, ConfDiff, Str2Str, ESMDiff, DiG, ANewSampling, JAMUN, EBA, EPO; new bioRxiv entries appear every month. Each paper benchmarks its own model on its own targets, often with its own bespoke metric panel.

There is no neutral cross-model comparison. Numbers in different papers aren't apples-to-apples, and the closest thing the field has to a shared evaluation (AlphaFlow's `analyze_ensembles.py` panel) is rarely run against models the AlphaFlow authors didn't release themselves.

This gap is real and undersupplied; a public, reproducible head-to-head comparison would let practitioners actually choose a model.

## Why Evaluation Is Hard

A few things make this harder than it looks.

**Reference data is heterogeneous.** ATLAS gives you standardized 3 × 100 ns CHARMM36m simulations across ~1,400 proteins, which is the closest thing to a canonical MD dataset. PED (Protein Ensemble Database) is the cleanest source of disordered-protein ensembles, mostly experimental. mdCATH ships 3.6 TB of simulations. DESRES fast-folders are smaller but kinetics-resolved. Beyond these, most papers cite their own bespoke MD.

**Metrics span families that don't substitute for each other.** Flexibility metrics (per-residue RMSF correlation) tell you whether the model is wiggling the right loops. Distributional metrics (RMWD, MD-PCA Wasserstein) measure whether the *shape* of the ensemble matches MD. Ensemble observables (contact Jaccard, exposed-residue analysis) measure whether functionally relevant motions are captured. Validity metrics (clashes, peptide-bond geometry, Ramachandran) measure whether per-frame samples are physically reasonable. Thermodynamic readouts (ΔG, ΔΔG) and kinetic ones (TICA divergences) require fundamentally different reference data.

**Sample counts matter, and most papers don't normalize.** AlphaFlow's README is explicit that its results "are not comparable across different sample counts." Quoting metrics from two papers that sampled differently is a category error.

**Training-set contamination is real but invisible.** AlphaFlow-MD is trained on ATLAS and is scored on ATLAS. ANewSampling treats ATLAS as held-out. BioEmu trained on a huge MD + AFDB + stability corpus whose overlap with ATLAS is unclear. No paper surfaces this on the same plot.

The headline rankings in each paper bake in all four of these choices.

## State of the Field: Generators

A short tour, organized by methodological family. (I'll skip historical baselines like idpGAN and EigenFold; the action is in the last two years.)

**Flow-matching on single-state predictors.** AlphaFlow and ESMFlow (Jing et al., ICML 2024) take AlphaFold2 and ESMFold respectively and fine-tune them under a flow-matching objective, turning a deterministic predictor into a sequence-conditioned generator. The MD-trained variants (AlphaFlow-MD, ESMFlow-MD) are further fine-tuned on ATLAS. Their evaluation script (`analyze_ensembles.py`) has become the de facto reference panel for the field, and their 250-frame multi-model PDB outputs on HuggingFace are the closest thing to a community baseline.

**SE(3) diffusion.** ConfDiff (Wang et al., ICML 2024) is an SE(3) diffusion model over backbone frames with a force-guided score combination, so samples track the Boltzmann distribution; it ships an ATLAS-fine-tuned checkpoint. Str2Str (Lu et al., ICLR 2024) is the zero-shot variant: trained only on PDB crystals, no MD data required, samples by perturbing-and-denoising an input structure.

**Structure language models.** ESMDiff (Lu et al., ICLR 2025) encodes 3D structures into a discrete latent and does masked diffusion over those tokens, on top of ESM3. Reports 20-100× speedup over continuous-coordinate diffusion methods. The catch: ESM3's weights are gated behind a non-commercial license, which constrains downstream use.

**Equilibrium emulators at scale.** BioEmu (Lewis, Hempel, Jiménez-Luna, et al., Science 2025) is the most ambitious. The authors integrated >200 ms of MD plus PDB/AFDB statics plus experimental stability data into a single emulator that generates thousands of independent structures per hour on one GPU. It reports relative free energies to ~1 kcal/mol and predicts cryptic-pocket and conformational-change behavior; the readouts go well beyond the AlphaFlow geometry panel.

**Distributional Graphormer (DiG).** Microsoft's earlier line (Zheng et al., Nature MI 2024), a diffusion model conditioned on a system descriptor; scope broader than proteins (also catalysts and ligand poses). The trained-weights link expired in April 2025, so in practice DiG is currently unusable from outside the original team.

**Physical-feedback alignment.** EBA (Lu et al., ICML 2025) and EPO (Sun et al., AAAI 2026) aren't standalone generators but DPO-style alignment layers that fine-tune an existing generator (Protenix base for EBA, MDGen for EPO) with an energy-based preference signal. EBA's checkpoint and a 250-sample artifact are live; EPO's GitHub repo is currently an empty placeholder.

A practical map of what's actually usable today:

| Method | Pre-generated ensembles | Weights runnable | License |
|---|---|---|---|
| AlphaFlow / ESMFlow | Yes (HF, multi-model PDB) | Yes | MIT |
| ConfDiff | No (generate yourself) | Yes | Apache-2.0 |
| Str2Str | No | Yes | MIT |
| ESMDiff | No | Partial (ESM3 gated) | Non-commercial |
| BioEmu | No (benchmark harness exists) | Yes | MIT |
| EBA | Partial (250 test samples) | Yes | MIT + Apache-2.0 |
| DiG | No | Effectively no (weights link dead) | Unstated |
| EPO | No | No (empty repo) | MIT (declared) |

If you wanted to compare these head-to-head today, AlphaFlow and ESMFlow are the only ones where you can skip the inference step entirely.

-->

<!-- TODO: section on contamination as the hidden killer -->
<!-- TODO: PREMVAL teaser / what I'm building -->
<!-- TODO: open questions and caveats -->
