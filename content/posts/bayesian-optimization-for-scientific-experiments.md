---
title: "Bayesian Optimization for Scientific Experiments"
date: 2026-03-09
draft: true
tags: ["bayesian-optimization", "machine-learning", "neuroscience"]
summary: "How Bayesian optimization can help scientists design better experiments by intelligently selecting which measurements to make next."
---

Experimental science is expensive. Each data point in a patch-clamp electrophysiology experiment might take hours of skilled labor, specialized equipment, and carefully prepared biological samples. When you can only run a handful of experiments, choosing *which* experiments to run becomes a critical decision.

This is where Bayesian optimization (BO) comes in.

## The Problem: Sequential Experimental Design

Imagine you have a catalog of ~1,000 neurons, each characterized by features like brain region, cell type, and morphological measurements. You want to find the neuron with the lowest rheobase (the minimum current needed to trigger an action potential), but measuring the rheobase requires running a full electrophysiology experiment on each candidate.

You can afford maybe 30 measurements. How do you pick which 30 neurons to test?

## The Naive Approaches

**Random selection** is the simplest baseline: just pick 30 neurons at random. This works surprisingly well as a sanity check, but it makes no use of what you learn along the way.

**Heuristic approaches** might rank candidates by some domain-knowledge-driven score, but they don't adapt as new data comes in.

## Bayesian Optimization

BO takes a fundamentally different approach. It fits a probabilistic model (typically a Gaussian process) to the observations collected so far, then uses that model to decide which point to measure next. The key insight is that the GP gives you both a prediction *and* an uncertainty estimate for every unmeasured candidate.

The acquisition function balances two competing goals:

- **Exploitation**: measure candidates where the model predicts good outcomes
- **Exploration**: measure candidates where the model is uncertain (and might be surprised)

Common acquisition functions include Expected Improvement (EI), Upper Confidence Bound (UCB), and Knowledge Gradient (KG). In my experience with the neuroscience use case, EI tends to be a reliable default, while UCB can be tuned for more or less exploration via its beta parameter.

## What I Built

I built a [Bayesian optimization framework](https://github.com/jsilter/botorch_mcp) specifically for this kind of discrete candidate selection problem using [BoTorch](https://botorch.org/). Some design choices that mattered:

- **Discrete candidate sets**: rather than optimizing acquisition functions over continuous space, we evaluate them over the full set of ~1,000 known candidates. This is tractable for datasets of this size and avoids the complexity of mixed continuous/categorical optimization.
- **Paired evaluation**: when comparing strategies, the same random seed produces the same initial observations, enabling paired statistical tests (Wilcoxon signed-rank with Benjamini-Hochberg correction).
- **MCP server integration**: the framework exposes BO tools via the Model Context Protocol, letting an LLM act as a research assistant that can suggest experiments, fit models, and explain its reasoning.

## Results

Across multiple electrophysiology objectives, BO consistently outperforms random selection, finding better optima with fewer measurements. The gap is most pronounced in the first 20-30 iterations, which is exactly the regime that matters when experiments are expensive.

The most interesting finding is that LLM-augmented BO (where an LLM reranks the top candidates from the acquisition function using domain knowledge) can sometimes beat pure BO, though the effect is inconsistent across objectives and the additional API cost may not always be justified.

## When to Use BO

Bayesian optimization shines when:

1. Evaluations are expensive (time, money, or resources)
2. You have a fixed or enumerable set of candidates
3. The objective is noisy but has some learnable structure
4. You can run experiments sequentially (each measurement informs the next)

If your experiments are cheap or highly parallelizable, simpler approaches (or batch BO) may be more appropriate. But for the kind of careful, one-at-a-time experimental design common in biology and materials science, BO is a powerful tool.
