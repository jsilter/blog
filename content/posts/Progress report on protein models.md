---
title: "Deep Learning for Proteins: From Prediction to Design"
date: 2026-03-29
draft: true
tags: ["structural-biology", "machine-learning", "protein-structure", "drug-discovery"]
summary: "Protein structure prediction is effectively solved for monomers. The field's center of gravity has shifted to a harder problem: designing new proteins that bind specific targets."
---

In 2020, AlphaFold 2 scored a median GDT-TS of 92.4 at CASP14, reaching experimental accuracy on two-thirds of its targets. Five years later, predicting a protein's structure from its sequence is table stakes. The interesting question now is whether we can design proteins that do what we want.

## Structure Prediction Is Saturating

For the decade before AlphaFold, CASP scores were flat. The best groups averaged a GDT-TS in the mid-40s. AlphaFold jumped to 64 at CASP13 (2018); AlphaFold 2 hit 88 at CASP14 (2020). The most recent winners (2022, 2024) were not from the AlphaFold group but used basically the same architecture.

{{< figure src="/images/protein-models/casp_gdt_ts.png" caption="Average GDT-TS for the top group at each CASP round. Source: predictioncenter.org." >}}

AlphaFold 3 (May 2024) expanded scope to complexes, ligands, and nucleic acids, but added only ~0.01 lDDT on monomers. Within 18 months, open-source models (Chai-1, Boltz-1, HelixFold3, Protenix) matched it. On FoldBench, every model lands between 0.87 and 0.89 lDDT. The spread is 0.02.

{{< figure src="/images/protein-models/monomer_lddt.png" caption="Protein monomer lDDT. Left: CAMEO2022 (easier test set). Right: FoldBench (post-2023 PDB, harder). Not directly comparable across panels. Sources: ProteinBench (arXiv:2409.06744); FoldBench (Nature Comms 2025); SeedFold (arXiv:2512.24354)." >}}

## Existing Gaps

The structure of the average protein monomer has been solved. But it's generally protein interactions that we care about most. To design a small-molecule drug, we'd like to know how it interacts with a target protein. If our intended drug is a biologic, then it's protein-protein interactions we care about. In particular, most therapeutic proteins are antibodies. Antibodies fall into a few fixed scaffolds combined with highly variable regions (complementarity determining regions, CDRs) which determine the binding activity. Most therapeutic proteins are antibodies, so it's critical to understand how antibodies fold and bind.

Unfortunately that still presents a serious weak spot. All of these models rely on multiple-sequence-alignments, which compare the sequence of a protein to a database of known proteins, and make the assumption that residues which mutate together are likely close together in space. This falls apart with antibodies. We know exactly where the complementarity determining regions are, and they represent ~20 of the ~500 amino acid antibody, or ~5%. For most classes of proteins, 90% sequence identity means they do the same thing. Here that's still partially true (they're both antibodies) but we require much finer resolution to have meaningful understanding. 

Antibody-antigen prediction ranges from 24% to 53% acceptable; ligand docking from 51% to 67%.

{{< figure src="/images/protein-models/complex_dockq.png" caption="% acceptable predictions by task on FoldBench. DockQ > 0.23 for complexes; RMSD < 2A for ligands. Sources: FoldBench (Nature Comms 2025); SeedFold (arXiv:2512.24354)." >}}

## From Prediction to Design

The harder problem is the inverse: given a target, design a molecule that binds it. A new generation of generative models (flow matching, diffusion) aims to do this. The metric that matters is the **per-candidate hit rate**: what fraction of designs actually bind when tested in the lab?

The most controlled comparison comes from NVIDIA's Proteina-Complexa paper (March 2026), which tested five methods head-to-head on the same 75 targets using the same phage display assay:

{{< figure src="/images/protein-models/antibody_hitrate.png" caption="On-design specific hit rate. Same 75 targets, same assay, comparable compute. Source: Proteina-Complexa (NVIDIA, March 2026)." >}}

Proteina-Complexa achieves 2.45%; BoltzGen with ProteinMPNN redesign reaches 1.81%; RFDiffusion3 trails at 0.04-0.41%. A 2.45% hit rate means testing roughly 40 candidates to find one binder. Much better than screening millions of compounds, but you still can't design one protein and expect it to work.

Published hit rates from other groups are hard to compare directly because the generate-and-filter pipeline varies. BoltzGen generates 60,000 candidates per target, filters through Boltz-2 confidence scoring, and tests only the top 15; on those filtered candidates, 11-24% bind. Manifold Bio's mBER tested over a million designs with no pre-filtering and saw 0.5-0.7% hits; filtering post-hoc by ipTM > 0.8 would have enriched to ~4%. The generative model and the confidence filter are equally important parts of the system.

Progress is fast. Proteina-Complexa achieved picomolar affinity (93.6 pM) on PDGFR in a single design round. BoltzGen produced single-digit nanomolar binders for novel targets without affinity maturation. The scale of validation campaigns has grown from dozens of designs to millions in under two years.

## Where This Is Going

Monomer structure prediction was solved and commoditized within five years of AlphaFold 2. 
Interaction predictions have made significant progress but are a far cry from solved. The OpenFold Consortium [recently declared as such](https://www.businesswire.com/news/home/20260313170622/en/OpenFold-Consortium-Announces-Major-OpenFold3-Update-and-Public-Release-of-Training-Data-for-Reproducible-Biomolecular-AI)

>The consortium noted that antibody–antigen complex prediction remains a challenging frontier for the field, with current methods (including OpenFold3) showing room for improvement in this domain. Improving antibody–antigen performance is a major 2026 priority, with planned work spanning data expansion, benchmarking, and model improvements targeted to immune-relevant complexes.



---

## References

### Structure Prediction

- Senior AW et al., "Improved protein structure prediction using potentials from deep learning," *Nature* 577, 706-710 (2020). DOI: 10.1038/s41586-019-1923-7
- Jumper J et al., "Highly accurate protein structure prediction with AlphaFold," *Nature* 596, 583-589 (2021). DOI: 10.1038/s41586-021-03819-2
- Abramson J et al., "Accurate structure prediction of biomolecular interactions with AlphaFold 3," *Nature* 630, 493-500 (2024). DOI: 10.1038/s41586-024-07487-w
- Kryshtafovych A et al., "CASP16 protein monomer structure prediction assessment," *Proteins* (2025).
- FoldBench: *Nature Communications* (2025). https://www.nature.com/articles/s41467-025-67127-3
- SeedFold: arXiv:2512.24354 (December 2025).
- Gao et al., *Briefings in Bioinformatics* 26(6), bbaf616 (2025).
- ProteinBench: arXiv:2409.06744 (ICLR 2025).

### Model Papers

- Chai-1: Boitreaud et al., "Chai-1: Decoding the molecular interactions of life," bioRxiv (2024). DOI: 10.1101/2024.10.10.615955
- Boltz-1: Wohlwend et al., "Boltz-1: Democratizing Biomolecular Interaction Modeling," bioRxiv (2024). DOI: 10.1101/2024.11.19.624167
- Boltz-2: Passaro et al., "Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction," bioRxiv (2025). DOI: 10.1101/2025.06.14.659707
- Protenix-v1: "Protenix-v1: Toward High-Accuracy Open-Source Biomolecular Structure Prediction," bioRxiv (2026). DOI: 10.64898/2026.02.05.703733
- HelixFold3: "Technical Report of HelixFold3 for Biomolecular Structure Prediction," arXiv:2408.16975 (2024).
- ESMFold: Lin Z et al., "Evolutionary-scale prediction of atomic-level protein structure with a language model," *Science* 379, 1123-1130 (2023). DOI: 10.1126/science.ade2574

### Protein Design

- Bennett et al., "Atomically accurate de novo design of antibodies with RFdiffusion," *Nature* (2025). DOI: 10.1038/s41586-025-09721-5
- Stark et al., "BoltzGen: Toward Universal Binder Design," *Science* (2025). DOI: 10.1101/2025.11.20.689494
- Manifold Bio, "mBER: Controllable de novo antibody design with million-scale experimental screening," bioRxiv (2025). DOI: 10.1101/2025.09.26.678877
- NVIDIA, "Latent Generative Search unlocks de novo Design of Untapped Biomolecular Interactions at Scale," ICLR 2026 Oral.
- NVIDIA, "Proteina-Complexa Validation," March 2026.
- Geffner et al., "Proteina: Scaling Flow-based Protein Structure Generative Models," arXiv:2503.00710, ICLR 2025 Oral.
- NVIDIA, "La-Proteina: Atomistic Protein Generation via Partially Latent Flow Matching," arXiv:2507.09466 (2025).

### CASP Official Results

- predictioncenter.org/casp{10-16}/groups_analysis.cgi
