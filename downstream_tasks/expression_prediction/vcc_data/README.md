# VCC data bundled with the training entrypoint

`context_arrays/` contains the compact qnorm-TPM training contract used by the
composite loss: expression, matched non-targeting control, true LFC,
Wilcoxon+BH significance mask, ordered target genes and ordered response genes.

Active training contexts:

- Orion HCT116: 300 perturbations x 20,343 response genes;
- Orion HEK293T: 300 x 20,343;
- K562 GWPS: 272 x 7,941.

H1 (25 x 17,552) is validation-only. RPE1, HepG2 and Jurkat are recorded in
`DATASETS.tsv`, but are deliberately excluded because their overlap with the
300 challenge targets is zero. No proxy labels, zero masks or duplicated
perturbations are substituted.

`intervals/` fixes the exact response-gene order. `pert_inputs/` contains the
TF-panel order/bin edges and provenance mappings. Large raw single-cell H5AD,
the reference genome, tokenizers and model checkpoints remain outside the Git
checkout at the unchanged paths in `final_pert_vcc.yaml`.
