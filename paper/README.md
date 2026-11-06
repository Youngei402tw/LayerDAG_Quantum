# Paper draft

`qai_paper.tex` and `qai_paper.pdf` are the active conference submission. They center the frozen qubit-label-invariant five-seed result, far-width transfer to 20 qubits, and matched GRU/Transformer controls. `main.*` and `qce_paper.*` are retained research-history drafts and should not be submitted. Author names and the current official QAI template remain placeholders.

`qai_submission_v1.pdf` is a versioned copy of the active QAI PDF for review tools that cache files by path.

Build from this directory with `pdflatex main.tex`, `bibtex main`, then two more `pdflatex` passes. The built `main.pdf` is committed for review. The generated figure is committed under `figures/results.pdf`; its source values come from the checked experiment JSON files.

`size_transfer_paper_v3.pdf` is a versioned copy of the current shared build for review tools that cache `main.pdf` by path. `representation_reversal_paper_v2.pdf` is retained as the pre-replication draft and should not be submitted.

Transfer `qai_paper.tex` into the live official QAI template only after confirming the current call's page limit, anonymity policy, and artifact rules. Direct access to the official site was blocked by the execution proxy, so no deadline or formatting rule is asserted in this repository.
