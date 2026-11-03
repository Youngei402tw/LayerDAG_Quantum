# Paper draft

`main.tex` is the shared portable two-column manuscript covering the frozen random-circuit study and fresh five-seed structured-family replication. `qce_paper.tex`/`.pdf` use quantum-compiler framing; `qai_paper.tex`/`.pdf` use out-of-distribution AI framing. `CONTRIBUTION_EXPLAINER.md` gives the argument in plain language. Author names, affiliations and official venue templates remain placeholders.

Build from this directory with `pdflatex main.tex`, `bibtex main`, then two more `pdflatex` passes. The built `main.pdf` is committed for review. The generated figure is committed under `figures/results.pdf`; its source values come from the checked experiment JSON files.

`size_transfer_paper_v3.pdf` is a versioned copy of the current shared build for review tools that cache `main.pdf` by path. `representation_reversal_paper_v2.pdf` is retained as the pre-replication draft and should not be submitted.

The QCE and QAI packages are the current submission drafts. Transfer the appropriate source into the live official template after confirming the current call's track, page limit, anonymity policy, and artifact rules. A main-track general ML submission would still require public real-circuit benchmarks and broader family transfer.
