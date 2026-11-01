# Paper draft

`main.tex` is a portable two-column conference-style draft centered on the frozen three-seed size-transfer result. It treats downstream rewrite experiments as secondary evidence and reports their failed confirmatory gates. Author names, affiliations and final venue formatting remain placeholders.

Build from this directory with `pdflatex main.tex`, `bibtex main`, then two more `pdflatex` passes. The built `main.pdf` is committed for review. The generated figure is committed under `figures/results.pdf`; its source values come from the checked experiment JSON files.

The most defensible current submission target is an IEEE QAI or IEEE Quantum Week/QCE research/short-paper track focused on quantum software and machine learning. A main-track general ML submission would require broader real-circuit benchmarks, more seeds and a passed practical compiler endpoint.
