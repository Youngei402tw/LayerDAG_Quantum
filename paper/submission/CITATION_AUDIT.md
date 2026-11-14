# Citation audit (2026-11-06)

Each entry cited in `paper/qai_paper.tex` was checked against a primary or publisher listing. Entries that are in `references.bib` but not cited in the QAI paper (`broughton2021tfq`) are retained for the archived manuscripts.

| Key | Verified against | Result |
|---|---|---|
| wang2022quest | arXiv 2210.16724 abstract page; MIT HAN Lab project page (ICCAD '22, 41st IEEE/ACM ICCAD); DOI 10.1145/3508352.3561118 | Added; authors, venue, DOI confirmed |
| li2025layerdag | ICLR 2025 proceedings page; arXiv 2411.02322 author list | Confirmed |
| beaudoin2025qfusion | arXiv 2504.20794 (v1, 29 Apr 2025), paper text | Added. Cited only for what the text says ("additional parameters that build a hidden representation of the wire connections, similar to the way node types are represented"). The explicit `[-1,-1]`/`[0,1]` dictionary from earlier project discussion is **not** described in the paper text, so the manuscript does not attribute it to Q-Fusion |
| fosel2021circuitrl | arXiv 2103.07585 (Fösel, Niu, Marquardt, Li) | Added |
| li2019tackling | bibtex.github.io ASPLOS-2019-LiDX (ACM): DOI 10.1145/3297858.3304023, pp. 1001–1014 | **Corrected** (old DOI was wrong) |
| cowtan2019qubit | Dagstuhl DROPS BibTeX, LIPIcs vol. 135, 5:1–5:32 | **Corrected** (missing author Krajenbrink) |
| preskill2018nisq | Quantum 2, 79 (2018), DOI 10.22331/q-2018-08-06-79 | Unchanged |
| bronstein2021geometric | arXiv 2104.13478 | Unchanged |
| battaglia2018relational | arXiv 1806.01261 | Unchanged |
| zaheer2017deepsets | researchr / NeurIPS 2017 pp. 3394–3404 | Added |
| gilmer2017mpnn | ICML 2017 (PMLR 70) pp. 1263–1272 | Unchanged |
| kipf2017gcn | ICLR 2017 | Unchanged |
| velickovic2020neural | mlanthology ICLR 2020 | Added |
| yehudai2021size | mlanthology / PMLR 139 pp. 11975–11986 | Added |
| xu2021extrapolate | mlanthology ICLR 2021 | Added |
| cho2014gru | ACL Anthology D14-1179 BibTeX, DOI 10.3115/v1/D14-1179 | Added |
| vaswani2017attention | mlanthology NeurIPS 2017 pp. 5998–6008 | Added |
| farhi2014qaoa | arXiv 1411.4028 | Unchanged |
| peruzzo2014vqe | Nat. Commun. 5, 4213 (2014), DOI 10.1038/ncomms5213 | Unchanged |
| qiskit2024 | arXiv 2405.08810 abstract page (accessed 2026-11-14): title "Quantum computing with Qiskit", 12 authors from Javadi-Abhari to Gambetta, v1 14 May 2024 | Added in v5 |

The previously verified entries (Preskill, Gilmer, Kipf, Farhi, Peruzzo, Battaglia, Bronstein) are standard and their bibliographic fields were not changed. Re-check page ranges against the publisher if the camera-ready requires DOIs for every entry.

v5 note (2026-11-14): `references.bib` now carries an `@IEEEtranBSTCTL` control entry so that references with more than six authors print as first author "et al." (IEEE reference style). The full author lists remain in the `.bib` file.
