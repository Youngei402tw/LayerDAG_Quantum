# Reproducibility guide: QAI qubit-label-invariant study

This guide covers the active IEEE QAI manuscript (`paper/qai_paper.tex`). Commands run from the repository root. No GPU is needed.

## Environment
```sh
python3.12 -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0   # recorded build: 2.8.0+cpu
pip install -r requirements-qai.txt
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 QISKIT_PARALLEL=FALSE
```
Recorded stack: Python 3.12, PyTorch 2.8.0+cpu, NumPy 2.3.4, Qiskit 2.2.3, pytest 8.4.2. The 2026-11-06 migration check used the PyPI `torch==2.8.0` (cu128) wheel on CPU with Python 3.12.3 and still reproduced seed 111 bit for bit. Other builds or CPU kernels may differ in the last floating-point bits.

## Fast checks (under one minute, read-only)
```sh
PYTHONPATH=. pytest -q                           # expect: 39 passed
python scripts/verify_claims.py                  # hashes + every headline number; expect FAILURES: none
PYTHONPATH=. python scripts/check_dataset_hashes.py   # regenerates frozen splits; expect 6 x OK
python scripts/check_manuscript_numbers.py       # every decimal in the paper is derivable from JSON
```

## Figures and manuscript
```sh
PYTHONPATH=. python scripts/make_figures.py      # writes paper/figures/fig_representation.pdf, fig_results.pdf
cd paper && latexmk -pdf qai_paper.tex           # needs TeX Live with IEEEtran (texlive-publishers)
```

## Training reproductions (write to NEW directories only)
| Command | Output | Approx. CPU time (2 threads) |
|---|---|---|
| `python scripts/spot_reproduce_seed.py 111 experiments/<new_dir>` | one paired seed vs. recorded values | ~10 min |
| `PYTHONPATH=. python -m qdag.invariant_benchmark --final --output experiments/qai_invariant_final_reproduction` | `final_result.json` (primary, 5 seeds × 2 models) | ~40–50 min (recorded 2,243 s of training) |
| `PYTHONPATH=. python -m qdag.invariant_benchmark --transformer-control --output experiments/qai_transformer_control_reproduction` | `control_result.json` | ~6–10 min (recorded 349 s) |
| `PYTHONPATH=. python -m qdag.connectivity_controls --output experiments/qai_connectivity_controls_reproduction` | `controls_result.json` (pointer GRU, pointer Transformer, role-ablated DAG; 5 seeds) | ~65 min (recorded 3,807 s) |
| `PYTHONPATH=. python -m qdag.augmentation_control --kind aug_pointer_grid --output <dir>` and `--kind aug_grid`, then `--aggregate --output <dir>` | `augmentation_result.json` (relabeling-augmented pointer GRU and grid GRU; 5 seeds each) | ~50 min sequential, ~27 min as two parallel processes (recorded 2,962 s) |
| `PYTHONPATH=. python -m qdag.structured_transfer --final --output experiments/structured_transfer_final_reproduction` | supporting width/depth study | ~35–45 min (recorded 2,022 s) |
| `PYTHONPATH=. python -m qdag.schedule_benchmark --final --updates 6000 --output experiments/schedule_dag_final_reproduction` | supporting random-circuit study | ~30 min (recorded 1,574 s) |

Never point `--output` at a frozen directory (`*_final_2026*`, `qai_transformer_control_20261106`, `qai_connectivity_controls_20261113`, `qai_augmentation_control_20261113`).

## Seeds and data
| Study | Data base seeds | Model seeds | Splits |
|---|---|---|---|
| QAI primary + Transformer | 14100–14105 (14105 = relabeling RNG) | 111–115 | train 1,000; val 256; same/near/far/relabeled 512 each |
| QAI development | 13100–13105 | 101 | train 600; val 128; tests 128 |
| Structured supporting | 12100–12106 | 81–85 | train 1,000; val 256; five tests × 512 |
| Random-circuit supporting | 6100 (train), 6200 (val), 6300 (test), 6500 (larger); see `qdag/schedule_benchmark.py::run_final` | 51–53 | train 1,000; val 256; same-size 512; larger 512 |

Data generators: `qdag/routed_rewrite.py::{_qaoa,_vqe,_arithmetic,_chemistry}` via `qdag/structured_transfer.py::generate_structured`. Labels: `qdag/schedule_benchmark.py::critical_labels`.

## Result hashes (SHA-256)
| File | Hash |
|---|---|
| experiments/qai_invariant_final_20261102/final_result.json | 23f47f27337d0cfc56242926dd3988db34342cf5ea26c5d871e822d9cdb21247 |
| experiments/qai_transformer_control_20261106/control_result.json | 6989761a20ef5b3b0c7ea1f50b5f0cee8aced285fec62ac3d7b85fa3747f9cb0 |
| experiments/structured_transfer_final_20261102/final_result.json | 5235c71446f61e66251aea5e3482d39921f7cf42b4b93f40eb611081bb255692 |
| experiments/schedule_dag_final_20261028/final_result.json | 9144132e2bcdb0aa0628cb62f75dc017d28d52498366aca178b2e5569625b435 |
| experiments/qai_invariant_spot_reproduction_seed111_20261106/spot_reproduction.json | 4d379f8fce6f75306e22b20af181a78033325e6b449bdcd9d54ad788435f1c12 |

The dataset content hashes are embedded in each result file under `dataset_hashes`.

## Anonymous artifact
`IDENTIFIERS='name1|name2|handle|email' bash scripts/make_anonymous_artifact.sh qai_artifact.tar.gz` exports the committed tree without git history and refuses to package if any listed identifier remains. Model checkpoints are not versioned. Results are reproduced by retraining.

## License and citation
No license file exists yet. The project owner must choose one before any public release.
