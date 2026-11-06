# Paper directory

**Active submission (IEEE QAI):** `qai_paper.tex` → `qai_paper.pdf`. The versioned copy is `qai_submission_v2.pdf`, built 2026-11-06 on branch `migration/qai-submission-20261106`. The source uses IEEEtran V1.8b (`conference,a4paper`), is double-blind, and is 5 pages including references.

Build: `latexmk -pdf qai_paper.tex` (needs TeX Live `texlive-publishers` for IEEEtran). Figures: `PYTHONPATH=.. python ../scripts/make_figures.py` from this directory, or `PYTHONPATH=. python scripts/make_figures.py` from the repo root. Number audit: `python ../scripts/check_manuscript_numbers.py`.

Submission documents are in `submission/`:

- `COMPLIANCE_REPORT.md`: official QAI rules with URLs, and template compliance;
- `CLAIM_EVIDENCE.md`: claim-to-evidence table, limitations and negative-results audit, and discrepancies;
- `CITATION_AUDIT.md`;
- `PORTAL_MATERIALS.md`: abstract, keywords, contribution and limitation statements, disclosures;
- `SUBMISSION_CHECKLIST.md`: checklist and unresolved author and venue fields;
- `CHANGELOG_FROM_304144c.md`.

History, kept unchanged:

- `archive/qai_paper_handoff_304144c.{tex,pdf}`: the handoff QAI draft;
- `qai_submission_v1.pdf`: the handoff versioned PDF (SHA-256 `426cb5f3…`);
- `archive/references_handoff_304144c.bib`;
- `figures/{qai_invariant,representation,results,structured_transfer}.*`: handoff figures, with no plotting scripts recorded.

The archived manuscripts `main.*`, `qce_paper.*`, `size_transfer_paper_v3.pdf` and `representation_reversal_paper_v2.pdf` are not under development and must not be submitted. They still build against the shared `references.bib`, whose two bibliographic errors were corrected.
