# Final submission checklist (IEEE QAI)

## Done on branch `migration/qai-submission-20261106`
- [x] Repository at handoff commit `304144c` verified; working tree clean; migration branch created
- [x] Primary result hash `23f47f27…` verified; all registry metric hashes verified
- [x] All handoff numbers re-derived from JSON (`scripts/verify_claims.py`)
- [x] All manuscript decimals cross-checked (`scripts/check_manuscript_numbers.py`)
- [x] Frozen datasets regenerated, with hashes identical (`scripts/check_dataset_hashes.py`)
- [x] Seed-111 DAG and grid re-run bitwise identical (new directory and registry entry)
- [x] 31/31 tests pass
- [x] Manuscript moved to IEEEtran V1.8b conference class, A4, anonymous, 5 pages including references
- [x] All fonts embedded; no Type 3 fonts; empty PDF author metadata
- [x] Figures regenerated from scripts (`scripts/make_figures.py`), colorblind-safe, vector
- [x] Formal invariance proposition and proof added
- [x] Evidence tiers separated; Transformer labeled post-confirmatory everywhere
- [x] Negative results and new limitations disclosed
- [x] Citations verified; two bibliographic errors corrected
- [x] Original draft preserved under `paper/archive/`

## Owner decisions and inputs (cannot be done without you)
- [ ] **Choose the target:** QAI 2026 late-breaking (deadline 6 Nov 2026, non-archival, CMT) **or** the next full-paper call (QAI 2027; not yet announced)
- [ ] Decide whether to approve and run `experiments/DRAFT_QAI_SERIALIZATION_CONTROL_PROTOCOL_20261106.md` before a full-paper submission (recommended; about 30 CPU-min)
- [ ] Confirm or amend the generative-AI disclosure in the Acknowledgment
- [ ] Fill in the fields in "Unresolved fields" below at camera-ready (not before; the review is double-blind)
- [ ] Decide on license and citation metadata for the code release (no LICENSE file exists)
- [ ] Build an anonymous artifact (`IDENTIFIERS='…' bash scripts/make_anonymous_artifact.sh`) and host it anonymously, if links are allowed
- [ ] Check CMT conflict-of-interest domains and any CMT-specific forms
- [ ] Re-verify page limit, template, and dates when the QAI 2027 call is published
- [ ] At camera-ready: de-anonymize, add the IEEE copyright notice, run IEEE PDF eXpress if required

## Unresolved author and venue fields
| Field | Status |
|---|---|
| Author names, order | Not supplied |
| Affiliations, addresses | Not supplied |
| Author e-mails, ORCID iDs | Not supplied |
| Corresponding author | Not supplied |
| Funding acknowledgments | Not supplied |
| Conflicts of interest | Not supplied |
| Generative-AI disclosure (final wording) | Drafted; needs author confirmation |
| Track (late-breaking vs. full paper) | Owner decision |
| Artifact hosting URL (anonymous) | Not created |
| Code license | Not chosen |
| Copyright form, registration, presenter | Camera-ready stage |
| QAI 2027 deadline, page limit, template | Not yet published |
| Supplementary-material policy | Not stated by the venue |
