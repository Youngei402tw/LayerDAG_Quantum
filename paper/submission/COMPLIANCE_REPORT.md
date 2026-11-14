# Official-template and venue compliance report: IEEE QAI

Prepared 2026-11-06 on branch `migration/qai-submission-20261106`. All URLs were accessed on 2026-11-06.

## 1. What the official sources say

| Item | Verified rule | Source |
|---|---|---|
| Active edition | 2nd IEEE International Conference on Quantum Artificial Intelligence (QAI 2026) | https://attend.ieee.org/qai-2026/ |
| Dates and location | 13–16 December 2026, Jubilee Conference Centre, Jubilee Campus, Nottingham, UK | https://www.nottingham.ac.uk/computerscience/events/the-second-ieee-international-conference-on-quantum-artificial-intelligence-qai-come-to-nottingham.aspx |
| Full-paper deadline | "Extended to 15 July 2026". **Closed** ("Full paper submission deadline is now closed") | https://attend.ieee.org/qai-2026/ ; https://attend.ieee.org/qai-2026/call-for-papers/ |
| Notification / final paper | 15 Sep 2026 / "Postponed to 15 October 2026" | https://attend.ieee.org/qai-2026/ |
| Late-breaking papers | Deadline **6 Nov 2026**, notification 10 Nov 2026; "up to 6 pages"; "a brief review by the conference Chairs for evaluating their fit"; **not part of the conference proceedings**; submit via CMT (`cmt3.research.microsoft.com/QAI2026/`) | https://attend.ieee.org/qai-2026/late-breaking/ |
| Page limit (full papers) | "Each paper is limited to 6 pages … including figures, tables, references, and appendices"; up to 2 extra pages at US$150 each (max 8) | https://attend.ieee.org/qai-2026/authors-instructions/ |
| Template | "IEEE style files for conference proceedings" (https://www.ieee.org/conferences/publishing/templates.html); two-column, single-spaced, 10-pt Times; A4; no page numbers | https://attend.ieee.org/qai-2026/authors-instructions/ |
| Review model | "The review process for QAI 2026 will be double-blind"; no author names in the PDF; cite own prior work in the third person; preprints allowed; author line `\IEEEauthorblockN{Anonymous Authors}` | https://attend.ieee.org/qai-2026/authors-instructions/ |
| Platform | Microsoft CMT ("IEEE QAI 2026 online submission system") | https://attend.ieee.org/qai-2026/authors-instructions/ |
| AI-generated content | Use "shall be disclosed in the acknowledgements section"; sections that use AI-generated text need a citation to the AI system | https://attend.ieee.org/qai-2026/authors-instructions/ |
| Originality | Must be original and not under review elsewhere; plagiarism check | https://attend.ieee.org/qai-2026/authors-instructions/ ; call for papers |
| Publication | Accepted and presented papers go to IEEE Xplore, indexed by Scopus | https://attend.ieee.org/qai-2026/call-for-papers/ |
| Supplementary material / artifacts | **Not stated.** The 2025 call mentions IEEE DataPort for data | https://computer.org/conferences/cfp-qai |
| Tracks | No formal tracks; topic list only. Separate calls exist for late-breaking, J2C, doctoral colloquium, special sessions | https://attend.ieee.org/qai-2026/call-for-papers/ |
| QAI 2027 | **No call found** on 2026-11-06 | web search; https://attend.ieee.org/qai-2026/ |

Inconsistencies in the official pages:

- The late-breaking page heading reads "IEEE QAI 2025" while the page sits under the 2026 site.
- The author instructions say both "4 to maximum 6 pages" and "limited to 6 pages".
- An EasyChair CFP mirror (https://easychair.org/cfp/QAI2026) lists 30 June 2026 as the deadline, an earlier value that the official site has since superseded.
- The "Important Dates" sub-page (`/author-instructions-2/`) failed to load.

Confirm any of these with the chairs (qai2026@ntu.ac.uk per the EasyChair mirror) before relying on it.

## 2. Compliance of `paper/qai_paper.pdf`

| Requirement | Status |
|---|---|
| IEEE conference template | Yes. `IEEEtran.cls` V1.8b (2015/08/26), `[conference,a4paper]`, which is the class distributed on IEEE's template page |
| A4 | Yes, 595.3 × 841.9 pt |
| ≤ 6 pages including references | Yes. 6 pages as of v5 (2026-11-14); 5 pages in the 2026-11-06 build. There is no slack for further additions without cuts or paid extra pages |
| Double-blind author block | Yes, "Anonymous Authors". PDF Author/Title metadata empty. No self-identifying text |
| No page numbers | Yes (IEEEtran conference mode) |
| Fonts embedded, no Type 3 | Yes. `pdffonts` shows 25 fonts in v5, all embedded, none Type 3 (the handoff PDF had one Type 3 bitmap font) |
| AI-use disclosure in acknowledgements | Present without placeholder since v3; v5 wording also covers the controls and the suite. **The authors should still confirm it matches their actual use before submitting** |
| IEEE PDF eXpress | Not mentioned for submission. Expected only at camera-ready; not run here |
| Copyright notice | Not required at submission; insert the IEEE copyright line at camera-ready |

## 3. Unresolved venue questions

- Whether the owner wants a **late-breaking** submission today (non-archival, and a fit check only) or to wait for the next full-paper call.
- QAI 2027 dates, page limit, and template version. Re-verify when that call appears.
- Supplementary-material and artifact-link policy. Not stated, so do not include a non-anonymous link.
- Reference-limit and appendix rules for late-breaking papers beyond "up to 6 pages".
