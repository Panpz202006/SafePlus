# SAFEPlus: evolving IJCAI 2027 draft

Working title: **SAFEPlus: Separating Fraud Occurrence from Delayed Detection**.
Started November 7, 2026. This is a research note, not a completed submission.

## Edit and build

- Entry point: `main.tex`; substantive text lives in `sections/`.
- Bibliography: `references.bib`.
- From the repository root, run `make -C paper` (requires TeX Live/MacTeX and latexmk).
- Compiled deliverable: `build/SAFEPlus-IJCAI2027-draft.pdf`.
- Temporary compiler files are under ignored `build/`.
- For Overleaf, upload the contents of this directory and select `main.tex`.

The draft uses anonymous authors and review line numbers. Do not add personal
paths, identifying acknowledgments, or author metadata to the review PDF.
The supplied conceptual note is provisionally cited; its attribution and
publication status must be settled before submission. No source PDFs are copied
into the repository. Existing SAFE code is untouched.

## Template provenance

Official author kit: https://ijcai.org/authors_kit/

Downloaded November 7, 2026 from:
https://ijcai.org/sites/default/files/Guidelines/FormattingGuidelines-IJCAI-27.zip

`ijcai27.sty` and `named.bst` are unmodified official files.
`template/ijcai27-example.tex` and `template/ijcai27-example.bib` preserve the
official example sources. To compile the example from that subfolder, copy
the style and bibliography style there temporarily or configure TEXINPUTS.
The official Word file is omitted because this project uses LaTeX.

## Target and working milestones

Verified against the main-track call on November 7, 2026:
https://2027.ijcai.org/ijcai-2027-call-for-papers-main-track/

- Abstract: January 4, 2027, 23:59 AoE (January 5, 19:59 Asia/Shanghai).
- Full paper: January 11, 2027, 23:59 AoE (January 12, 19:59 Asia/Shanghai).
- Limit: 7 body pages plus up to 2 reference pages; 9 total.
- Anonymous review; recheck the live call before submission.

Internal targets, not conference deadlines:

1. November 7--15: refine assumptions, likelihood, and relevant literature.
2. November 16--30: verify discrete likelihood and synthetic recovery;
   establish the SAFE baseline and audit data semantics.
3. December 1--15: controlled comparisons and delay/supervision ablations.
4. December 16--27: complete experiments and rewrite around supported claims.
5. December 28--January 3: review novelty, attribution, limitations, and abstract.
6. January 4--10: final validation, anonymization, formatting, and submission preparation.

These targets should change as evidence arrives. Submission is not scheduled
or automated by creating this draft.

## Next increment

Implement the finite-grid likelihood and compare exact enumeration with the
linear-time recurrence. Then test the exponential rate-swap ambiguity and
recovery with known delay or a small audited onset subset. Keep observed
detection indicators distinct from latent fraud indicators.

## Claims and source ledger

- SAFE architecture/objective: supplied SAFE paper and verified AAAI DOI
  `10.1609/aaai.v33i01.33011278`.
- Separate onset/detection processes and latent inference: supplied scan titled
  *Survival Analysis for Fraud Detection*, Alex Smola, April 28, 2022.
- Finite-grid formulation and rate-swap argument: derivations in this working
  draft; require independent review and broader literature positioning.
- Empirical results: none. Do not convert planned experiments into past-tense claims.
- A broader review of delayed feedback and multi-state/latent survival models
  is outstanding. No novelty claim has been established.
