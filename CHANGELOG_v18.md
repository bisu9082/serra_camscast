# v1.3 — submitted state (24 October 2026)

Front matter restored and completed for the *Environmetrics* submission, which is
single-anonymous: the manuscript carries the authors, not a separate anonymised body.

### `manuscript/main_env.tex`

- Author block restored on a single line with numbered affiliations, equal-contribution
  dagger and corresponding-author asterisk; affiliation block set `\small\singlespacing`
  so the restored front matter costs no page against the 40-page limit.
- `Data availability` section added: repository URL and release tag, the statement that
  raw CAMS EAC4 fields are not redistributed, and the pointer to
  `data_retrieval/RETRIEVAL.md` for the request parameters.
- `Conflict of interest` section added.
- `Use of AI technologies` section added, naming Claude (Anthropic) and the uses
  (drafting, language editing, analysis/simulation/figure code), and stating that no
  data, numerical results or references were AI-generated. The journal requires this
  disclosure in the manuscript; it previously existed only on the title page, which is
  no longer part of the upload.

### `manuscript/figlegends.tex`

- Leading `\clearpage` removed. It produced a near-empty page between the references and
  the first figure, which the 40-page limit could not spare.

### `manuscript/SI_serra.tex`

- Author names restored on the supplement cover page, matching the manuscript rather
  than deferring to a title page that is no longer uploaded.
- Reproducibility note now names the repository and release directly instead of pointing
  at a data availability statement "on the title page".

### Not changed

`body.tex`, `refs_serra.bib`, the simulation code, the aggregation scripts and the figure
scripts are byte-identical to `v1.2`. The temporal-inference-unit experiment and the
covariance-sensitivity experiment were already committed there.
