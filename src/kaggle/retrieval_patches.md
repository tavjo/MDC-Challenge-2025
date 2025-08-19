### Implementation checklist for updating `retrieval_module.py`

- [x] Keep `hybrid_retrieve_with_boost` signature unchanged; use existing `prototypes: np.ndarray | None` and `id_to_dense` (no new params).
- [x] Remove any remaining use of `_rank_by_prototype_affinity` from the main flow.

- [x] Add helpers (if missing):
  - [x] `_minmax_norm(arr)` and `_clip01(x)` for score normalization.
  - [x] `_prototype_topk(proto_mat, chunk_mat, k)` to compute max-sim prototype scores and top indices.
  - [x] `_das_prior(text)` and `_regex_specific_score(text)` with caps per plan.
  - [x] `_doi_repo_prior(text)` using only text (no section headers); rely on repo-DOI regex + repo keywords + DAS language cues.
  - [x] Tiny, capped neighbor helper based on adjacency in the fused ranking (no section gating since section titles are unavailable).

- [x] Constants/config:
  - [x] Define `RETRIEVAL_WEIGHTS` exactly as in the plan.
  - [x] Define `NEIGHBOR_BOOST_CAP`, `REGEX_MAX_HITS_FOR_SCORE`.
  - [x] Compute `TOPK_PER_SIGNAL = boost_cfg.mmr_top_k * 3` dynamically inside the function.

- [x] Repository DOIs and specific regex:
  - [x] Add `REGEX_SPECIFIC` and a compiled map used by `_regex_specific_score`.
  - [x] Add `REGEX_REPO_DOI` and compiled map used by `_doi_repo_prior`.
  - [x] Remove generic DOI patterns entirely from `ACCESSION_REGEXES` (delete `doi_url`, `doi_numeric` entries).

- [x] Prototype-first candidate generation (primary when prototypes provided):
  - [x] If `prototypes` is provided and non-empty:
    - [x] Build chunk matrix in id order from `id_to_dense`.
    - [x] Call `_prototype_topk(prototypes, chunk_mat, k=TOPK_PER_SIGNAL)`.
    - [x] Normalize per-chunk prototype scores with `_minmax_norm`.
    - [x] Record prototype candidates and a `cid -> proto_score` map.

- [x] Secondary retrieval (RRF):
  - [x] Compute sparse top-k: `_sparse_topk(query_text, id_to_text, k=TOPK_PER_SIGNAL)`.
  - [x] Compute dense top-k with provided `dense_query_vec` (do not add centroid fallback param).
  - [x] Fuse via existing `reciprocal_rank_fusion`, normalize RRF scores over its candidate pool.

- [x] Candidate union and features:
  - [x] Union prototype candidates (if any) with RRF candidates (preserve order with dict-from-keys or similar).
  - [x] For each candidate, assemble:
    - [x] `score_proto` (0 if no prototypes).
    - [x] `score_rrf` (normalized).
    - [x] `das_prior`, `regex_specific`, `doi_repo_prior`.
    - [x] `neighbor` tiny capped boost from adjacency in the fused RRF ranking.

- [x] Final scoring and MMR:
  - [x] Compute final score = weighted blend per `RETRIEVAL_WEIGHTS`:
        prototype + rrf + das + regex + doi_repo + neighbor − ref_penalty
        Note: no section headers available → set `ref_penalty` = 0.
  - [x] Sort by final score descending; keep only ids present in `id_to_dense`.
  - [x] Run existing `mmr_rerank` with `lambda_diversity=boost_cfg.mmr_lambda` and `top_k=boost_cfg.mmr_top_k`.

- [x] Cleanups and safeguards:
  - [x] Do not reference `id_to_section`; remove any section-based gating code.
  - [x] Do not use `id_to_neighbors` unless explicitly required by you later; use the plan’s adjacency-on-ranking approach.
  - [x] Ensure all new logic is gated and cannot throw if inputs are empty; default to zeros.

- [ ] Sanity logging (optional, behind a flag or temporary):
  - [ ] For the top-N, log: `score_proto`, `score_rrf`, `das_prior`, `regex_specific`, `doi_repo`, `neighbor`.

- [x] Tests/thought checks:
  - [x] Repo-DOI in text with DAS language → boosted.
  - [x] Random DOI removed (since generic DOIs are not kept).
  - [x] GSE/SRR/PXD presence → promoted via `regex_specific_score`.
  - [x] With prototypes provided, prototype candidates dominate pool formation.