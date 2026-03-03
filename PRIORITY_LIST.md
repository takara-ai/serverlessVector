# Vector DB Gaps: Priority List (Serverless / Lambda)

Tuned for **Lambda-style environments**: 1–2 vCPUs, constrained memory, single-request latency and cold start matter; parallelism has limited or negative payoff.

**Design: exact nearest neighbour only.** Total vectors in storage is intended to be low; we do not implement and do not plan ANN (approximate NN).

---

## P0 – Critical (Lambda-friendly) [DONE]

| # | Gap | Status |
|---|-----|--------|
| 1 | **Reduce search allocations** | Done: size-k heap for topK. |
| 2 | **Dense vector storage** | Done: `Vector.Data` is `[]float32` only. |
| 3 | **Expose metadata/tag filter in Search** | Done: `SearchWithFilter(query, topK, filter)`. |

---

## P1 – High (single-core and memory)

| # | Gap | Why (serverless) |
|---|-----|------------------|
| 4 | **Optional norm cache for cosine** | Reuse query norm; optionally cache per-vector norms. Fewer CPU cycles per search and no extra goroutines. Good for repeated searches in one invocation. |
| 5 | **MMR allocation and algo** | MMR now uses typed distance (much improved). Further buffer reuse possible if needed. |
| 6 | **Score threshold / min similarity** | Optional `minScore` so callers get "only results above X". Simple filter; no parallelism; keeps responses small and predictable. |

---

## P2 – High (functionality, serverless-shaped)

| # | Gap | Why (serverless) |
|---|-----|------------------|
| 7 | **Serialize/deserialize (export/import)** | In Lambda, state often lives in S3 or is loaded at init. Export/Import (binary or JSON) lets you load a snapshot from S3 on cold start or save after BatchAdd. More important than "full" persistence. |
| 8 | **List / enumerate IDs** | `IDs()` or `List(offset, limit)` for debugging and admin. Cheap; no extra threads. |
| 9 | **BatchGet / BatchDelete** | Fewer round-trips and lock cycles when loading or pruning many by ID. Stays single-threaded. |

---

## P3 – Medium (polish)

| # | Gap | Why (serverless) |
|---|-----|------------------|
| 10 | **Exists(id)** | `Exists(id) bool` under RLock; avoids full Get when you only need presence. Small, single-core friendly. |
| 11 | **Validation** | Use or remove `ValidationResult`/`ValidationError`; optional NaN/Inf rejection in Add. Keeps behavior predictable. |

---

## Deprioritized or deferred for Lambda

| Item | Change | Reason |
|------|--------|--------|
| **BatchSearch parallelism** | Demote / don't do by default | On 1 vCPU, extra goroutines add scheduling and memory cost with little or no speedup. Keep BatchSearch sequential unless you explicitly target 2+ vCPU and many queries per invocation. |
| **ANN** | Never | Exact NN only; total vectors in storage is low. No approximate indexes. |
| **SIMD** | Optional later | Dense storage done; SIMD could further speed distance kernels if profiling justifies it. |

---

## Suggested order of work (serverless)

1. ~~**P0#1–#3**~~ Done.
2. **P1#4** – Norm cache for cosine (query norm; per-vector if memory allows).
3. **P1#5** – MMR: further buffer reuse if needed.
4. **P1#6** – Score threshold in search.
5. **P2#7** – Export/Import for load from S3 (or similar) at init / between invocations.
6. **P2#8, #9** – List IDs, BatchGet, BatchDelete as needed.
7. **P3** – Exists, validation.
8. **SIMD** – Only if profiling justifies it.

---

## Reference (benchmarks, after P0)

- Search (1k vectors, 128D, topK=10): ~0.5ms, 4 KB/op, 54 allocs/op.
- SearchMMR (same): ~1.4ms, 33 KB/op, 268 allocs/op.
- BatchAdd (500 vectors, 128D): ~0.16ms, 351 KB/op, 1008 allocs/op.
