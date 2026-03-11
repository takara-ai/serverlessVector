# Changelog

## [2.1.0](https://github.com/takara-ai/serverlessVector/compare/v2.0.0...v2.1.0) (2026-03-11)


### Features

* **mmr:** add SelectMMRFromCandidates and SearchMMRWithScores with BaseScore/Query/Blend modes ([074cd2f](https://github.com/takara-ai/serverlessVector/commit/074cd2fa5a7006824ac206c1e573dcb646a29e76))

## [2.0.0](https://github.com/takara-ai/serverlessVector/compare/v1.0.0...v2.0.0) (2026-03-03)


### ⚠ BREAKING CHANGES

* Vectors are float32 only. Add/Update/BatchAdd and Search accept []float32; []float64 is rejected. Vector.Data is []float32; Vector.Type removed. Float64 constant removed from public API.

### Features

* **search:** add MMR search with SearchMMR API and MMROptions ([c66cb57](https://github.com/takara-ai/serverlessVector/commit/c66cb57d67b33e48063356e533b2397670553199))


### Performance Improvements

* float32-only storage, heap topK, typed distance; add SearchWithFilter ([e740842](https://github.com/takara-ai/serverlessVector/commit/e7408427f464771452d66bc4b1bd6c7154ce09de))
* **lib:** shorten lock hold time in BatchAdd and GetStats ([0ebb0ef](https://github.com/takara-ai/serverlessVector/commit/0ebb0effb79905e6b103be31db52665532529921))

## 1.0.0 (2025-09-21)


### Features

* add demos ([9c49fd3](https://github.com/takara-ai/serverlessVector/commit/9c49fd3e989f442af1dc4edd299588413c5a7e2a))
* init commit ([94be0d9](https://github.com/takara-ai/serverlessVector/commit/94be0d91ab68d444cd0ff01a185a533f66d663df))


### Bug Fixes

* make this true ([f005383](https://github.com/takara-ai/serverlessVector/commit/f005383890fa657b2426f70a86744d623396ef1c))
* url ([251a57d](https://github.com/takara-ai/serverlessVector/commit/251a57d16cec1a60909f9322527ff6a77759bffd))
