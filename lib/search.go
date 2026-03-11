package lib

import (
	"container/heap"
	"errors"
	"fmt"
	"math"
	"sort"
)

// resultHeap keeps the top K results by score. For similarity (higher better), root is min score;
// for distance (lower better), root is max score.
type resultHeap struct {
	results       []SimilarityResult
	lowerIsBetter bool
}

func (h resultHeap) Len() int      { return len(h.results) }
func (h resultHeap) Swap(i, j int) { h.results[i], h.results[j] = h.results[j], h.results[i] }
func (h resultHeap) Less(i, j int) bool {
	if h.lowerIsBetter {
		return h.results[i].Score > h.results[j].Score // max at root for distance
	}
	return h.results[i].Score < h.results[j].Score // min at root for similarity
}
func (h *resultHeap) Push(x any) { h.results = append(h.results, x.(SimilarityResult)) }
func (h *resultHeap) Pop() any {
	n := len(h.results)
	item := h.results[n-1]
	h.results = h.results[:n-1]
	return item
}

// Search performs fast similarity search
// Returns top 10 results by default, or specify topK
func (db *VectorDB) Search(query any, topK ...int) (*SearchResult, error) {
	k := 10 // smart default
	if len(topK) > 0 {
		k = topK[0]
	}
	return db.searchCore(query, k, true, nil)
}

// SearchWithFilter performs similarity search with a filter on vectors (e.g. by metadata/tags).
// filter is called for each vector; only vectors for which filter returns true are considered.
func (db *VectorDB) SearchWithFilter(query any, topK int, filter func(*Vector) bool) (*SearchResult, error) {
	if topK <= 0 {
		topK = 10
	}
	return db.searchCore(query, topK, true, filter)
}

// BatchSearch performs search on multiple queries efficiently
func (db *VectorDB) BatchSearch(queries map[string]any, topK ...int) (map[string]*SearchResult, error) {
	k := 10 // smart default
	if len(topK) > 0 {
		k = topK[0]
	}

	results := make(map[string]*SearchResult)
	for queryID, query := range queries {
		result, err := db.searchCore(query, k, true, nil)
		if err != nil {
			return nil, fmt.Errorf("search failed for query %s: %v", queryID, err)
		}
		result.QueryID = queryID
		results[queryID] = result
	}
	return results, nil
}

// SearchMMR performs Maximal Marginal Relevance search. Call with (query, topK) for defaults;
// pass optional *MMROptions to tune. Results are relevant to the query but diverse from each other.
func (db *VectorDB) SearchMMR(query any, topK int, opts ...*MMROptions) (*SearchResult, error) {
	if topK <= 0 {
		topK = 10
	}
	lambda := 0.6
	ff := 5
	if len(opts) > 0 && opts[0] != nil {
		if opts[0].Lambda > 0 {
			lambda = math.Max(0, math.Min(1, opts[0].Lambda))
		}
		if opts[0].FetchFactor > 0 {
			ff = opts[0].FetchFactor
		}
	}
	return db.searchMMRCore(query, topK, lambda, ff)
}

// SearchMMRParams is the explicit-parameter form of MMR (lambda and optional fetchFactor).
// For the simpler API use SearchMMR(query, topK) or SearchMMR(query, topK, &MMROptions{...}).
func (db *VectorDB) SearchMMRParams(query any, topK int, lambda float64, fetchFactor ...int) (*SearchResult, error) {
	if topK <= 0 {
		topK = 10
	}
	ff := 5
	if len(fetchFactor) > 0 && fetchFactor[0] > 0 {
		ff = fetchFactor[0]
	}
	lambda = math.Max(0, math.Min(1, lambda))
	return db.searchMMRCore(query, topK, lambda, ff)
}

// SelectMMRFromCandidates runs MMR over a provided candidate set.
// It uses db only for its DistanceFunction. FetchFactor is ignored.
// Callers must pass normalized embeddings if using CosineSimilarity.
func (db *VectorDB) SelectMMRFromCandidates(candidates []MMRCandidate, topK int, opts *MMROptions) (*SearchResult, error) {
	if topK <= 0 {
		topK = 10
	}
	if len(candidates) == 0 {
		return &SearchResult{Results: []SimilarityResult{}, Total: 0}, nil
	}
	lambda := 0.6
	if opts != nil && opts.Lambda > 0 {
		lambda = math.Max(0, math.Min(1, opts.Lambda))
	}
	return mmrGreedyCandidates(candidates, topK, lambda, db.distFunc)
}

// SearchMMRWithScores runs MMR with custom relevance scoring (QueryOnly, BaseScoreOnly, or Blend).
// When baseScores are provided, they can override or blend with query similarity.
func (db *VectorDB) SearchMMRWithScores(query any, topK int, baseScores map[string]float64, opts *MMROptions) (*SearchResult, error) {
	if topK <= 0 {
		topK = 10
	}
	lambda := 0.6
	ff := 5
	var scoreMode MMRScoreMode
	var blendAlpha float64

	if opts != nil {
		if opts.Lambda > 0 {
			lambda = math.Max(0, math.Min(1, opts.Lambda))
		}
		if opts.FetchFactor > 0 {
			ff = opts.FetchFactor
		}
		scoreMode = opts.ScoreMode
		blendAlpha = opts.BlendAlpha
	}

	// 1. Get candidates via standard search
	candidateK := topK * ff
	candidates, err := db.searchCore(query, candidateK, true, nil)
	if err != nil {
		return nil, err
	}
	if len(candidates.Results) == 0 {
		return &SearchResult{Results: []SimilarityResult{}, Total: 0}, nil
	}

	// 2. Resolve candidate vectors
	candVecs := make(map[string][]float32, len(candidates.Results))
	db.mu.RLock()
	for _, r := range candidates.Results {
		v, ok := db.vectors[r.ID]
		if ok {
			candVecs[r.ID] = v.Data
		}
	}
	db.mu.RUnlock()
	if len(candVecs) != len(candidates.Results) {
		return nil, errors.New("MMR: could not resolve all candidate vectors")
	}

	// 3. Compute relevance based on scoreMode
	toRelevance := func(score float64) float64 {
		switch db.distFunc {
		case EuclideanDistance, ManhattanDistance:
			return 1.0 / (1.0 + score)
		default:
			return score
		}
	}

	relevance := make(map[string]float64, len(candidates.Results))
	candResults := make(map[string]SimilarityResult, len(candidates.Results))

	for _, r := range candidates.Results {
		candResults[r.ID] = r
		queryRel := toRelevance(r.Score)
		baseScore := 0.0
		if baseScores != nil {
			baseScore = baseScores[r.ID] // missing treats as 0
		}

		var finalRel float64
		switch scoreMode {
		case MMRScoreQueryOnly:
			finalRel = queryRel
		case MMRScoreBaseOnly:
			finalRel = baseScore
		case MMRScoreBlend:
			finalRel = blendAlpha*queryRel + (1.0-blendAlpha)*baseScore
		default:
			finalRel = queryRel
		}
		relevance[r.ID] = finalRel
	}

	return mmrGreedy(candResults, candVecs, relevance, topK, lambda, db.distFunc)
}

func (db *VectorDB) searchMMRCore(query any, topK int, lambda float64, ff int) (*SearchResult, error) {
	// Re-implement searchMMRCore using SearchMMRWithScores logic (QueryOnly mode)
	// or directly call mmrGreedy to share code.

	if topK <= 0 {
		topK = 10
	}

	candidateK := topK * ff
	candidates, err := db.searchCore(query, candidateK, true, nil)
	if err != nil {
		return nil, err
	}
	if len(candidates.Results) == 0 {
		return &SearchResult{Results: []SimilarityResult{}, Total: 0}, nil
	}
	// Note: Original searchMMRCore had a check `if len(candidates.Results) <= topK { return candidates, nil }`
	// but strictly speaking, even if we have fewer candidates, MMR re-ranking might change order if lambda < 1?
	// Actually if K <= topK, we select all of them. The order might change though.
	// The original code returned early, implying order from searchCore (similarity) is accepted if we don't have enough candidates to pick from?
	// But MMR is about diversity. If I have 3 results and topK=10, do I want them re-ordered by diversity?
	// Original code returned candidates directly. I will preserve this behavior.
	if len(candidates.Results) <= topK {
		return candidates, nil
	}

	candVecs := make(map[string][]float32, len(candidates.Results))
	db.mu.RLock()
	for _, r := range candidates.Results {
		v, ok := db.vectors[r.ID]
		if ok {
			candVecs[r.ID] = v.Data
		}
	}
	db.mu.RUnlock()
	if len(candVecs) != len(candidates.Results) {
		return nil, errors.New("MMR: could not resolve all candidate vectors")
	}

	toRelevance := func(score float64) float64 {
		switch db.distFunc {
		case EuclideanDistance, ManhattanDistance:
			return 1.0 / (1.0 + score)
		default:
			return score
		}
	}

	relevance := make(map[string]float64, len(candidates.Results))
	candResults := make(map[string]SimilarityResult, len(candidates.Results))
	for _, r := range candidates.Results {
		candResults[r.ID] = r
		relevance[r.ID] = toRelevance(r.Score)
	}

	return mmrGreedy(candResults, candVecs, relevance, topK, lambda, db.distFunc)
}

// mmrGreedy implements the shared greedy selection loop.
func mmrGreedy(
	candidates map[string]SimilarityResult,
	vectors map[string][]float32,
	relevance map[string]float64,
	topK int,
	lambda float64,
	distFunc DistanceFunction,
) (*SearchResult, error) {
	toRelevance := func(score float64) float64 {
		switch distFunc {
		case EuclideanDistance, ManhattanDistance:
			return 1.0 / (1.0 + score)
		default:
			return score
		}
	}

	selected := make([]SimilarityResult, 0, topK)
	remaining := make(map[string]SimilarityResult, len(candidates))
	for id, r := range candidates {
		remaining[id] = r
	}

	// For loop
	for len(selected) < topK && len(remaining) > 0 {
		var bestID string
		bestMMR := math.Inf(-1)

		for id := range remaining {
			rel := relevance[id]
			maxSimToSelected := 0.0
			vecD := vectors[id]

			for _, s := range selected {
				vecS := vectors[s.ID]
				// Use package-level DistanceFloat32 to avoid DB dependency
				raw := DistanceFloat32(vecD, vecS, distFunc)
				sim := toRelevance(raw)
				if sim > maxSimToSelected {
					maxSimToSelected = sim
				}
			}

			mmrScore := lambda*rel - (1.0-lambda)*maxSimToSelected
			if mmrScore > bestMMR {
				bestMMR = mmrScore
				bestID = id
			}
		}

		if bestID == "" {
			break
		}
		selected = append(selected, remaining[bestID])
		delete(remaining, bestID)
	}

	return &SearchResult{
		Results: selected,
		Total:   len(selected),
	}, nil
}

// mmrGreedyCandidates is an index-driven MMR path for caller-provided candidates.
// It avoids string-keyed maps in the hot loop to reduce allocations and lookup cost.
func mmrGreedyCandidates(candidates []MMRCandidate, topK int, lambda float64, distFunc DistanceFunction) (*SearchResult, error) {
	n := len(candidates)
	if n == 0 {
		return &SearchResult{Results: []SimilarityResult{}, Total: 0}, nil
	}
	if topK > n {
		topK = n
	}

	results := make([]SimilarityResult, n)
	relevance := make([]float64, n)
	maxSimToSelected := make([]float64, n)
	remaining := make([]int, n)
	selected := make([]SimilarityResult, 0, topK)
	norms := make([]float64, 0)
	if distFunc == CosineSimilarity {
		norms = make([]float64, n)
	}

	for i, c := range candidates {
		remaining[i] = i
		rel := c.BaseScore
		if math.IsNaN(rel) || rel < 0 {
			rel = 0
		}
		relevance[i] = rel
		if distFunc == CosineSimilarity {
			norms[i] = norm32(c.Embedding)
		}
		results[i] = SimilarityResult{
			ID:    c.ID,
			Score: c.BaseScore, // Keep base score in result
		}
	}

	similarity := func(i, j int) float64 {
		switch distFunc {
		case CosineSimilarity:
			ni, nj := norms[i], norms[j]
			if ni == 0 || nj == 0 {
				return 0
			}
			return dotProduct32(candidates[i].Embedding, candidates[j].Embedding) / (ni * nj)
		case DotProduct:
			return dotProduct32(candidates[i].Embedding, candidates[j].Embedding)
		case EuclideanDistance:
			return 1.0 / (1.0 + euclidean32(candidates[i].Embedding, candidates[j].Embedding))
		case ManhattanDistance:
			return 1.0 / (1.0 + manhattan32(candidates[i].Embedding, candidates[j].Embedding))
		default:
			return dotProduct32(candidates[i].Embedding, candidates[j].Embedding)
		}
	}

	for len(selected) < topK && len(remaining) > 0 {
		bestPos := 0
		bestIdx := remaining[0]
		bestMMR := lambda*relevance[bestIdx] - (1.0-lambda)*maxSimToSelected[bestIdx]

		for pos := 1; pos < len(remaining); pos++ {
			idx := remaining[pos]
			mmrScore := lambda*relevance[idx] - (1.0-lambda)*maxSimToSelected[idx]
			if mmrScore > bestMMR {
				bestMMR = mmrScore
				bestIdx = idx
				bestPos = pos
			}
		}

		selected = append(selected, results[bestIdx])

		last := len(remaining) - 1
		remaining[bestPos] = remaining[last]
		remaining = remaining[:last]

		for _, idx := range remaining {
			sim := similarity(idx, bestIdx)
			if sim > maxSimToSelected[idx] {
				maxSimToSelected[idx] = sim
			}
		}
	}

	return &SearchResult{
		Results: selected,
		Total:   len(selected),
	}, nil
}

// searchCore is the shared backend implementation.
func (db *VectorDB) searchCore(query any, topK int, includeMetadata bool, filterFunc func(*Vector) bool) (*SearchResult, error) {
	query32, err := queryToFloat32(query)
	if err != nil {
		return nil, err
	}
	if len(query32) == 0 {
		return nil, errors.New("query vector cannot be empty")
	}
	if topK <= 0 {
		topK = 10 // Default
	}

	db.mu.RLock()
	defer db.mu.RUnlock()

	if len(db.vectors) == 0 {
		return &SearchResult{Results: []SimilarityResult{}}, nil
	}

	lowerIsBetter := db.distFunc == EuclideanDistance || db.distFunc == ManhattanDistance
	h := &resultHeap{
		results:       make([]SimilarityResult, 0, topK+1),
		lowerIsBetter: lowerIsBetter,
	}

	for _, vector := range db.vectors {
		if filterFunc != nil && !filterFunc(vector) {
			continue
		}
		if vector.Dimension != len(query32) {
			return nil, fmt.Errorf("query vector dimension %d does not match stored vector dimension %d", len(query32), vector.Dimension)
		}

		score := db.distanceFloat32(query32, vector.Data, db.distFunc)

		result := SimilarityResult{
			ID:    vector.ID,
			Score: score,
		}
		if includeMetadata {
			result.Metadata = vector.Metadata
		}

		if h.Len() < topK {
			heap.Push(h, result)
		} else {
			worst := h.results[0]
			replace := lowerIsBetter && score < worst.Score || !lowerIsBetter && score > worst.Score
			if replace {
				heap.Pop(h)
				heap.Push(h, result)
			}
		}
	}

	results := h.results
	sort.Slice(results, func(i, j int) bool {
		if lowerIsBetter {
			return results[i].Score < results[j].Score
		}
		return results[i].Score > results[j].Score
	})

	return &SearchResult{
		Results: results,
		Total:   len(results),
	}, nil
}
