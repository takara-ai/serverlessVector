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

func (h resultHeap) Len() int { return len(h.results) }
func (h resultHeap) Swap(i, j int) { h.results[i], h.results[j] = h.results[j], h.results[i] }
func (h resultHeap) Less(i, j int) bool {
	if h.lowerIsBetter {
		return h.results[i].Score > h.results[j].Score // max at root for distance
	}
	return h.results[i].Score < h.results[j].Score // min at root for similarity
}
func (h *resultHeap) Push(x interface{}) { h.results = append(h.results, x.(SimilarityResult)) }
func (h *resultHeap) Pop() interface{} {
	n := len(h.results)
	item := h.results[n-1]
	h.results = h.results[:n-1]
	return item
}

// Search performs fast similarity search
// Returns top 10 results by default, or specify topK
func (db *VectorDB) Search(query interface{}, topK ...int) (*SearchResult, error) {
	k := 10 // smart default
	if len(topK) > 0 {
		k = topK[0]
	}
	return db.searchCore(query, k, true, nil)
}

// SearchWithFilter performs similarity search with a filter on vectors (e.g. by metadata/tags).
// filter is called for each vector; only vectors for which filter returns true are considered.
func (db *VectorDB) SearchWithFilter(query interface{}, topK int, filter func(*Vector) bool) (*SearchResult, error) {
	if topK <= 0 {
		topK = 10
	}
	return db.searchCore(query, topK, true, filter)
}

// BatchSearch performs search on multiple queries efficiently
func (db *VectorDB) BatchSearch(queries map[string]interface{}, topK ...int) (map[string]*SearchResult, error) {
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
func (db *VectorDB) SearchMMR(query interface{}, topK int, opts ...*MMROptions) (*SearchResult, error) {
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
func (db *VectorDB) SearchMMRParams(query interface{}, topK int, lambda float64, fetchFactor ...int) (*SearchResult, error) {
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

func (db *VectorDB) searchMMRCore(query interface{}, topK int, lambda float64, ff int) (*SearchResult, error) {
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
	if len(candidates.Results) <= topK {
		return candidates, nil
	}

	candVecs := make(map[string]*Vector, len(candidates.Results))
	db.mu.RLock()
	for _, r := range candidates.Results {
		v, ok := db.vectors[r.ID]
		if ok {
			candVecs[r.ID] = v
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

	relevanceToQuery := make(map[string]float64)
	for _, r := range candidates.Results {
		relevanceToQuery[r.ID] = toRelevance(r.Score)
	}

	selected := make([]SimilarityResult, 0, topK)
	remaining := make(map[string]SimilarityResult)
	for _, r := range candidates.Results {
		remaining[r.ID] = r
	}

	for len(selected) < topK && len(remaining) > 0 {
		var bestID string
		bestMMR := math.Inf(-1)
		for id := range remaining {
			relQ := relevanceToQuery[id]
			maxSimToSelected := 0.0
			vecD := candVecs[id]
			for _, s := range selected {
				vecS := candVecs[s.ID]
				raw := db.distanceFloat32(vecD.Data, vecS.Data, db.distFunc)
				sim := toRelevance(raw)
				if sim > maxSimToSelected {
					maxSimToSelected = sim
				}
			}
			mmrScore := lambda*relQ - (1.0-lambda)*maxSimToSelected
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

// searchCore is the shared backend implementation.
func (db *VectorDB) searchCore(query interface{}, topK int, includeMetadata bool, filterFunc func(*Vector) bool) (*SearchResult, error) {
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