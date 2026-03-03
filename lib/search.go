package lib

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

// Search performs fast similarity search
// Returns top 10 results by default, or specify topK
func (db *VectorDB) Search(query interface{}, topK ...int) (*SearchResult, error) {
	k := 10 // smart default
	if len(topK) > 0 {
		k = topK[0]
	}
	return db.searchCore(query, k, true, nil)
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
				raw := db.calculateSimilarity(vecD.Data, vecS.Data, vecD.Type, db.distFunc)
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
	querySlice, queryType, err := convertToInterfaceSlice(query)
	if err != nil {
		return nil, err
	}

	if len(querySlice) == 0 {
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

	results := make([]SimilarityResult, 0, len(db.vectors))

	for _, vector := range db.vectors {
		if filterFunc != nil && !filterFunc(vector) {
			continue
		}

		if len(vector.Data) != len(querySlice) {
			return nil, fmt.Errorf("query vector dimension %d does not match stored vector dimension %d", len(querySlice), len(vector.Data))
		}

		if vector.Type != queryType {
			continue
		}

		score := db.calculateSimilarity(querySlice, vector.Data, queryType, db.distFunc)

		result := SimilarityResult{
			ID:    vector.ID,
			Score: score,
		}

		if includeMetadata {
			result.Metadata = vector.Metadata
		}

		results = append(results, result)
	}

	sort.Slice(results, func(i, j int) bool {
		switch db.distFunc {
		case EuclideanDistance, ManhattanDistance:
			return results[i].Score < results[j].Score // Lower distance is better
		default:
			return results[i].Score > results[j].Score // Higher similarity is better
		}
	})

	if len(results) > topK {
		results = results[:topK]
	}

	return &SearchResult{
		Results: results,
		Total:   len(results),
	}, nil
}

// calculateSimilarity calculates similarity/distance between two vectors
func (db *VectorDB) calculateSimilarity(a, b []interface{}, vecType VectorType, distanceFunc DistanceFunction) float64 {
	switch distanceFunc {
	case CosineSimilarity:
		return db.cosineSimilarity(a, b, vecType)
	case DotProduct:
		return db.dotProduct(a, b, vecType)
	case EuclideanDistance:
		return db.euclideanDistance(a, b, vecType)
	case ManhattanDistance:
		return db.manhattanDistance(a, b, vecType)
	default:
		return db.dotProduct(a, b, vecType) // Default fallback
	}
}