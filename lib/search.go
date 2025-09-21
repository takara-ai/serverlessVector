package lib

import (
	"errors"
	"fmt"
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

// searchCore is the shared backend implementation
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
		// Apply filter if provided
		if filterFunc != nil && !filterFunc(vector) {
			continue
		}

		if len(vector.Data) != len(querySlice) {
			return nil, fmt.Errorf("query vector dimension %d does not match stored vector dimension %d", len(querySlice), len(vector.Data))
		}

		// Skip if types don't match
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

	// Sort by score (higher is better for similarity, lower for distance)
	sort.Slice(results, func(i, j int) bool {
		switch db.distFunc {
		case EuclideanDistance, ManhattanDistance:
			return results[i].Score < results[j].Score // Lower distance is better
		default:
			return results[i].Score > results[j].Score // Higher similarity is better
		}
	})

	// Return top K results
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