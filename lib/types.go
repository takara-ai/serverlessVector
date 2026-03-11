package lib

import (
	"fmt"
	"math"
)

// copyFloat32Slice copies []float32 and returns (copy, dimension, error). Rejects non-float32.
func copyFloat32Slice(data any) ([]float32, int, error) {
	v, ok := data.([]float32)
	if !ok {
		return nil, 0, fmt.Errorf("unsupported vector type: %T (use []float32)", data)
	}
	if len(v) == 0 {
		return nil, 0, nil
	}
	c := make([]float32, len(v))
	copy(c, v)
	return c, len(c), nil
}

// queryToFloat32 validates and returns the query as []float32.
func queryToFloat32(query any) ([]float32, error) {
	v, ok := query.([]float32)
	if !ok {
		return nil, fmt.Errorf("unsupported query type: %T (use []float32)", query)
	}
	return v, nil
}

// VectorType is the scalar type for vector storage. Only Float32 is supported.
type VectorType int

const (
	Float32 VectorType = iota
)

// DistanceFunction represents different distance/similarity metrics
type DistanceFunction int

const (
	CosineSimilarity DistanceFunction = iota
	DotProduct
	EuclideanDistance
	ManhattanDistance
)

// VectorMetadata holds additional information about vectors
type VectorMetadata struct {
	CreatedAt int64             `json:"created_at,omitempty"`
	UpdatedAt int64             `json:"updated_at,omitempty"`
	Tags      map[string]string `json:"tags,omitempty"`
	Score     float64           `json:"score,omitempty"` // Internal use
}

// ValidationResult holds the result of vector validation
type ValidationResult struct {
	IsValid bool
	Errors  []ValidationError
}

// ValidationError represents a specific validation error
type ValidationError struct {
	Field   string
	Value   any
	Message string
}

// Vector represents a vector with metadata. Data is float32 only (matches common embedding APIs).
type Vector struct {
	ID        string
	Data      []float32
	Metadata  VectorMetadata
	Dimension int
}

// SimilarityResult holds the result of a similarity search
type SimilarityResult struct {
	ID       string
	Score    float64
	Metadata VectorMetadata
}

// SearchResult contains the search results with scores
type SearchResult struct {
	QueryID string
	Results []SimilarityResult
	Total   int
}

// MMROptions configures Maximal Marginal Relevance search. Nil or zero values use defaults.
type MMROptions struct {
	Lambda      float64      // Balance relevance (1) vs diversity (0). Default 0.6.
	FetchFactor int          // Candidate pool size = FetchFactor * topK. Default 5.
	ScoreMode   MMRScoreMode // Mode for relevance scoring (QueryOnly, BaseScoreOnly, or Blend).
	BlendAlpha  float64      // Alpha for Blend mode: relevance = Alpha*querySim + (1-Alpha)*baseScore.
}

// MMRScoreMode defines how relevance is computed in MMR.
type MMRScoreMode int

const (
	// MMRScoreQueryOnly uses query similarity as relevance (default behavior).
	MMRScoreQueryOnly MMRScoreMode = iota
	// MMRScoreBaseOnly uses caller-provided BaseScore as relevance (ignoring query similarity).
	MMRScoreBaseOnly
	// MMRScoreBlend uses a weighted blend of query similarity and BaseScore.
	MMRScoreBlend
)

// MMRCandidate represents a candidate for MMR selection provided by the caller.
type MMRCandidate struct {
	ID        string
	Embedding []float32
	BaseScore float64 // External relevance score. Negative/NaN treated as 0.
}

// String returns a string representation of the distance function
func (df DistanceFunction) String() string {
	switch df {
	case CosineSimilarity:
		return "cosine_similarity"
	case DotProduct:
		return "dot_product"
	case EuclideanDistance:
		return "euclidean_distance"
	case ManhattanDistance:
		return "manhattan_distance"
	default:
		return "unknown"
	}
}

// NormalizeVector normalizes a float32 vector to unit length.
func NormalizeVector(data []float32) []float32 {
	if len(data) == 0 {
		return data
	}
	var sum float64
	for _, v := range data {
		sum += float64(v) * float64(v)
	}
	norm := math.Sqrt(sum)
	if norm == 0 {
		return data
	}
	out := make([]float32, len(data))
	inv := 1 / norm
	for i := range data {
		out[i] = float32(float64(data[i]) * inv)
	}
	return out
}
