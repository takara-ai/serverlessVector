// Package serverlessVector provides a simple, fast vector database
// optimized for serverless applications and embedding storage/search.
package serverlessVector

// Re-export the main types and functions from the lib package
import "github.com/takarajordan/serverlessVector/lib"

// VectorDB is the main vector database interface
type VectorDB = lib.VectorDB

// Vector represents a stored vector with metadata
type Vector = lib.Vector

// VectorMetadata holds additional information about vectors
type VectorMetadata = lib.VectorMetadata

// VectorType represents supported numeric types
type VectorType = lib.VectorType

// DistanceFunction represents different distance/similarity metrics
type DistanceFunction = lib.DistanceFunction

// SearchResult contains search results
type SearchResult = lib.SearchResult

// SimilarityResult holds individual search result
type SimilarityResult = lib.SimilarityResult

// Constants for vector types
const (
	Float32 VectorType = lib.Float32
	Float64 VectorType = lib.Float64
)

// Constants for distance functions
const (
	CosineSimilarity  DistanceFunction = lib.CosineSimilarity
	DotProduct        DistanceFunction = lib.DotProduct
	EuclideanDistance DistanceFunction = lib.EuclideanDistance
	ManhattanDistance DistanceFunction = lib.ManhattanDistance
)

// NewVectorDB creates a new vector database
// dimension: vector dimension (e.g., 384 for OpenAI, 1536 for text-embedding-ada-002)
// distanceFunc: optional distance function (defaults to CosineSimilarity if not provided)
func NewVectorDB(dimension int, distanceFunc ...DistanceFunction) *VectorDB {
	return lib.NewVectorDB(dimension, distanceFunc...)
}