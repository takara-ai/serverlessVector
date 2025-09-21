package lib

import (
	"fmt"
	"math"
)

// NumericType represents supported numeric types for vectors
type NumericType interface {
	float32 | float64
}

// NumericOps provides operations for different numeric types
type NumericOps interface {
	Add(a, b interface{}) interface{}
	Mul(a, b interface{}) interface{}
	Sqrt(a interface{}) interface{}
	ToFloat64(a interface{}) float64
	FromFloat64(a float64) interface{}
	IsNaN(a interface{}) bool
	IsInf(a interface{}) bool
}

// getNumericOps returns the appropriate NumericOps for a VectorType
func getNumericOps(vt VectorType) NumericOps {
	switch vt {
	case Float32:
		return Float32Ops{}
	case Float64:
		return Float64Ops{}
	default:
		return Float64Ops{} // fallback
	}
}

// convertToInterfaceSlice converts a typed slice to []interface{} and returns the VectorType
func convertToInterfaceSlice(data interface{}) ([]interface{}, VectorType, error) {
	switch v := data.(type) {
	case []float32:
		result := make([]interface{}, len(v))
		for i, val := range v {
			result[i] = val
		}
		return result, Float32, nil
	case []float64:
		result := make([]interface{}, len(v))
		for i, val := range v {
			result[i] = val
		}
		return result, Float64, nil
	default:
		return nil, 0, fmt.Errorf("unsupported vector type: %T", data)
	}
}

// convertFromInterfaceSlice converts []interface{} back to the appropriate typed slice
func convertFromInterfaceSlice(data []interface{}, vt VectorType) interface{} {
	switch vt {
	case Float32:
		result := make([]float32, len(data))
		for i, val := range data {
			result[i] = val.(float32)
		}
		return result
	case Float64:
		result := make([]float64, len(data))
		for i, val := range data {
			result[i] = val.(float64)
		}
		return result
	default:
		return data
	}
}

// Float32Ops implements NumericOps for float32
type Float32Ops struct{}

func (Float32Ops) Add(a, b interface{}) interface{} { return a.(float32) + b.(float32) }
func (Float32Ops) Mul(a, b interface{}) interface{} { return a.(float32) * b.(float32) }
func (Float32Ops) Sqrt(a interface{}) interface{} { return float32(math.Sqrt(float64(a.(float32)))) }
func (Float32Ops) ToFloat64(a interface{}) float64 { return float64(a.(float32)) }
func (Float32Ops) FromFloat64(a float64) interface{} { return float32(a) }
func (Float32Ops) IsNaN(a interface{}) bool { return math.IsNaN(float64(a.(float32))) }
func (Float32Ops) IsInf(a interface{}) bool { return math.IsInf(float64(a.(float32)), 0) }

// Float64Ops implements NumericOps for float64
type Float64Ops struct{}

func (Float64Ops) Add(a, b interface{}) interface{} { return a.(float64) + b.(float64) }
func (Float64Ops) Mul(a, b interface{}) interface{} { return a.(float64) * b.(float64) }
func (Float64Ops) Sqrt(a interface{}) interface{} { return math.Sqrt(a.(float64)) }
func (Float64Ops) ToFloat64(a interface{}) float64 { return a.(float64) }
func (Float64Ops) FromFloat64(a float64) interface{} { return a }
func (Float64Ops) IsNaN(a interface{}) bool { return math.IsNaN(a.(float64)) }
func (Float64Ops) IsInf(a interface{}) bool { return math.IsInf(a.(float64), 0) }

// VectorType represents the configuration for vector storage
type VectorType int

const (
	Float32 VectorType = iota
	Float64
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
	Value   interface{}
	Message string
}

// Vector represents a vector with metadata that can hold different numeric types
type Vector struct {
	ID       string
	Data     []interface{} // Can hold []float32 or []float64 elements
	Type     VectorType
	Metadata VectorMetadata
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


// NormalizeVector normalizes a vector to unit length
func NormalizeVector(data interface{}) interface{} {
	dataSlice, vecType, err := convertToInterfaceSlice(data)
	if err != nil || len(dataSlice) == 0 {
		return data
	}

	ops := getNumericOps(vecType)

	// Calculate norm
	sum := ops.FromFloat64(0.0)
	for _, val := range dataSlice {
		sum = ops.Add(sum, ops.Mul(val, val))
	}
	norm := ops.Sqrt(sum)

	if ops.ToFloat64(norm) == 0 {
		return data // Avoid division by zero
	}

	// Normalize
	resultSlice := make([]interface{}, len(dataSlice))
	for i, val := range dataSlice {
		resultSlice[i] = ops.Mul(val, ops.FromFloat64(1.0/ops.ToFloat64(norm)))
	}

	return convertFromInterfaceSlice(resultSlice, vecType)
}

// ConvertVectorType converts between different numeric types
func ConvertVectorType(data interface{}, targetType VectorType) interface{} {
	sourceSlice, sourceType, err := convertToInterfaceSlice(data)
	if err != nil || sourceType == targetType {
		return data // No conversion needed or error
	}

	sourceOps := getNumericOps(sourceType)
	targetOps := getNumericOps(targetType)

	// Convert
	resultSlice := make([]interface{}, len(sourceSlice))
	for i, val := range sourceSlice {
		resultSlice[i] = targetOps.FromFloat64(sourceOps.ToFloat64(val))
	}

	return convertFromInterfaceSlice(resultSlice, targetType)
}
