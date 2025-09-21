package lib

import "math"

// dotProduct calculates the dot product of two vectors (optimized for loop unrolling)
func (db *VectorDB) dotProduct(a, b []interface{}, vecType VectorType) float64 {
	if len(a) != len(b) {
		return 0
	}

	ops := getNumericOps(vecType)
	sum := ops.FromFloat64(0.0)
	n := len(a)

	// Bounds-checking elimination: ensure we can safely access elements
	if n == 0 {
		return 0
	}

	// Unroll loop by 8 for better CPU pipelining
	for i := 0; i <= n-8; i += 8 {
		sum = ops.Add(sum, ops.Mul(a[i], b[i]))
		sum = ops.Add(sum, ops.Mul(a[i+1], b[i+1]))
		sum = ops.Add(sum, ops.Mul(a[i+2], b[i+2]))
		sum = ops.Add(sum, ops.Mul(a[i+3], b[i+3]))
		sum = ops.Add(sum, ops.Mul(a[i+4], b[i+4]))
		sum = ops.Add(sum, ops.Mul(a[i+5], b[i+5]))
		sum = ops.Add(sum, ops.Mul(a[i+6], b[i+6]))
		sum = ops.Add(sum, ops.Mul(a[i+7], b[i+7]))
	}

	// Handle remaining elements
	for i := (n / 8) * 8; i < n; i++ {
		sum = ops.Add(sum, ops.Mul(a[i], b[i]))
	}

	return ops.ToFloat64(sum)
}

// cosineSimilarity calculates cosine similarity between two vectors
func (db *VectorDB) cosineSimilarity(a, b []interface{}, vecType VectorType) float64 {
	dotProd := db.dotProduct(a, b, vecType)
	normA := db.norm(a, vecType)
	normB := db.norm(b, vecType)

	if normA == 0 || normB == 0 {
		return 0 // Avoid division by zero
	}

	return dotProd / (normA * normB)
}

// euclideanDistance calculates Euclidean distance between two vectors
func (db *VectorDB) euclideanDistance(a, b []interface{}, vecType VectorType) float64 {
	if len(a) != len(b) {
		return math.Inf(1) // Return infinity for dimension mismatch
	}

	ops := getNumericOps(vecType)
	sum := ops.FromFloat64(0.0)

	for i := range a {
		diff := ops.Add(a[i], ops.Mul(b[i], ops.FromFloat64(-1)))
		sum = ops.Add(sum, ops.Mul(diff, diff))
	}

	return ops.ToFloat64(ops.Sqrt(sum))
}

// manhattanDistance calculates Manhattan distance between two vectors
func (db *VectorDB) manhattanDistance(a, b []interface{}, vecType VectorType) float64 {
	if len(a) != len(b) {
		return math.Inf(1) // Return infinity for dimension mismatch
	}

	ops := getNumericOps(vecType)
	sum := ops.FromFloat64(0.0)

	for i := range a {
		diff := ops.Add(a[i], ops.Mul(b[i], ops.FromFloat64(-1)))
		absDiff := diff
		if ops.ToFloat64(diff) < 0 {
			absDiff = ops.Mul(diff, ops.FromFloat64(-1))
		}
		sum = ops.Add(sum, absDiff)
	}

	return ops.ToFloat64(sum)
}

// norm calculates the L2 norm of a vector
func (db *VectorDB) norm(v []interface{}, vecType VectorType) float64 {
	ops := getNumericOps(vecType)
	sum := ops.FromFloat64(0.0)

	for _, val := range v {
		sum = ops.Add(sum, ops.Mul(val, val))
	}

	return ops.ToFloat64(ops.Sqrt(sum))
}
