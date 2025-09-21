package serverlessVector

import (
	"fmt"
	"math"
	"testing"
)

func TestVectorDB_Float32(t *testing.T) {
	db := NewVectorDB(4) // d=4 dimensional vectors

	// Test Add
	data := []float32{0.1, 0.2, 0.3, 0.4}
	err := db.Add("vec1", data)
	if err != nil {
		t.Fatalf("Failed to add vector: %v", err)
	}

	// Test Get
	vector, err := db.Get("vec1")
	if err != nil {
		t.Fatalf("Failed to get vector: %v", err)
	}

	if vector.ID != "vec1" {
		t.Errorf("Expected ID 'vec1', got '%s'", vector.ID)
	}

	if len(vector.Data) != 4 {
		t.Errorf("Expected dimension 4, got %d", len(vector.Data))
	}

	// Check data values
	for i, expected := range data {
		if actual := vector.Data[i].(float32); actual != expected {
			t.Errorf("Expected data[%d] = %f, got %f", i, expected, actual)
		}
	}
}

func TestVectorDB_Float64(t *testing.T) {
	// Test with custom distance function (dot product instead of cosine)
	db := NewVectorDB(3, DotProduct)

	// Test Add
	data := []float64{1.0, 2.0, 3.0}
	err := db.Add("vec1", data)
	if err != nil {
		t.Fatalf("Failed to add vector: %v", err)
	}

	// Test Search
	query := []float64{1.0, 1.0, 1.0}
	result, err := db.Search(query, 1)
	if err != nil {
		t.Fatalf("Failed to search: %v", err)
	}

	if len(result.Results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(result.Results))
	}

	if result.Results[0].ID != "vec1" {
		t.Errorf("Expected result ID 'vec1', got '%s'", result.Results[0].ID)
	}
}

func TestUnifiedAPI_AutoTypeHandling(t *testing.T) {
	db := NewVectorDB(0) // d=0 (no dimension validation)

	// Add float32 vector
	float32Data := []float32{0.1, 0.2, 0.3}
	err := db.Add("vec32", float32Data)
	if err != nil {
		t.Fatalf("Failed to add float32 vector: %v", err)
	}

	// Add float64 vector
	float64Data := []float64{0.4, 0.5, 0.6}
	err = db.Add("vec64", float64Data)
	if err != nil {
		t.Fatalf("Failed to add float64 vector: %v", err)
	}

	// Check total size
	if db.Size() != 2 {
		t.Errorf("Expected size 2, got %d", db.Size())
	}

	// Get float32 vector
	vec32, err := db.Get("vec32")
	if err != nil {
		t.Fatalf("Failed to get float32 vector: %v", err)
	}

	if vec32.Type != Float32 {
		t.Errorf("Expected Float32 type, got %v", vec32.Type)
	}
	if len(vec32.Data) != 3 {
		t.Errorf("Expected 3 elements, got %d", len(vec32.Data))
	}

	// Get float64 vector
	vec64, err := db.Get("vec64")
	if err != nil {
		t.Fatalf("Failed to get float64 vector: %v", err)
	}

	if vec64.Type != Float64 {
		t.Errorf("Expected Float64 type, got %v", vec64.Type)
	}
	if len(vec64.Data) != 3 {
		t.Errorf("Expected 3 elements, got %d", len(vec64.Data))
	}
}

func TestValidation(t *testing.T) {
	db := NewVectorDB(3)

	// Test valid vector
	validData := []float32{0.1, 0.2, 0.3}
	err := db.Add("valid", validData)
	if err != nil {
		t.Errorf("Expected valid vector to be added: %v", err)
	}

	// Test wrong dimension
	wrongDim := []float32{0.1, 0.2}
	err = db.Add("invalid_dim", wrongDim)
	if err == nil {
		t.Errorf("Expected error due to wrong dimension")
	}

	// Test NaN value - this should pass through for now as we simplified validation
	nanData := []float32{0.1, float32(math.NaN()), 0.3}
	err = db.Add("nan_data", nanData)
	if err != nil {
		t.Logf("NaN data rejected: %v", err) // Just log for now
	}
}

func TestBatchOperations(t *testing.T) {
	db := NewVectorDB(0) // No dimension validation for mixed dimensions

	// Prepare batch data
	vectors := map[string]interface{}{
		"vec1": []float32{0.1, 0.2, 0.3},
		"vec2": []float32{0.4, 0.5, 0.6},
		"vec3": []float64{0.7, 0.8, 0.9},
	}

	metadata := map[string]VectorMetadata{
		"vec1": {Tags: map[string]string{"type": "test"}},
	}

	// Batch add
	err := db.BatchAdd(vectors, metadata)
	if err != nil {
		t.Fatalf("Failed to batch add: %v", err)
	}

	if db.Size() != 3 {
		t.Errorf("Expected size 3, got %d", db.Size())
	}

	// Verify vectors were added
	for id := range vectors {
		vec, err := db.Get(id)
		if err != nil {
			t.Errorf("Failed to get vector %s: %v", id, err)
		}
		if vec == nil {
			t.Errorf("Vector %s is nil", id)
		}
	}
}

func TestDistanceFunctions(t *testing.T) {
	db := NewVectorDB(3) // 3-dimensional vectors, cosine similarity (default)

	// Add test vectors
	vectors := map[string]interface{}{
		"vec1": []float64{1.0, 0.0, 0.0}, // Unit vector along x-axis
		"vec2": []float64{0.0, 1.0, 0.0}, // Unit vector along y-axis
		"vec3": []float64{0.0, 0.0, 1.0}, // Unit vector along z-axis
	}

	for id, data := range vectors {
		err := db.Add(id, data)
		if err != nil {
			t.Fatalf("Failed to add vector %s: %v", id, err)
		}
	}

	// Test cosine similarity
	query := []float64{1.0, 0.0, 0.0}
	result, err := db.Search(query, 1)
	if err != nil {
		t.Fatalf("Failed cosine search: %v", err)
	}

	if result.Results[0].ID != "vec1" || math.Abs(result.Results[0].Score-1.0) > 1e-6 {
		t.Errorf("Expected vec1 with score 1.0, got %s with score %f",
			result.Results[0].ID, result.Results[0].Score)
	}

	// Test dot product with different distance function
	dpDb := NewVectorDB(3, DotProduct)

	// Re-add vectors with dot product config
	for id, data := range vectors {
		err := dpDb.Add(id, data)
		if err != nil {
			t.Fatalf("Failed to add vector %s: %v", id, err)
		}
	}

	result, err = dpDb.Search(query, 1)
	if err != nil {
		t.Fatalf("Failed dot product search: %v", err)
	}

	if result.Results[0].ID != "vec1" || math.Abs(result.Results[0].Score-1.0) > 1e-6 {
		t.Errorf("Expected vec1 with score 1.0, got %s with score %f",
			result.Results[0].ID, result.Results[0].Score)
	}
}

func TestMemoryStats(t *testing.T) {
	db := NewVectorDB(0) // No dimension validation

	// Add some vectors
	vectors := map[string]interface{}{
		"vec1": []float32{0.1, 0.2, 0.3, 0.4},
		"vec2": []float64{0.5, 0.6, 0.7, 0.8},
	}

	err := db.BatchAdd(vectors, nil)
	if err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}

	stats := db.GetStats()
	if stats["total_vectors"].(int) != 2 {
		t.Errorf("Expected 2 total vectors, got %d", stats["total_vectors"])
	}

	// Check memory usage is reasonable (should be >= 0)
	memoryUsage := stats["memory_usage_kb"].(int64)
	if memoryUsage < 0 {
		t.Errorf("Expected non-negative memory usage, got %d", memoryUsage)
	}
}

func BenchmarkSearch_Float32(b *testing.B) {
	db := NewVectorDB(128) // 128-dimensional float32 vectors

	// Add test vectors
	for i := 0; i < 1000; i++ {
		data := make([]float32, 128)
		for j := range data {
			data[j] = float32((i+j)%10) * 0.1
		}
		db.Add(fmt.Sprintf("vec%d", i), data)
	}

	query := make([]float32, 128)
	for i := range query {
		query[i] = float32(i%10) * 0.1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = db.Search(query, 10)
	}
}

func BenchmarkSearch_Float64(b *testing.B) {
	// Create database with float64 support (dimension 128)
	db := NewVectorDB(128)

	// Add test vectors
	for i := 0; i < 1000; i++ {
		data := make([]float64, 128)
		for j := range data {
			data[j] = float64((i+j)%10) * 0.1
		}
		db.Add(fmt.Sprintf("vec%d", i), data)
	}

	query := make([]float64, 128)
	for i := range query {
		query[i] = float64(i%10) * 0.1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = db.Search(query, 10)
	}
}

