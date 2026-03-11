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

	if vector.Dimension != 4 {
		t.Errorf("Expected dimension 4, got %d", vector.Dimension)
	}

	// Check data values
	for i, expected := range data {
		if actual := vector.Data[i]; actual != expected {
			t.Errorf("Expected data[%d] = %f, got %f", i, expected, actual)
		}
	}
}

func TestVectorDB_RejectsFloat64(t *testing.T) {
	db := NewVectorDB(3)
	err := db.Add("vec1", []float64{1.0, 2.0, 3.0})
	if err == nil {
		t.Error("Expected Add to reject []float64")
	}
}

func TestVectorDB_DotProduct(t *testing.T) {
	db := NewVectorDB(3, DotProduct)
	_ = db.Add("vec1", []float32{1.0, 2.0, 3.0})
	result, err := db.Search([]float32{1.0, 1.0, 1.0}, 1)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(result.Results) != 1 || result.Results[0].ID != "vec1" {
		t.Errorf("Expected vec1, got %v", result.Results)
	}
}

func TestUnifiedAPI_Float32Only(t *testing.T) {
	db := NewVectorDB(0)
	_ = db.Add("a", []float32{0.1, 0.2, 0.3})
	_ = db.Add("b", []float32{0.4, 0.5, 0.6})
	if db.Size() != 2 {
		t.Errorf("Expected size 2, got %d", db.Size())
	}
	vec, err := db.Get("a")
	if err != nil || vec.Dimension != 3 {
		t.Fatalf("Get: %v", err)
	}
	if vec.Data[0] != 0.1 || vec.Data[1] != 0.2 || vec.Data[2] != 0.3 {
		t.Errorf("Expected [0.1,0.2,0.3], got %v", vec.Data)
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

	// Prepare batch data (float32 only)
	vectors := map[string]any{
		"vec1": []float32{0.1, 0.2, 0.3},
		"vec2": []float32{0.4, 0.5, 0.6},
		"vec3": []float32{0.7, 0.8, 0.9},
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

func TestSearchWithFilter(t *testing.T) {
	db := NewVectorDB(3)
	_ = db.Add("a", []float32{1, 0, 0}, VectorMetadata{Tags: map[string]string{"cat": "x"}})
	_ = db.Add("b", []float32{0, 1, 0}, VectorMetadata{Tags: map[string]string{"cat": "y"}})
	_ = db.Add("c", []float32{0, 0, 1}, VectorMetadata{Tags: map[string]string{"cat": "x"}})
	query := []float32{1, 0, 0}
	// Filter: only cat=x
	filter := func(v *Vector) bool {
		if v.Metadata.Tags == nil {
			return false
		}
		return v.Metadata.Tags["cat"] == "x"
	}
	res, err := db.SearchWithFilter(query, 10, filter)
	if err != nil {
		t.Fatalf("SearchWithFilter failed: %v", err)
	}
	if res.Total != 2 {
		t.Errorf("Expected 2 results (cat=x), got %d", res.Total)
	}
	ids := make(map[string]bool)
	for _, r := range res.Results {
		ids[r.ID] = true
	}
	if !ids["a"] || !ids["c"] || ids["b"] {
		t.Errorf("Expected IDs a and c only, got %v", res.Results)
	}
}

func TestDistanceFunctions(t *testing.T) {
	db := NewVectorDB(3)
	vectors := map[string]any{
		"vec1": []float32{1.0, 0.0, 0.0},
		"vec2": []float32{0.0, 1.0, 0.0},
		"vec3": []float32{0.0, 0.0, 1.0},
	}
	for id, data := range vectors {
		_ = db.Add(id, data)
	}
	query := []float32{1.0, 0.0, 0.0}
	result, err := db.Search(query, 1)
	if err != nil {
		t.Fatalf("Failed cosine search: %v", err)
	}
	if result.Results[0].ID != "vec1" || math.Abs(result.Results[0].Score-1.0) > 1e-6 {
		t.Errorf("Expected vec1 with score 1.0, got %s with score %f",
			result.Results[0].ID, result.Results[0].Score)
	}
	dpDb := NewVectorDB(3, DotProduct)
	for id, data := range vectors {
		_ = dpDb.Add(id, data)
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

func TestSearchMMR(t *testing.T) {
	db := NewVectorDB(3)
	_ = db.Add("vec1", []float32{1.0, 0.0, 0.0})
	_ = db.Add("vec2", []float32{0.99, 0.01, 0.0})
	_ = db.Add("vec3", []float32{0.98, 0.02, 0.0})
	_ = db.Add("vec4", []float32{0.0, 1.0, 0.0})
	_ = db.Add("vec5", []float32{0.0, 0.0, 1.0})
	query := []float32{1.0, 0.0, 0.0}

	// Lambda=1: pure relevance, should match Search order
	mmr1, err := db.SearchMMRParams(query, 3, 1.0)
	if err != nil {
		t.Fatalf("SearchMMRParams failed: %v", err)
	}
	if mmr1.Total != 3 {
		t.Errorf("Expected 3 results, got %d", mmr1.Total)
	}
	searchRes, _ := db.Search(query, 3)
	if searchRes.Results[0].ID != mmr1.Results[0].ID {
		t.Errorf("MMR lambda=1 first result should match Search: got %s", mmr1.Results[0].ID)
	}

	// Lambda=0.5: some diversity; just ensure we get 3 results and no error
	mmr5, err := db.SearchMMRParams(query, 3, 0.5)
	if err != nil {
		t.Fatalf("SearchMMRParams(0.5) failed: %v", err)
	}
	if mmr5.Total != 3 {
		t.Errorf("Expected 3 results for lambda=0.5, got %d", mmr5.Total)
	}

	// topK larger than candidates: should return all
	mmrAll, err := db.SearchMMRParams(query, 10, 0.7)
	if err != nil {
		t.Fatalf("SearchMMRParams(10) failed: %v", err)
	}
	if mmrAll.Total != 5 {
		t.Errorf("Expected 5 results when topK=10, got %d", mmrAll.Total)
	}

	// SearchMMR (defaults) returns same count
	mmrDefault, err := db.SearchMMR(query, 3)
	if err != nil {
		t.Fatalf("SearchMMR failed: %v", err)
	}
	if mmrDefault.Total != 3 {
		t.Errorf("Expected 3 results from SearchMMR, got %d", mmrDefault.Total)
	}

	// SearchMMR with options
	optsCustom, err := db.SearchMMR(query, 3, &MMROptions{Lambda: 0.8})
	if err != nil {
		t.Fatalf("SearchMMR(opts) failed: %v", err)
	}
	if optsCustom.Total != 3 {
		t.Errorf("Expected 3 results from SearchMMR(custom), got %d", optsCustom.Total)
	}
}

func TestSelectMMRFromCandidates(t *testing.T) {
	// Use DotProduct for simpler manual calculation (no norm)
	db := NewVectorDB(3, DotProduct)

	// Candidates:
	// A: BaseScore=1.0, Vector=[1,0,0]
	// B: BaseScore=0.9, Vector=[1,0,0] (Very similar to A)
	// C: BaseScore=0.8, Vector=[0,1,0] (Orthogonal to A)
	candidates := []MMRCandidate{
		{ID: "A", BaseScore: 1.0, Embedding: []float32{1, 0, 0}},
		{ID: "B", BaseScore: 0.9, Embedding: []float32{1, 0, 0}},
		{ID: "C", BaseScore: 0.8, Embedding: []float32{0, 1, 0}},
	}

	// 1. Lambda=1 (Pure BaseScore relevance)
	// Should pick A (1.0), then B (0.9), then C (0.8)
	res1, err := db.SelectMMRFromCandidates(candidates, 3, &MMROptions{Lambda: 1.0})
	if err != nil {
		t.Fatalf("SelectMMR failed: %v", err)
	}
	if len(res1.Results) != 3 {
		t.Fatalf("Expected 3 results, got %d", len(res1.Results))
	}
	if res1.Results[0].ID != "A" || res1.Results[1].ID != "B" || res1.Results[2].ID != "C" {
		t.Errorf("Lambda=1 expected A, B, C; got %v", res1.Results)
	}

	// 2. Lambda=0.5 (Diversity matters)
	// Pick 1: A (highest score 1.0)
	// Pick 2:
	//   B: rel=0.9, sim(A,B)=1.0 -> 0.5*0.9 - 0.5*1.0 = -0.05
	//   C: rel=0.8, sim(A,C)=0.0 -> 0.5*0.8 - 0.5*0.0 = 0.40
	// So C should be picked second.
	res2, err := db.SelectMMRFromCandidates(candidates, 3, &MMROptions{Lambda: 0.5})
	if err != nil {
		t.Fatalf("SelectMMR failed: %v", err)
	}
	if res2.Results[0].ID != "A" {
		t.Errorf("First result should be A, got %s", res2.Results[0].ID)
	}
	if res2.Results[1].ID != "C" {
		t.Errorf("Second result should be C (diversity), got %s", res2.Results[1].ID)
	}
	if res2.Results[2].ID != "B" {
		t.Errorf("Third result should be B, got %s", res2.Results[2].ID)
	}
}

func TestSearchMMRWithScores(t *testing.T) {
	db := NewVectorDB(3, DotProduct)
	_ = db.Add("A", []float32{1, 0, 0})
	_ = db.Add("B", []float32{1, 0, 0}) // Same as A
	_ = db.Add("C", []float32{0, 1, 0}) // Orthogonal

	query := []float32{1, 0, 0}
	// Query Similarity: A=1, B=1, C=0

	// 1. QueryOnly (Default) - Regression check
	// Should behave like standard SearchMMR
	// Lambda=0.4
	// A: 1.0. Pick A.
	// B: rel=1.0, sim=1.0 -> 0.4*1 - 0.6*1 = -0.2
	// C: rel=0.0, sim=0.0 -> 0.0
	// C > B. So A, C, B.
	resQ, err := db.SearchMMRWithScores(query, 3, nil, &MMROptions{Lambda: 0.4, ScoreMode: MMRScoreQueryOnly})
	if err != nil {
		t.Fatalf("SearchMMRWithScores failed: %v", err)
	}
	if resQ.Results[0].ID != "A" && resQ.Results[0].ID != "B" {
		t.Errorf("First should be A or B")
	}
	if resQ.Results[1].ID != "C" {
		t.Errorf("Second should be C (diversity), got %s", resQ.Results[1].ID)
	}

	// 2. BaseScoreOnly
	// Base Scores: C=1.0, A=0.0, B=0.0
	// Pick 1: C (1.0)
	// Pick 2:
	//   A: rel=0.0, sim(C,A)=0.0 -> 0.4*0 - 0.6*0 = 0
	//   B: rel=0.0, sim(C,B)=0.0 -> 0
	// Order: C, then A/B.
	baseScores := map[string]float64{
		"C": 1.0,
		"A": 0.0,
		"B": 0.0,
	}
	resB, err := db.SearchMMRWithScores(query, 3, baseScores, &MMROptions{Lambda: 0.5, ScoreMode: MMRScoreBaseOnly})
	if err != nil {
		t.Fatalf("SearchMMRWithScores failed: %v", err)
	}
	if resB.Results[0].ID != "C" {
		t.Errorf("BaseScoreOnly: expected C first (score 1.0), got %s", resB.Results[0].ID)
	}

	// 3. Blend
	// Base: A=0.2, B=0.0, C=1.0
	// Alpha=0.5
	// Rel A = 0.5*1 + 0.5*0.2 = 0.6
	// Rel B = 0.5*1 + 0.5*0.0 = 0.5
	// Rel C = 0.5*0 + 0.5*1.0 = 0.5
	// Pick 1: A (0.6)
	// Pick 2:
	//   B: rel=0.5, sim(A,B)=1.0. MMR = lambda*0.5 - (1-lambda)*1.0.
	//      If lambda=0.8: 0.8*0.5 - 0.2*1.0 = 0.4 - 0.2 = 0.2
	//   C: rel=0.5, sim(A,C)=0.0. MMR = 0.8*0.5 - 0.2*0 = 0.4
	// C (0.4) > B (0.2). So A, C, B.
	baseScores2 := map[string]float64{
		"A": 0.2,
		"B": 0.0,
		"C": 1.0,
	}
	resBlend, err := db.SearchMMRWithScores(query, 3, baseScores2, &MMROptions{
		Lambda:     0.8,
		ScoreMode:  MMRScoreBlend,
		BlendAlpha: 0.5,
	})
	if err != nil {
		t.Fatalf("SearchMMRWithScores failed: %v", err)
	}
	if resBlend.Results[0].ID != "A" {
		t.Errorf("Blend: expected A first, got %s", resBlend.Results[0].ID)
	}
	if resBlend.Results[1].ID != "C" {
		t.Errorf("Blend: expected C second (diversity), got %s", resBlend.Results[1].ID)
	}
}

func TestMemoryStats(t *testing.T) {
	db := NewVectorDB(0) // No dimension validation

	// Add some vectors
	vectors := map[string]any{
		"vec1": []float32{0.1, 0.2, 0.3, 0.4},
		"vec2": []float32{0.5, 0.6, 0.7, 0.8},
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
	for i := range 1000 {
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

// BenchmarkSearch_512D benchmarks search with 1k vectors of 512 dims (e.g. Takara ds1-en-v1).
func BenchmarkSearch_512D(b *testing.B) {
	const dim = 512
	db := NewVectorDB(dim)
	for i := range 1000 {
		data := make([]float32, dim)
		for j := range data {
			data[j] = float32((i+j)%10) * 0.1
		}
		db.Add(fmt.Sprintf("vec%d", i), data)
	}
	query := make([]float32, dim)
	for i := range query {
		query[i] = float32(i%10) * 0.1
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = db.Search(query, 10)
	}
}

// BenchmarkSearch_512D_DotProduct is for L2-normalized embeddings (e.g. Takara ds1-en-v1):
// dot product gives same ranking as cosine, no norm computation.
func BenchmarkSearch_512D_DotProduct(b *testing.B) {
	const dim = 512
	db := NewVectorDB(dim, DotProduct)
	for i := range 1000 {
		data := make([]float32, dim)
		for j := range data {
			data[j] = float32((i+j)%10) * 0.1
		}
		db.Add(fmt.Sprintf("vec%d", i), data)
	}
	query := make([]float32, dim)
	for i := range query {
		query[i] = float32(i%10) * 0.1
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = db.Search(query, 10)
	}
}

func BenchmarkSearchMMR_Float32(b *testing.B) {
	db := NewVectorDB(128)
	for i := range 1000 {
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
		_, _ = db.SearchMMR(query, 10)
	}
}

// BenchmarkBatchAdd measures BatchAdd with write lock held only for map merge.
func BenchmarkBatchAdd(b *testing.B) {
	dim := 128
	n := 500
	vectors := make(map[string]any, n)
	for i := range n {
		data := make([]float32, dim)
		for j := range data {
			data[j] = float32((i+j)%10) * 0.1
		}
		vectors[fmt.Sprintf("vec%d", i)] = data
	}

	db := NewVectorDB(dim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = db.BatchAdd(vectors, nil)
		db.Clear()
	}
}
