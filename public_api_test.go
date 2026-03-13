// Package serverlessVector: comprehensive API contract tests.
// These tests lock down the public API so breaking changes are caught.
package serverlessVector

import (
	"math"
	"testing"
)

// --- NewVectorDB API ---

func TestAPI_NewVectorDB_DimensionZero(t *testing.T) {
	db := NewVectorDB(0)
	if db == nil {
		t.Fatal("NewVectorDB(0) must not return nil")
	}
	if db.Size() != 0 {
		t.Errorf("new db Size() must be 0, got %d", db.Size())
	}
}

func TestAPI_NewVectorDB_FixedDimension(t *testing.T) {
	db := NewVectorDB(4)
	if db == nil {
		t.Fatal("NewVectorDB(4) must not return nil")
	}
	err := db.Add("v", []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("Add with correct dimension must succeed: %v", err)
	}
	err = db.Add("v2", []float32{1, 2, 3})
	if err == nil {
		t.Fatal("Add with wrong dimension must return error")
	}
}

func TestAPI_NewVectorDB_DefaultDistance(t *testing.T) {
	db := NewVectorDB(3)
	// Default is CosineSimilarity
	_ = db.Add("a", []float32{1, 0, 0})
	_ = db.Add("b", []float32{0, 1, 0})
	res, err := db.Search([]float32{1, 0, 0}, 1)
	if err != nil {
		t.Fatalf("Search must succeed: %v", err)
	}
	if len(res.Results) != 1 || res.Results[0].ID != "a" {
		t.Errorf("cosine default: expected a, got %v", res.Results)
	}
}

func TestAPI_NewVectorDB_WithDistanceFunc(t *testing.T) {
	db := NewVectorDB(3, DotProduct)
	if db == nil {
		t.Fatal("NewVectorDB(3, DotProduct) must not return nil")
	}
	_ = db.Add("x", []float32{1, 0, 0})
	res, _ := db.Search([]float32{1, 0, 0}, 1)
	if len(res.Results) != 1 || res.Results[0].ID != "x" {
		t.Errorf("dot product: expected x, got %v", res.Results)
	}
}

func TestAPI_NewVectorDB_PanicNegativeDimension(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("NewVectorDB(-1) must panic")
		}
	}()
	_ = NewVectorDB(-1)
}

// --- Add API ---

func TestAPI_Add_EmptyID(t *testing.T) {
	db := NewVectorDB(2)
	err := db.Add("", []float32{1, 2})
	if err == nil {
		t.Fatal("Add with empty ID must return error")
	}
}

func TestAPI_Add_RejectsFloat64(t *testing.T) {
	db := NewVectorDB(2)
	err := db.Add("id", []float64{1.0, 2.0})
	if err == nil {
		t.Fatal("Add must reject []float64")
	}
}

func TestAPI_Add_EmptyData(t *testing.T) {
	db := NewVectorDB(0)
	err := db.Add("id", []float32{})
	if err == nil {
		t.Fatal("Add with empty vector must return error")
	}
}

func TestAPI_Add_WithMetadata(t *testing.T) {
	db := NewVectorDB(2)
	meta := VectorMetadata{Tags: map[string]string{"k": "v"}}
	err := db.Add("id", []float32{1, 2}, meta)
	if err != nil {
		t.Fatalf("Add with metadata must succeed: %v", err)
	}
	vec, err := db.Get("id")
	if err != nil || vec == nil {
		t.Fatalf("Get must return vector: %v", err)
	}
	if vec.Metadata.Tags == nil || vec.Metadata.Tags["k"] != "v" {
		t.Errorf("metadata tags must be preserved: %+v", vec.Metadata)
	}
	if vec.Metadata.CreatedAt == 0 || vec.Metadata.UpdatedAt == 0 {
		t.Error("CreatedAt/UpdatedAt must be set")
	}
}

func TestAPI_Add_OverwritesSameID(t *testing.T) {
	db := NewVectorDB(2)
	_ = db.Add("id", []float32{1, 2})
	err := db.Add("id", []float32{3, 4})
	if err != nil {
		t.Fatalf("second Add same ID must succeed (overwrite): %v", err)
	}
	vec, _ := db.Get("id")
	if vec.Data[0] != 3 || vec.Data[1] != 4 {
		t.Errorf("data must be overwritten: got %v", vec.Data)
	}
}

// --- Get API ---

func TestAPI_Get_NotFound(t *testing.T) {
	db := NewVectorDB(2)
	vec, err := db.Get("nonexistent")
	if err == nil {
		t.Fatal("Get nonexistent must return error")
	}
	if vec != nil {
		t.Error("Get nonexistent must return nil vector")
	}
}

func TestAPI_Get_ReturnsCopy(t *testing.T) {
	db := NewVectorDB(2)
	orig := []float32{1, 2}
	_ = db.Add("id", orig)
	vec, _ := db.Get("id")
	vec.Data[0] = 999
	vec2, _ := db.Get("id")
	if vec2.Data[0] != 1 {
		t.Error("Get must return a copy; mutating result must not affect stored data")
	}
}

func TestAPI_Get_ResultShape(t *testing.T) {
	db := NewVectorDB(2)
	_ = db.Add("id", []float32{1, 2})
	vec, err := db.Get("id")
	if err != nil || vec == nil {
		t.Fatalf("Get failed: %v", err)
	}
	if vec.ID != "id" {
		t.Errorf("Vector.ID must be set: %s", vec.ID)
	}
	if vec.Dimension != 2 {
		t.Errorf("Vector.Dimension must be set: %d", vec.Dimension)
	}
	if len(vec.Data) != 2 {
		t.Errorf("Vector.Data length must match: %d", len(vec.Data))
	}
}

// --- Update API ---

func TestAPI_Update_EmptyID(t *testing.T) {
	db := NewVectorDB(2)
	_ = db.Add("a", []float32{1, 2})
	err := db.Update("", []float32{3, 4})
	if err == nil {
		t.Fatal("Update with empty ID must return error")
	}
}

func TestAPI_Update_NotFound(t *testing.T) {
	db := NewVectorDB(2)
	err := db.Update("nonexistent", []float32{1, 2})
	if err == nil {
		t.Fatal("Update nonexistent must return error")
	}
}

func TestAPI_Update_Success(t *testing.T) {
	db := NewVectorDB(2)
	_ = db.Add("a", []float32{1, 2})
	err := db.Update("a", []float32{5, 6})
	if err != nil {
		t.Fatalf("Update must succeed: %v", err)
	}
	vec, _ := db.Get("a")
	if vec.Data[0] != 5 || vec.Data[1] != 6 {
		t.Errorf("Update must change data: got %v", vec.Data)
	}
	if vec.Metadata.UpdatedAt == 0 {
		t.Error("UpdatedAt must be set")
	}
}

func TestAPI_Update_RejectsWrongDimension(t *testing.T) {
	db := NewVectorDB(3)
	_ = db.Add("a", []float32{1, 2, 3})
	err := db.Update("a", []float32{1, 2})
	if err == nil {
		t.Fatal("Update with wrong dimension must return error")
	}
}

// --- Delete API ---

func TestAPI_Delete_NotFound(t *testing.T) {
	db := NewVectorDB(2)
	err := db.Delete("nonexistent")
	if err == nil {
		t.Fatal("Delete nonexistent must return error")
	}
}

func TestAPI_Delete_Success(t *testing.T) {
	db := NewVectorDB(2)
	_ = db.Add("a", []float32{1, 2})
	err := db.Delete("a")
	if err != nil {
		t.Fatalf("Delete must succeed: %v", err)
	}
	if db.Size() != 0 {
		t.Errorf("Size must be 0 after delete: %d", db.Size())
	}
	_, err = db.Get("a")
	if err == nil {
		t.Fatal("Get after Delete must return error")
	}
}

// --- Size / Clear API ---

func TestAPI_Size(t *testing.T) {
	db := NewVectorDB(2)
	if db.Size() != 0 {
		t.Errorf("empty Size() must be 0: %d", db.Size())
	}
	_ = db.Add("a", []float32{1, 2})
	_ = db.Add("b", []float32{1, 2})
	if db.Size() != 2 {
		t.Errorf("Size() must be 2: %d", db.Size())
	}
}

func TestAPI_Clear(t *testing.T) {
	db := NewVectorDB(2)
	_ = db.Add("a", []float32{1, 2})
	db.Clear()
	if db.Size() != 0 {
		t.Errorf("Clear must leave Size 0: %d", db.Size())
	}
	_, err := db.Get("a")
	if err == nil {
		t.Fatal("Get after Clear must return error")
	}
}

// --- BatchAdd API ---

func TestAPI_BatchAdd_Empty(t *testing.T) {
	db := NewVectorDB(2)
	err := db.BatchAdd(map[string]any{}, nil)
	if err == nil {
		t.Fatal("BatchAdd with no vectors must return error")
	}
}

func TestAPI_BatchAdd_EmptyID(t *testing.T) {
	db := NewVectorDB(2)
	err := db.BatchAdd(map[string]any{"": []float32{1, 2}}, nil)
	if err == nil {
		t.Fatal("BatchAdd with empty ID must return error")
	}
}

func TestAPI_BatchAdd_RejectsFloat64(t *testing.T) {
	db := NewVectorDB(2)
	err := db.BatchAdd(map[string]any{"a": []float64{1, 2}}, nil)
	if err == nil {
		t.Fatal("BatchAdd must reject []float64")
	}
}

func TestAPI_BatchAdd_WithMetadata(t *testing.T) {
	db := NewVectorDB(2)
	vecs := map[string]any{"a": []float32{1, 2}, "b": []float32{3, 4}}
	meta := map[string]VectorMetadata{"a": {Tags: map[string]string{"t": "1"}}}
	err := db.BatchAdd(vecs, meta)
	if err != nil {
		t.Fatalf("BatchAdd must succeed: %v", err)
	}
	if db.Size() != 2 {
		t.Errorf("BatchAdd must add 2: %d", db.Size())
	}
	va, _ := db.Get("a")
	if va.Metadata.Tags == nil || va.Metadata.Tags["t"] != "1" {
		t.Errorf("BatchAdd metadata must be applied: %+v", va.Metadata)
	}
}

// --- Search API ---

func TestAPI_Search_DefaultTopK(t *testing.T) {
	db := NewVectorDB(2)
	for i := 0; i < 15; i++ {
		_ = db.Add(string(rune('a'+i)), []float32{float32(i), float32(i)})
	}
	res, err := db.Search([]float32{0, 0})
	if err != nil {
		t.Fatalf("Search must succeed: %v", err)
	}
	if res.Total != 10 {
		t.Errorf("Search() with no topK must return 10 results: got %d", res.Total)
	}
	if len(res.Results) != 10 {
		t.Errorf("Search().Results length must be 10: got %d", len(res.Results))
	}
}

func TestAPI_Search_ExplicitTopK(t *testing.T) {
	db := NewVectorDB(2)
	_ = db.Add("a", []float32{1, 0})
	_ = db.Add("b", []float32{0, 1})
	res, err := db.Search([]float32{1, 0}, 1)
	if err != nil {
		t.Fatalf("Search(query, 1) must succeed: %v", err)
	}
	if res.Total != 1 || len(res.Results) != 1 {
		t.Errorf("topK=1 must return 1 result: %d", res.Total)
	}
}

func TestAPI_Search_RejectsFloat64(t *testing.T) {
	db := NewVectorDB(2)
	_, err := db.Search([]float64{1, 2})
	if err == nil {
		t.Fatal("Search must reject []float64 query")
	}
}

func TestAPI_Search_EmptyQuery(t *testing.T) {
	db := NewVectorDB(2)
	_, err := db.Search([]float32{})
	if err == nil {
		t.Fatal("Search with empty query must return error")
	}
}

func TestAPI_Search_DimensionMismatch(t *testing.T) {
	db := NewVectorDB(3)
	_ = db.Add("a", []float32{1, 2, 3})
	_, err := db.Search([]float32{1, 2})
	if err == nil {
		t.Fatal("Search with wrong query dimension must return error")
	}
}

func TestAPI_Search_EmptyDB(t *testing.T) {
	db := NewVectorDB(2)
	res, err := db.Search([]float32{1, 2})
	if err != nil {
		t.Fatalf("Search on empty DB must succeed with empty results: %v", err)
	}
	if res.Total != 0 {
		t.Errorf("Search empty DB Total must be 0: %d", res.Total)
	}
	// Results may be nil or empty slice; must be safe to iterate
	for _ = range res.Results {
		// no-op
	}
}

func TestAPI_Search_ResultShape(t *testing.T) {
	db := NewVectorDB(2)
	_ = db.Add("id", []float32{1, 2}, VectorMetadata{Tags: map[string]string{"x": "y"}})
	res, _ := db.Search([]float32{1, 2}, 1)
	if len(res.Results) != 1 {
		t.Fatalf("expected 1 result: %d", len(res.Results))
	}
	r := res.Results[0]
	if r.ID != "id" {
		t.Errorf("SimilarityResult.ID: got %s", r.ID)
	}
	if r.Score == 0 && db.Size() > 0 {
		// cosine of same vector is 1
		t.Logf("Score: %f (expect 1.0 for same vector)", r.Score)
	}
	if r.Metadata.Tags == nil || r.Metadata.Tags["x"] != "y" {
		t.Errorf("metadata in result: %+v", r.Metadata)
	}
}

// --- SearchWithFilter API ---

func TestAPI_SearchWithFilter_TopKZero(t *testing.T) {
	db := NewVectorDB(2)
	_ = db.Add("a", []float32{1, 2})
	res, err := db.SearchWithFilter([]float32{1, 2}, 0, func(*Vector) bool { return true })
	if err != nil {
		t.Fatalf("SearchWithFilter topK=0 must succeed (defaults to 10): %v", err)
	}
	if res.Total != 1 {
		t.Errorf("expected 1 result: %d", res.Total)
	}
}

func TestAPI_SearchWithFilter_NilFilter(t *testing.T) {
	db := NewVectorDB(2)
	_ = db.Add("a", []float32{1, 2})
	res, err := db.SearchWithFilter([]float32{1, 2}, 10, nil)
	if err != nil {
		t.Fatalf("SearchWithFilter with nil filter must succeed: %v", err)
	}
	if res.Total != 1 {
		t.Errorf("nil filter should include all: %d", res.Total)
	}
}

// --- BatchSearch API ---

func TestAPI_BatchSearch_DefaultTopK(t *testing.T) {
	db := NewVectorDB(2)
	_ = db.Add("a", []float32{1, 0})
	_ = db.Add("b", []float32{0, 1})
	queries := map[string]any{"q1": []float32{1, 0}, "q2": []float32{0, 1}}
	results, err := db.BatchSearch(queries)
	if err != nil {
		t.Fatalf("BatchSearch must succeed: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("BatchSearch must return one result per query: %d", len(results))
	}
	for qid, res := range results {
		if res.QueryID != qid {
			t.Errorf("result QueryID must match key: %s vs %s", res.QueryID, qid)
		}
		if res.Total < 1 {
			t.Errorf("each query must have at least one result: %d", res.Total)
		}
	}
}

func TestAPI_BatchSearch_ExplicitTopK(t *testing.T) {
	db := NewVectorDB(2)
	_ = db.Add("a", []float32{1, 2})
	results, err := db.BatchSearch(map[string]any{"q": []float32{1, 2}}, 1)
	if err != nil {
		t.Fatalf("BatchSearch(queries, 1) must succeed: %v", err)
	}
	if results["q"].Total != 1 {
		t.Errorf("topK=1: got %d", results["q"].Total)
	}
}

func TestAPI_BatchSearch_QueryError(t *testing.T) {
	db := NewVectorDB(2)
	_, err := db.BatchSearch(map[string]any{"q": []float64{1, 2}})
	if err == nil {
		t.Fatal("BatchSearch with invalid query must return error")
	}
}

// --- SearchMMR API ---

func TestAPI_SearchMMR_DefaultTopK(t *testing.T) {
	db := NewVectorDB(2)
	_ = db.Add("a", []float32{1, 0})
	_ = db.Add("b", []float32{0, 1})
	res, err := db.SearchMMR([]float32{1, 0}, 0)
	if err != nil {
		t.Fatalf("SearchMMR topK=0 must succeed (default 10): %v", err)
	}
	if res.Total != 2 {
		t.Errorf("expected 2 results: %d", res.Total)
	}
}

func TestAPI_SearchMMR_WithOptions(t *testing.T) {
	db := NewVectorDB(2)
	_ = db.Add("a", []float32{1, 0})
	res, err := db.SearchMMR([]float32{1, 0}, 5, &MMROptions{Lambda: 0.8, FetchFactor: 3})
	if err != nil {
		t.Fatalf("SearchMMR with options must succeed: %v", err)
	}
	if res.Total != 1 {
		t.Errorf("expected 1 result: %d", res.Total)
	}
}

func TestAPI_SearchMMR_NilOptions(t *testing.T) {
	db := NewVectorDB(2)
	_ = db.Add("a", []float32{1, 0})
	res, err := db.SearchMMR([]float32{1, 0}, 5)
	if err != nil {
		t.Fatalf("SearchMMR with no options must succeed: %v", err)
	}
	if res == nil {
		t.Fatal("SearchMMR must not return nil result")
	}
}

// --- SearchMMRParams API ---

func TestAPI_SearchMMRParams(t *testing.T) {
	db := NewVectorDB(2)
	_ = db.Add("a", []float32{1, 0})
	res, err := db.SearchMMRParams([]float32{1, 0}, 5, 0.6)
	if err != nil {
		t.Fatalf("SearchMMRParams must succeed: %v", err)
	}
	if res.Total != 1 {
		t.Errorf("expected 1 result: %d", res.Total)
	}
	res2, err := db.SearchMMRParams([]float32{1, 0}, 5, 0.5, 10)
	if err != nil {
		t.Fatalf("SearchMMRParams with fetchFactor must succeed: %v", err)
	}
	if res2.Total != 1 {
		t.Errorf("expected 1 result: %d", res2.Total)
	}
}

// --- SearchMMRWithScores API ---

func TestAPI_SearchMMRWithScores_NilOptions(t *testing.T) {
	db := NewVectorDB(2)
	_ = db.Add("a", []float32{1, 0})
	res, err := db.SearchMMRWithScores([]float32{1, 0}, 5, nil, nil)
	if err != nil {
		t.Fatalf("SearchMMRWithScores with nil opts must succeed: %v", err)
	}
	if res == nil {
		t.Fatal("result must not be nil")
	}
}

func TestAPI_SearchMMRWithScores_EmptyDB(t *testing.T) {
	db := NewVectorDB(2)
	res, err := db.SearchMMRWithScores([]float32{1, 0}, 5, nil, nil)
	if err != nil {
		t.Fatalf("SearchMMRWithScores on empty DB must succeed: %v", err)
	}
	if res.Total != 0 {
		t.Errorf("empty DB must return Total 0: %d", res.Total)
	}
}

// --- SelectMMRFromCandidates API ---

func TestAPI_SelectMMRFromCandidates_Empty(t *testing.T) {
	db := NewVectorDB(2)
	res, err := db.SelectMMRFromCandidates(nil, 5, nil)
	if err != nil {
		t.Fatalf("SelectMMRFromCandidates empty must succeed: %v", err)
	}
	if res.Total != 0 {
		t.Errorf("empty candidates must return Total 0: %d", res.Total)
	}
	if res.Results == nil {
		t.Error("Results must be non-nil slice")
	}
}

func TestAPI_SelectMMRFromCandidates_TopKZero(t *testing.T) {
	db := NewVectorDB(2)
	cands := []MMRCandidate{{ID: "a", Embedding: []float32{1, 0}, BaseScore: 1}}
	res, err := db.SelectMMRFromCandidates(cands, 0, nil)
	if err != nil {
		t.Fatalf("topK=0 must default and succeed: %v", err)
	}
	if res.Total != 1 {
		t.Errorf("expected 1 result: %d", res.Total)
	}
}

func TestAPI_SelectMMRFromCandidates_ResultShape(t *testing.T) {
	db := NewVectorDB(2)
	cands := []MMRCandidate{
		{ID: "A", Embedding: []float32{1, 0}, BaseScore: 1.0},
		{ID: "B", Embedding: []float32{0, 1}, BaseScore: 0.5},
	}
	res, err := db.SelectMMRFromCandidates(cands, 2, &MMROptions{Lambda: 0.6})
	if err != nil {
		t.Fatalf("SelectMMRFromCandidates must succeed: %v", err)
	}
	if res.Total != 2 {
		t.Errorf("expected 2 results: %d", res.Total)
	}
	for _, r := range res.Results {
		if r.ID == "" {
			t.Error("SimilarityResult.ID must be set")
		}
	}
}

// --- GetStats API ---

func TestAPI_GetStats_Shape(t *testing.T) {
	db := NewVectorDB(4)
	_ = db.Add("a", []float32{1, 2, 3, 4})
	stats := db.GetStats()
	if stats == nil {
		t.Fatal("GetStats must not return nil")
	}
	keys := []string{"total_vectors", "total_dimensions", "avg_dimensions", "memory_usage_kb", "distance_function", "dimension"}
	for _, k := range keys {
		if _, ok := stats[k]; !ok {
			t.Errorf("GetStats must contain key %q", k)
		}
	}
	if stats["total_vectors"].(int) != 1 {
		t.Errorf("total_vectors: got %v", stats["total_vectors"])
	}
	if stats["dimension"].(int) != 4 {
		t.Errorf("dimension: got %v", stats["dimension"])
	}
	if stats["memory_usage_kb"].(int64) < 0 {
		t.Error("memory_usage_kb must be non-negative")
	}
}

// --- Exported type and constant compatibility ---

func TestAPI_ExportedTypesExist(t *testing.T) {
	// Compile-time checks that re-exported types are usable
	var _ VectorDB
	var _ *VectorDB
	var _ Vector
	var _ VectorMetadata
	var _ SearchResult
	var _ SimilarityResult
	var _ MMROptions
	var _ MMRCandidate
	var _ DistanceFunction
	var _ MMRScoreMode

	var _ VectorType = Float32
	var _ DistanceFunction = CosineSimilarity
	var _ DistanceFunction = DotProduct
	var _ DistanceFunction = EuclideanDistance
	var _ DistanceFunction = ManhattanDistance
	var _ MMRScoreMode = MMRScoreQueryOnly
	var _ MMRScoreMode = MMRScoreBaseOnly
	var _ MMRScoreMode = MMRScoreBlend
}

// --- Distance function behavior (contract) ---

func TestAPI_EuclideanDistance_LowerIsBetter(t *testing.T) {
	db := NewVectorDB(2, EuclideanDistance)
	_ = db.Add("near", []float32{1, 1})
	_ = db.Add("far", []float32{10, 10})
	res, _ := db.Search([]float32{1, 1}, 2)
	if res.Results[0].ID != "near" {
		t.Errorf("Euclidean: nearer point must win: got %s", res.Results[0].ID)
	}
	if len(res.Results) >= 2 && res.Results[0].Score >= res.Results[1].Score {
		t.Error("Euclidean: first result must have lower (better) score than second")
	}
}

func TestAPI_ManhattanDistance_LowerIsBetter(t *testing.T) {
	db := NewVectorDB(2, ManhattanDistance)
	_ = db.Add("a", []float32{0, 0})
	_ = db.Add("b", []float32{5, 5})
	res, _ := db.Search([]float32{0, 0}, 1)
	if res.Results[0].ID != "a" {
		t.Errorf("Manhattan: origin must win for query [0,0]: got %s", res.Results[0].ID)
	}
}

func TestAPI_CosineSimilarity_HigherIsBetter(t *testing.T) {
	db := NewVectorDB(2)
	_ = db.Add("same", []float32{1, 0})
	_ = db.Add("orth", []float32{0, 1})
	res, _ := db.Search([]float32{1, 0}, 2)
	if res.Results[0].ID != "same" {
		t.Errorf("Cosine: same direction must win: got %s", res.Results[0].ID)
	}
	if len(res.Results) >= 2 && res.Results[0].Score < res.Results[1].Score {
		t.Error("Cosine: first result must have higher score")
	}
}

// --- Edge: NaN/Inf in BaseScore for MMR ---

func TestAPI_SelectMMRFromCandidates_NaNBaseScoreTreatedAsZero(t *testing.T) {
	db := NewVectorDB(2)
	cands := []MMRCandidate{
		{ID: "a", Embedding: []float32{1, 0}, BaseScore: math.NaN()},
		{ID: "b", Embedding: []float32{0, 1}, BaseScore: 1.0},
	}
	res, err := db.SelectMMRFromCandidates(cands, 2, &MMROptions{Lambda: 1})
	if err != nil {
		t.Fatalf("must succeed: %v", err)
	}
	// b should be first (higher relevance)
	if res.Total >= 1 && res.Results[0].ID != "b" {
		t.Errorf("NaN BaseScore should be treated as 0; b should be first: %v", res.Results)
	}
}
