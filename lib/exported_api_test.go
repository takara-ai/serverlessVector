// Package lib: API contract tests for exported types and functions.
// Ensures breaking changes to NormalizeVector, DistanceFloat32, and type shapes are caught.
package lib

import (
	"math"
	"testing"
)

func TestAPI_NormalizeVector_Empty(t *testing.T) {
	out := NormalizeVector(nil)
	if out != nil {
		t.Errorf("NormalizeVector(nil) must return nil: %v", out)
	}
	out = NormalizeVector([]float32{})
	if len(out) != 0 {
		t.Errorf("NormalizeVector([]) must return empty slice: len=%d", len(out))
	}
}

func TestAPI_NormalizeVector_UnitLength(t *testing.T) {
	in := []float32{3, 4}
	out := NormalizeVector(in)
	if len(out) != 2 {
		t.Fatalf("length must be 2: %d", len(out))
	}
	norm := math.Sqrt(float64(out[0]*out[0] + out[1]*out[1]))
	if math.Abs(norm-1.0) > 1e-5 {
		t.Errorf("normalized vector must have length 1: got %f", norm)
	}
	// 3,4 -> 0.6, 0.8
	if math.Abs(float64(out[0])-0.6) > 1e-5 || math.Abs(float64(out[1])-0.8) > 1e-5 {
		t.Errorf("expected [0.6, 0.8]: got %v", out)
	}
}

func TestAPI_NormalizeVector_DoesNotMutateInput(t *testing.T) {
	in := []float32{1, 2, 3}
	orig := make([]float32, len(in))
	copy(orig, in)
	_ = NormalizeVector(in)
	for i := range in {
		if in[i] != orig[i] {
			t.Error("NormalizeVector must not mutate input slice")
			break
		}
	}
}

func TestAPI_NormalizeVector_ZeroVector(t *testing.T) {
	in := []float32{0, 0, 0}
	out := NormalizeVector(in)
	if len(out) != 3 {
		t.Fatalf("length must be 3: %d", len(out))
	}
	// Implementation returns input unchanged for zero norm
	for i := range in {
		if out[i] != 0 {
			t.Errorf("zero vector may be returned as-is: out[%d]=%f", i, out[i])
		}
	}
}

func TestAPI_DistanceFloat32_CosineSimilarity(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{1, 0, 0}
	score := DistanceFloat32(a, b, CosineSimilarity)
	if math.Abs(score-1.0) > 1e-6 {
		t.Errorf("cosine same vector: expected 1, got %f", score)
	}
	c := []float32{0, 1, 0}
	score2 := DistanceFloat32(a, c, CosineSimilarity)
	if math.Abs(score2) > 1e-6 {
		t.Errorf("cosine orthogonal: expected 0, got %f", score2)
	}
}

func TestAPI_DistanceFloat32_DotProduct(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{1, 0, 0}
	score := DistanceFloat32(a, b, DotProduct)
	if math.Abs(score-1.0) > 1e-6 {
		t.Errorf("dot product: expected 1, got %f", score)
	}
}

func TestAPI_DistanceFloat32_EuclideanDistance(t *testing.T) {
	a := []float32{0, 0}
	b := []float32{3, 4}
	score := DistanceFloat32(a, b, EuclideanDistance)
	if math.Abs(score-5.0) > 1e-5 {
		t.Errorf("euclidean [0,0] to [3,4]: expected 5, got %f", score)
	}
	same := DistanceFloat32(a, a, EuclideanDistance)
	if same != 0 {
		t.Errorf("euclidean same point: expected 0, got %f", same)
	}
}

func TestAPI_DistanceFloat32_ManhattanDistance(t *testing.T) {
	a := []float32{0, 0}
	b := []float32{1, 1}
	score := DistanceFloat32(a, b, ManhattanDistance)
	if math.Abs(score-2.0) > 1e-5 {
		t.Errorf("manhattan: expected 2, got %f", score)
	}
}

func TestAPI_DistanceFloat32_MismatchedLength(t *testing.T) {
	a := []float32{1, 2}
	b := []float32{1, 2, 3}
	eu := DistanceFloat32(a, b, EuclideanDistance)
	if !math.IsInf(eu, 1) {
		t.Errorf("euclidean mismatched length must return +Inf: %f", eu)
	}
	man := DistanceFloat32(a, b, ManhattanDistance)
	if !math.IsInf(man, 1) {
		t.Errorf("manhattan mismatched length must return +Inf: %f", man)
	}
}

func TestAPI_DistanceFloat32_EmptySlice(t *testing.T) {
	a := []float32{}
	b := []float32{}
	dot := DistanceFloat32(a, b, DotProduct)
	if dot != 0 {
		t.Errorf("dot product empty: expected 0, got %f", dot)
	}
	cos := DistanceFloat32(a, b, CosineSimilarity)
	if cos != 0 {
		t.Errorf("cosine empty: expected 0, got %f", cos)
	}
}

func TestAPI_DistanceFunction_String(t *testing.T) {
	tests := []struct {
		df   DistanceFunction
		want string
	}{
		{CosineSimilarity, "cosine_similarity"},
		{DotProduct, "dot_product"},
		{EuclideanDistance, "euclidean_distance"},
		{ManhattanDistance, "manhattan_distance"},
	}
	for _, tt := range tests {
		got := tt.df.String()
		if got != tt.want {
			t.Errorf("DistanceFunction(%d).String() = %q, want %q", tt.df, got, tt.want)
		}
	}
}

func TestAPI_SearchResult_Shape(t *testing.T) {
	var res SearchResult
	res.QueryID = "q"
	res.Results = []SimilarityResult{{ID: "a", Score: 0.5}}
	res.Total = 1
	if res.Total != len(res.Results) {
		t.Error("SearchResult.Total should match len(Results)")
	}
}

func TestAPI_SimilarityResult_Shape(t *testing.T) {
	var r SimilarityResult
	r.ID = "id"
	r.Score = 0.9
	r.Metadata = VectorMetadata{CreatedAt: 1, UpdatedAt: 2}
	if r.ID == "" || r.Metadata.CreatedAt != 1 {
		t.Error("SimilarityResult fields must be assignable")
	}
}

func TestAPI_Vector_Shape(t *testing.T) {
	var v Vector
	v.ID = "x"
	v.Data = []float32{1, 2}
	v.Dimension = 2
	v.Metadata = VectorMetadata{}
	if v.ID != "x" || v.Dimension != 2 {
		t.Error("Vector fields must be assignable")
	}
}

func TestAPI_MMROptions_Shape(t *testing.T) {
	opts := &MMROptions{
		Lambda:      0.5,
		FetchFactor: 5,
		ScoreMode:   MMRScoreQueryOnly,
		BlendAlpha:  0.5,
	}
	if opts.Lambda != 0.5 || opts.FetchFactor != 5 {
		t.Error("MMROptions fields must be assignable")
	}
}

func TestAPI_MMRCandidate_Shape(t *testing.T) {
	c := MMRCandidate{
		ID:        "c1",
		Embedding: []float32{1, 0},
		BaseScore: 0.8,
	}
	if c.ID != "c1" || len(c.Embedding) != 2 || c.BaseScore != 0.8 {
		t.Error("MMRCandidate fields must be assignable")
	}
}

func TestAPI_MMRScoreMode_Constants(t *testing.T) {
	var _ MMRScoreMode = MMRScoreQueryOnly
	var _ MMRScoreMode = MMRScoreBaseOnly
	var _ MMRScoreMode = MMRScoreBlend
}
