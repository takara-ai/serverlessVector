<img src="https://github.com/takara-ai/serverlessVector/blob/main/media/serverlessVector.svg" width="200" alt="Takara.ai Logo" />

From the Frontier Research Team at **Takara.ai** we present a fast, in-memory vector database optimized for serverless environments. Store and search vectors with minimal setup and predictable performance.

## Installation

```bash
go get github.com/takara-ai/serverlessVector/v2@latest
```

## Quick Start

```go
package main

import (
    "fmt"
    "log"
    "github.com/takara-ai/serverlessVector/v2"
)

func main() {
    // Create database with 4 dimensions (example)
    db := serverlessVector.NewVectorDB(4)

    // Add vectors (float32 only, matches embedding APIs)
    if err := db.Add("cat", []float32{0.1, 0.3, 0.2, 0.4}); err != nil {
        log.Fatal(err)
    }
    if err := db.Add("dog", []float32{0.2, 0.4, 0.1, 0.3}); err != nil {
        log.Fatal(err)
    }

    // Search for similar vectors
    query := []float32{0.15, 0.35, 0.15, 0.25}
    results, err := db.Search(query, 5)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Found %d similar vectors\n", len(results.Results))
}
```

## API

### Creating a Database

```go
db := serverlessVector.NewVectorDB(384)                    // Fixed dimension
db := serverlessVector.NewVectorDB(384, serverlessVector.DotProduct)       // Custom distance (CosineSimilarity, DotProduct, EuclideanDistance, ManhattanDistance)
db := serverlessVector.NewVectorDB(0)                     // Flexible dimensions (no validation)
```

### Operations

```go
// Add/Update/Delete (metadata optional on Add/Update)
err := db.Add("id1", []float32{1.0, 2.0, 3.0})
err := db.Add("id1", []float32{1.0, 2.0, 3.0}, serverlessVector.VectorMetadata{Tags: map[string]string{"key": "value"}})
err := db.Update("id1", newData)
err := db.Update("id1", newData, metadata)
err := db.Delete("id1")

// Batch add (map[id]vector; metadata map optional)
err := db.BatchAdd(map[string]any{"id1": []float32{...}, "id2": []float32{...}}, nil)

// Get by ID / clear all
vec, err := db.Get("id1")
db.Clear()

// Search (topK optional, default 10)
results, err := db.Search(queryVector, 5)
results, err := db.SearchWithFilter(queryVector, 5, func(v *serverlessVector.Vector) bool { return v.Metadata.Tags["category"] == "news" })
results, err := db.BatchSearch(queries, 10)  // queries: map[queryID]queryVector

// Info
size := db.Size()
stats := db.GetStats()

// MMR: relevant but diverse results (optional; see Performance section)
results, err = db.SearchMMR(queryVector, 5)
results, err = db.SearchMMR(queryVector, 5, &serverlessVector.MMROptions{Lambda: 0.7, FetchFactor: 5})

// Advanced MMR: External scores and Candidate sets
// 1. Run MMR on a provided set of candidates (ignoring DB storage)
candidates := []serverlessVector.MMRCandidate{
    {ID: "A", Embedding: []float32{...}, BaseScore: 0.9},
    {ID: "B", Embedding: []float32{...}, BaseScore: 0.8},
}
results, err = db.SelectMMRFromCandidates(candidates, 5, &serverlessVector.MMROptions{Lambda: 0.5})

// 2. Search with external base scores (blending or overriding query similarity)
baseScores := map[string]float64{"doc1": 1.0, "doc2": 0.5}
// Blend query similarity (0.5) and base score (0.5)
results, err = db.SearchMMRWithScores(query, 5, baseScores, &serverlessVector.MMROptions{
    Lambda: 0.6,
    ScoreMode: serverlessVector.MMRScoreBlend,
    BlendAlpha: 0.5,
})
```

## Performance

Benchmarks on Apple M1. Scale roughly with n and d. For **L2-normalised** embeddings (e.g. Takara ds1-en-v1), use `DotProduct` for same ranking as cosine with less work.

**Search latency (ms)**

| n | 128D | 384D | 768D | 1536D |
|--:|-----:|-----:|-----:|------:|
| 100 | 0.05 | 0.15 | 0.3 | 0.6 |
| 1K | 0.5 | 1.5 | 3 | 6 |
| 10K | 5 | 15 | 30 | 60 |

**Memory (per vector / 10K vectors)** — 128D: 0.5KB / 27MB · 384D: 1.5KB / 76MB · 768D: 3KB / 149MB · 1536D: 6KB / 295MB

**SearchMMR** — Balances relevance and diversity. ~1.4ms (2.9x Search) at 1K vectors 128D, topK=10. `SearchMMR(query, topK)` or add `&MMROptions{Lambda: 0.7, FetchFactor: 5}` to tune.

**SelectMMRFromCandidates** — Run MMR on a provided set of candidates (ignoring DB storage). Useful for re-ranking external search results.

**SearchMMRWithScores** — Standard MMR search but with external base scores (e.g. from a keyword search or other signals). Supports `QueryOnly`, `BaseScoreOnly`, and `Blend` modes.

**Throughput (1K vectors, topK=10, cosine)** — Takara ds1-en-v1 512D (use DotProduct): ~3600/s · OpenAI 1536D: ~170/s · Sentence Transformers 384–768D: ~330–670/s · 128D: ~2000/s

## Use Cases

- Semantic search and document similarity
- Recommendation systems and content filtering
- Embedding comparison and clustering
- Serverless ML applications
- Prototyping vector operations

## Limitations

- In-memory only (no persistence)
- Linear search O(n×d) complexity
- Single-threaded operations
- Not optimized for millions of vectors

## Features

- Zero external dependencies
- Thread-safe operations
- float32 only (matches Takara ds1, OpenAI, Cohere, sentence-transformers, etc.)
- Automatic metadata inclusion (CreatedAt, UpdatedAt, Tags)
- Batch add and batch search
- Filtered search (SearchWithFilter by metadata/tags)
- Multiple distance functions (CosineSimilarity, DotProduct, EuclideanDistance, ManhattanDistance)

## License

MIT License
