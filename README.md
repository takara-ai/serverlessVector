<img src="https://github.com/takara-ai/serverlessVector/blob/main/media/serverlessVector.svg" width="200" alt="Takara.ai Logo" />

From the Frontier Research Team at **Takara.ai** we present a fast, in-memory vector database optimized for serverless environments. Store and search vectors with minimal setup and predictable performance.

## Installation

```bash
go get github.com/takara-ai/serverlessVector
```

## Quick Start

```go
package main

import (
    "fmt"
    "log"
    "github.com/takara-ai/serverlessVector"
)

func main() {
    // Create database with 4 dimensions (example)
    db := serverlessVector.NewVectorDB(4)

    // Add vectors (float32 only, matches embedding APIs)
    db.Add("cat", []float32{0.1, 0.3, 0.2, 0.4})
    db.Add("dog", []float32{0.2, 0.4, 0.1, 0.3})

    // Search for similar vectors
    query := []float32{0.15, 0.35, 0.15, 0.25}
    results, err := db.Search(query, 5)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Found %d similar vectors\n", len(results.Results))

    // Or MMR: relevant but diverse results
    results, err = db.SearchMMR(query, 5)
}
```

## API

### Creating a Database

```go
db := serverlessVector.NewVectorDB(384)                    // Fixed dimension
db := serverlessVector.NewVectorDB(384, serverlessVector.DotProduct)       // Custom distance function
db := serverlessVector.NewVectorDB(0)                     // Flexible dimensions
```

### Operations

```go
// Add/Update/Delete
err := db.Add("id1", []float32{1.0, 2.0, 3.0})
err := db.Update("id1", newData, metadata)
err := db.Delete("id1")

// Search
results, err := db.Search(queryVector, 5)
results, err := db.BatchSearch(queries, 10)

// MMR: relevant but diverse results (defaults) or with options
results, err = db.SearchMMR(queryVector, 5)
results, err = db.SearchMMR(queryVector, 5, &serverlessVector.MMROptions{Lambda: 0.7})

// Info
size := db.Size()
stats := db.GetStats()
```

## Performance

### SearchMMR (Maximal Marginal Relevance)

MMR balances relevance to the query with diversity among results. Same 1000 vectors, 128D, topK=10:

| Variant | Time per call | Relative to Search |
|---------|---------------|--------------------|
| Search | ~0.5ms | 1x |
| SearchMMR (lambda=0.6) | ~1.4ms | ~2.9x |

Use `SearchMMR(query, topK)` for defaults; add `&MMROptions{Lambda: 0.7}` as third arg to tune.

### Search Performance (Linear Scaling)

Approximate times on Apple M1 (1k vectors 128D ~0.5ms). Scale roughly with n and d:

| Vectors | 128D | 384D | 768D | 1536D |
|---------|------|------|------|-------|
| 100     | 0.05ms | 0.15ms | 0.3ms | 0.6ms |
| 1,000   | 0.5ms | 1.5ms | 3ms | 6ms |
| 10,000  | 5ms | 15ms | 30ms | 60ms |

### Distance Functions

All use the same float32 path. CosineSimilarity (default) is typical; others are similar. For **L2-normalised** embeddings (e.g. Takara ds1-en-v1), use `DotProduct` for the same ranking as cosine with less work.

### Memory Usage

| Dimension | Per Vector | 10K Vectors |
|-----------|------------|-------------|
| 128       | 0.5KB      | 27MB        |
| 384       | 1.5KB      | 76MB        |
| 768       | 3.0KB      | 149MB       |
| 1536      | 6.0KB      | 295MB       |

## Common Embedding Dimensions

All throughput numbers below are for **cosine similarity** (default) with 1k vectors, topK=10.

| Model | Dimensions | Approx throughput (1k vectors) |
|-------|------------|-------------------------------|
| OpenAI ada-002 | 1536 | ~170 searches/s |
| OpenAI text-embedding-3-small | 1536 | ~170 searches/s |
| Sentence Transformers | 384-768 | ~330-670 searches/s |
| Takara ds1-en-v1 | 512 | ~3600 (use `DotProduct`; embeddings are L2-normalised) |
| 128D embeddings | 128 | ~2000 searches/s |

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
- float32 only (matches OpenAI, Cohere, sentence-transformers, etc.)
- Automatic metadata inclusion
- Batch operations
- Multiple distance functions

## License

MIT License
