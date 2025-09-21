<img src="https://takara.ai/images/logo-24/TakaraAi.svg" width="200" alt="Takara.ai Logo" />

From the Frontier Research Team at **Takara.ai** we present a fast, in-memory vector database optimized for serverless environments. Store and search vectors with minimal setup and predictable performance.

## Installation

```bash
go get github.com/takarajordan/serverlessVector
```

## Quick Start

```go
package main

import (
    "fmt"
    "log"
    "github.com/takarajordan/serverlessVector"
)

func main() {
    // Create database with dimension 384 (OpenAI embeddings)
    db := serverlessVector.NewVectorDB(384)

    // Add vectors
    db.Add("cat", []float32{0.1, 0.3, 0.2, 0.4})
    db.Add("dog", []float64{0.2, 0.4, 0.1, 0.3})

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
db := NewVectorDB(384)                    // Fixed dimension
db := NewVectorDB(384, DotProduct)       // Custom distance function
db := NewVectorDB(0)                     // Flexible dimensions
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

// Info
size := db.Size()
stats := db.GetStats()
```

## Performance

### Search Performance (Linear Scaling)

| Vectors | 128D | 384D | 768D | 1536D |
|---------|------|------|------|-------|
| 100     | 1ms  | 3ms  | 7ms  | 13ms  |
| 1,000   | 11ms | 33ms | 66ms | 129ms |
| 10,000  | 117ms| 326ms| 653ms| 1.3s  |

### Distance Functions

| Function | Performance | Best For |
|----------|-------------|----------|
| DotProduct | 0.4ms | Speed |
| EuclideanDistance | 0.9ms | Geometric distance |
| ManhattanDistance | 1.0ms | Sparse vectors |
| CosineSimilarity | 1.1ms | Embeddings (default) |

### Memory Usage

| Dimension | Per Vector | 10K Vectors |
|-----------|------------|-------------|
| 128       | 0.5KB      | 27MB        |
| 384       | 1.5KB      | 76MB        |
| 768       | 3.0KB      | 149MB       |
| 1536      | 6.0KB      | 295MB       |

## Common Embedding Dimensions

| Model | Dimensions | Performance |
|-------|------------|-------------|
| OpenAI ada-002 | 1536 | 75 searches/sec |
| OpenAI text-embedding-3-small | 1536 | 75 searches/sec |
| Sentence Transformers | 384-768 | 150-300 searches/sec |
| BERT | 768 | 150 searches/sec |

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
- Support for float32/float64
- Automatic metadata inclusion
- Batch operations
- Multiple distance functions

## License

MIT License