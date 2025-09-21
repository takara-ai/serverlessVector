# Serverless Vector Database

A fast, in-memory vector database optimized for serverless environments. Store and search vectors with minimal setup.

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
    results, err := db.Search(query, 5) // top 5 results
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Found %d similar vectors\n", len(results.Results))
}
```

## API

### Creating a Database

```go
// Fixed dimension (most common)
db := NewVectorDB(384)

// With custom distance function
db := NewVectorDB(384, DotProduct)

// Flexible dimensions (no validation)
db := NewVectorDB(0)
```

### Adding Vectors

```go
// Single vector
err := db.Add("id1", []float32{1.0, 2.0, 3.0})

// With metadata
metadata := VectorMetadata{Tags: map[string]string{"type": "embedding"}}
err := db.Add("id2", []float64{1.0, 2.0, 3.0}, metadata)

// Batch add
vectors := map[string]interface{}{
    "vec1": []float32{1.0, 2.0, 3.0},
    "vec2": []float32{4.0, 5.0, 6.0},
}
err := db.BatchAdd(vectors, nil)
```

### Searching

```go
// Basic search (returns top 10)
results, err := db.Search(queryVector)

// Specify number of results
results, err := db.Search(queryVector, 5)

// Batch search multiple queries
queries := map[string]interface{}{
    "q1": []float32{1.0, 2.0, 3.0},
    "q2": []float32{4.0, 5.0, 6.0},
}
results, err := db.BatchSearch(queries, 10)
```

### Other Operations

```go
// Get vector by ID
vector, err := db.Get("id1")

// Update vector
err := db.Update("id1", newData, newMetadata)

// Delete vector
err := db.Delete("id1")

// Database info
size := db.Size()
stats := db.GetStats()
```

## Distance Functions

- `CosineSimilarity` - Default, best for embeddings
- `DotProduct` - Dot product similarity  
- `EuclideanDistance` - Euclidean distance
- `ManhattanDistance` - Manhattan distance

## Supported Vector Types

- `[]float32` - Recommended for embeddings
- `[]float64` - Higher precision

## Use Cases

- Serverless functions (Lambda, Vercel, Netlify)
- Edge computing and microservices
- Embedding similarity search
- Caching layer for vector operations
- Prototyping and experimentation

## Performance

- Memory usage: O(n×d) where n = vectors, d = dimension
- Search: O(n×d) linear search
- Thread-safe with read/write mutexes
- No external dependencies

## Limitations

- In-memory only (no persistence)
- Linear search (not optimized for millions of vectors)
- No indexing for large datasets

## License

MIT License