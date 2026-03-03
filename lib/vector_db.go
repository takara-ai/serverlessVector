package lib

import (
	"errors"
	"fmt"
	"maps"
	"sync"
	"time"
)

// VectorDB is a simple, fast vector database for serverless applications
type VectorDB struct {
	mu        sync.RWMutex
	vectors   map[string]*Vector
	dimension int
	distFunc  DistanceFunction
}

// NewVectorDB creates a new vector database
// dimension: vector dimension (e.g., 384 for OpenAI, 1536 for text-embedding-ada-002)
//
//	use 0 for no dimension validation (flexible dimensions)
//
// distanceFunc: optional distance function (defaults to CosineSimilarity if not provided)
func NewVectorDB(dimension int, distanceFunc ...DistanceFunction) *VectorDB {
	if dimension < 0 {
		panic("dimension must be >= 0 (use 0 for no validation)")
	}

	df := CosineSimilarity // smart default for embeddings
	if len(distanceFunc) > 0 {
		df = distanceFunc[0]
	}

	return &VectorDB{
		vectors:   make(map[string]*Vector),
		dimension: dimension,
		distFunc:  df,
	}
}

// Add adds a vector to the database. data must be []float32 (matches embedding APIs).
func (db *VectorDB) Add(id string, data any, metadata ...VectorMetadata) error {
	if id == "" {
		return errors.New("vector ID cannot be empty")
	}
	vec, dim, err := copyFloat32Slice(data)
	if err != nil {
		return err
	}
	if dim == 0 {
		return errors.New("vector data cannot be empty")
	}
	if db.dimension > 0 && dim != db.dimension {
		return fmt.Errorf("vector dimension %d does not match expected %d", dim, db.dimension)
	}
	db.mu.Lock()
	defer db.mu.Unlock()
	now := time.Now().Unix()
	vector := &Vector{ID: id, Data: vec, Dimension: dim}
	if len(metadata) > 0 {
		vector.Metadata = metadata[0]
		vector.Metadata.CreatedAt = now
		vector.Metadata.UpdatedAt = now
	} else {
		vector.Metadata = VectorMetadata{CreatedAt: now, UpdatedAt: now}
	}
	db.vectors[id] = vector
	return nil
}

// Get retrieves a vector by ID
func (db *VectorDB) Get(id string) (*Vector, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	vector, exists := db.vectors[id]
	if !exists {
		return nil, fmt.Errorf("vector with ID %s not found", id)
	}

	dataCopy := make([]float32, vector.Dimension)
	copy(dataCopy, vector.Data)
	return &Vector{
		ID:        vector.ID,
		Data:      dataCopy,
		Metadata:  vector.Metadata,
		Dimension: vector.Dimension,
	}, nil
}

// Update updates an existing vector. data must be []float32.
func (db *VectorDB) Update(id string, data any, metadata ...VectorMetadata) error {
	if id == "" {
		return errors.New("vector ID cannot be empty")
	}
	vec, dim, err := copyFloat32Slice(data)
	if err != nil {
		return err
	}
	if db.dimension > 0 && dim != db.dimension {
		return fmt.Errorf("vector dimension %d does not match expected %d", dim, db.dimension)
	}
	db.mu.Lock()
	defer db.mu.Unlock()
	vector, exists := db.vectors[id]
	if !exists {
		return fmt.Errorf("vector with ID %s not found", id)
	}
	vector.Data = vec
	vector.Dimension = dim
	now := time.Now().Unix()
	if len(metadata) > 0 {
		vector.Metadata = metadata[0]
		vector.Metadata.UpdatedAt = now
	} else {
		vector.Metadata.UpdatedAt = now
	}
	return nil
}

// Delete removes a vector from the database
func (db *VectorDB) Delete(id string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if _, exists := db.vectors[id]; !exists {
		return fmt.Errorf("vector with ID %s not found", id)
	}

	delete(db.vectors, id)
	return nil
}

// Size returns the number of vectors in the database
func (db *VectorDB) Size() int {
	db.mu.RLock()
	defer db.mu.RUnlock()
	return len(db.vectors)
}

// Clear removes all vectors from the database
func (db *VectorDB) Clear() {
	db.mu.Lock()
	defer db.mu.Unlock()
	db.vectors = make(map[string]*Vector)
}

// BatchAdd adds multiple vectors efficiently in a single operation.
// New vectors are built outside the lock; the write lock is held only for the map merge,
// so tail latencies for concurrent readers are not raised by long write lock duration.
func (db *VectorDB) BatchAdd(vectors map[string]any, metadata map[string]VectorMetadata) error {
	if len(vectors) == 0 {
		return errors.New("no vectors provided")
	}

	now := time.Now().Unix()
	batchMap := make(map[string]*Vector, len(vectors))

	for id, data := range vectors {
		if id == "" {
			return errors.New("vector ID cannot be empty")
		}
		vec, dim, err := copyFloat32Slice(data)
		if err != nil {
			return fmt.Errorf("unsupported vector type for %s: %T (use []float32)", id, data)
		}
		if db.dimension > 0 && dim != db.dimension {
			return fmt.Errorf("vector %s dimension %d does not match expected %d", id, dim, db.dimension)
		}
		vector := &Vector{
			ID:        id,
			Data:      vec,
			Dimension: dim,
			Metadata:  VectorMetadata{CreatedAt: now, UpdatedAt: now},
		}
		if meta, exists := metadata[id]; exists {
			vector.Metadata = meta
			vector.Metadata.CreatedAt = now
			vector.Metadata.UpdatedAt = now
		}
		batchMap[id] = vector
	}

	db.mu.Lock()
	newMap := make(map[string]*Vector, len(db.vectors)+len(batchMap))
	maps.Copy(newMap, db.vectors)
	maps.Copy(newMap, batchMap)
	db.vectors = newMap
	db.mu.Unlock()

	return nil
}
