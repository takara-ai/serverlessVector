package lib

import (
	"errors"
	"fmt"
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
//           use 0 for no dimension validation (flexible dimensions)
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

// Add adds a vector to the database
// id: unique identifier for the vector
// data: vector data as []float32 or []float64
// metadata: optional metadata
func (db *VectorDB) Add(id string, data interface{}, metadata ...VectorMetadata) error {
	if id == "" {
		return errors.New("vector ID cannot be empty")
	}

	dataSlice, vecType, err := convertToInterfaceSlice(data)
	if err != nil {
		return err
	}

	if len(dataSlice) == 0 {
		return errors.New("vector data cannot be empty")
	}

	if db.dimension > 0 && len(dataSlice) != db.dimension {
		return fmt.Errorf("vector dimension %d does not match expected %d", len(dataSlice), db.dimension)
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	vector := &Vector{
		ID:   id,
		Data: make([]interface{}, len(dataSlice)),
		Type: vecType,
	}

	copy(vector.Data, dataSlice)

	// Add metadata if provided
	now := time.Now().Unix()
	if len(metadata) > 0 {
		vector.Metadata = metadata[0]
		vector.Metadata.CreatedAt = now
		vector.Metadata.UpdatedAt = now
	} else {
		vector.Metadata = VectorMetadata{
			CreatedAt: now,
			UpdatedAt: now,
		}
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

	// Return a copy to prevent external modification
	result := &Vector{
		ID:       vector.ID,
		Data:     make([]interface{}, len(vector.Data)),
		Type:     vector.Type,
		Metadata: vector.Metadata,
	}
	copy(result.Data, vector.Data)
	return result, nil
}

// Update updates an existing vector
func (db *VectorDB) Update(id string, data interface{}, metadata ...VectorMetadata) error {
	if id == "" {
		return errors.New("vector ID cannot be empty")
	}

	dataSlice, vecType, err := convertToInterfaceSlice(data)
	if err != nil {
		return err
	}

	if db.dimension > 0 && len(dataSlice) != db.dimension {
		return fmt.Errorf("vector dimension %d does not match expected %d", len(dataSlice), db.dimension)
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	vector, exists := db.vectors[id]
	if !exists {
		return fmt.Errorf("vector with ID %s not found", id)
	}

	// Update data
	vector.Data = make([]interface{}, len(dataSlice))
	copy(vector.Data, dataSlice)
	vector.Type = vecType

	// Update metadata
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


// BatchAdd adds multiple vectors efficiently in a single operation
func (db *VectorDB) BatchAdd(vectors map[string]interface{}, metadata map[string]VectorMetadata) error {
	if len(vectors) == 0 {
		return errors.New("no vectors provided")
	}

	// Validate all vectors first
	for id, data := range vectors {
		if id == "" {
			return errors.New("vector ID cannot be empty")
		}

		dataSlice, _, err := convertToInterfaceSlice(data)
		if err != nil {
			return fmt.Errorf("unsupported vector type for %s: %T", id, data)
		}

		if db.dimension > 0 && len(dataSlice) != db.dimension {
			return fmt.Errorf("vector %s dimension %d does not match expected %d", id, len(dataSlice), db.dimension)
		}
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	now := time.Now().Unix()
	for id, data := range vectors {
		dataSlice, vecType, _ := convertToInterfaceSlice(data)

		vector := &Vector{
			ID:   id,
			Data: make([]interface{}, len(dataSlice)),
			Type: vecType,
			Metadata: VectorMetadata{
				CreatedAt: now,
				UpdatedAt: now,
			},
		}

		copy(vector.Data, dataSlice)

		// Add metadata if provided
		if meta, exists := metadata[id]; exists {
			vector.Metadata = meta
			vector.Metadata.CreatedAt = now
			vector.Metadata.UpdatedAt = now
		}

		db.vectors[id] = vector
	}

	return nil
}