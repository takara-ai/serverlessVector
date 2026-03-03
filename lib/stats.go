package lib

// GetStats returns database statistics.
// It snapshots under RLock then computes stats outside the lock to reduce lock hold time.
func (db *VectorDB) GetStats() map[string]any {
	db.mu.RLock()
	totalVectors := len(db.vectors)
	totalDimensions := 0
	for _, vector := range db.vectors {
		totalDimensions += vector.Dimension
	}
	distFunc := db.distFunc
	dimension := db.dimension
	db.mu.RUnlock()

	avgDimensions := 0.0
	if totalVectors > 0 {
		avgDimensions = float64(totalDimensions) / float64(totalVectors)
	}
	// float32: 4 bytes per dimension + per-vector overhead
	memoryUsage := int64(totalDimensions)*4 + int64(totalVectors)*256

	return map[string]any{
		"total_vectors":     totalVectors,
		"total_dimensions":  totalDimensions,
		"avg_dimensions":    avgDimensions,
		"memory_usage_kb":   memoryUsage / 1024,
		"distance_function": distFunc.String(),
		"dimension":         dimension,
	}
}
