package lib

// statsSnapshotEntry holds minimal per-vector data for stats computation outside the lock.
type statsSnapshotEntry struct {
	dimLen int
	vecType VectorType
}

// GetStats returns database statistics.
// It snapshots under RLock then computes stats outside the lock to reduce lock hold time.
func (db *VectorDB) GetStats() map[string]interface{} {
	db.mu.RLock()
	totalVectors := len(db.vectors)
	snapshot := make([]statsSnapshotEntry, 0, totalVectors)
	for _, vector := range db.vectors {
		snapshot = append(snapshot, statsSnapshotEntry{dimLen: len(vector.Data), vecType: vector.Type})
	}
	distFunc := db.distFunc
	dimension := db.dimension
	db.mu.RUnlock()

	totalDimensions := 0
	memoryUsage := int64(0)
	float32Count := 0
	float64Count := 0

	for _, entry := range snapshot {
		totalDimensions += entry.dimLen
		var typeSize int
		switch entry.vecType {
		case Float32:
			typeSize = 4
			float32Count++
		case Float64:
			typeSize = 8
			float64Count++
		}
		vectorMem := int64(16*entry.dimLen + typeSize*entry.dimLen + 256)
		memoryUsage += vectorMem
		if memoryUsage == 0 && totalVectors > 0 {
			memoryUsage = 1
		}
	}

	avgDimensions := float64(0)
	if totalVectors > 0 {
		avgDimensions = float64(totalDimensions) / float64(totalVectors)
	}

	return map[string]interface{}{
		"total_vectors":     totalVectors,
		"total_dimensions":  totalDimensions,
		"avg_dimensions":    avgDimensions,
		"memory_usage_kb":   int64(float64(memoryUsage) / 1024.0),
		"float32_count":     float32Count,
		"float64_count":     float64Count,
		"distance_function": distFunc.String(),
		"dimension":         dimension,
	}
}
