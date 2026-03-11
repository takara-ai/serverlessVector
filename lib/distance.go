package lib

import "math"

func dotProduct32(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var sum float64
	n := len(a)
	for i := 0; i <= n-8; i += 8 {
		sum += float64(a[i])*float64(b[i]) + float64(a[i+1])*float64(b[i+1]) +
			float64(a[i+2])*float64(b[i+2]) + float64(a[i+3])*float64(b[i+3]) +
			float64(a[i+4])*float64(b[i+4]) + float64(a[i+5])*float64(b[i+5]) +
			float64(a[i+6])*float64(b[i+6]) + float64(a[i+7])*float64(b[i+7])
	}
	for i := (n / 8) * 8; i < n; i++ {
		sum += float64(a[i]) * float64(b[i])
	}
	return sum
}

func norm32(a []float32) float64 {
	return math.Sqrt(dotProduct32(a, a))
}

// DistanceFloat32 computes the distance/similarity between two vectors.
func DistanceFloat32(a, b []float32, distanceFunc DistanceFunction) float64 {
	switch distanceFunc {
	case CosineSimilarity:
		dot := dotProduct32(a, b)
		na, nb := norm32(a), norm32(b)
		if na == 0 || nb == 0 {
			return 0
		}
		return dot / (na * nb)
	case DotProduct:
		return dotProduct32(a, b)
	case EuclideanDistance:
		return euclidean32(a, b)
	case ManhattanDistance:
		return manhattan32(a, b)
	default:
		return dotProduct32(a, b)
	}
}

func (db *VectorDB) distanceFloat32(a, b []float32, distanceFunc DistanceFunction) float64 {
	return DistanceFloat32(a, b, distanceFunc)
}

func sameLen32(a, b []float32) bool { return len(a) == len(b) }

func euclidean32(a, b []float32) float64 {
	if !sameLen32(a, b) {
		return math.Inf(1)
	}
	var sum float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		sum += d * d
	}
	return math.Sqrt(sum)
}

func manhattan32(a, b []float32) float64 {
	if !sameLen32(a, b) {
		return math.Inf(1)
	}
	var sum float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		if d < 0 {
			d = -d
		}
		sum += d
	}
	return sum
}
