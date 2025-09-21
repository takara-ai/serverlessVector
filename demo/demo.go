package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/takara-ai/serverlessVector"
)

// Simulate text embeddings for different categories
func generateEmbedding(category string, dimension int) []float32 {
	rand.Seed(int64(len(category))) // Deterministic based on category
	embedding := make([]float32, dimension)
	
	// Create different "semantic clusters" for different categories
	switch category {
	case "animals":
		embedding[0] = 0.8 + rand.Float32()*0.2 - 0.1
		embedding[1] = 0.2 + rand.Float32()*0.2 - 0.1
		embedding[2] = 0.1 + rand.Float32()*0.2 - 0.1
	case "food":
		embedding[0] = 0.2 + rand.Float32()*0.2 - 0.1
		embedding[1] = 0.8 + rand.Float32()*0.2 - 0.1
		embedding[2] = 0.3 + rand.Float32()*0.2 - 0.1
	case "technology":
		embedding[0] = 0.1 + rand.Float32()*0.2 - 0.1
		embedding[1] = 0.3 + rand.Float32()*0.2 - 0.1
		embedding[2] = 0.9 + rand.Float32()*0.2 - 0.1
	case "sports":
		embedding[0] = 0.6 + rand.Float32()*0.2 - 0.1
		embedding[1] = 0.1 + rand.Float32()*0.2 - 0.1
		embedding[2] = 0.2 + rand.Float32()*0.2 - 0.1
	default:
		for i := range embedding {
			embedding[i] = rand.Float32()
		}
		return embedding
	}
	
	// Fill remaining dimensions with small random values
	for i := 3; i < dimension; i++ {
		embedding[i] = rand.Float32()*0.1 - 0.05
	}
	
	// Normalize the vector
	norm := float32(0)
	for _, val := range embedding {
		norm += val * val
	}
	norm = float32(math.Sqrt(float64(norm)))
	
	for i := range embedding {
		embedding[i] /= norm
	}
	
	return embedding
}

func main() {
	fmt.Println("Serverless Vector Database Demo")
	fmt.Println("===============================")
	
	// Create a vector database for 128-dimensional embeddings
	db := serverlessVector.NewVectorDB(128)
	
	// Sample documents with their categories
	documents := map[string]string{
		"doc1":  "Golden retriever is a friendly dog breed",
		"doc2":  "Machine learning algorithms process data",
		"doc3":  "Pizza is a delicious Italian food",
		"doc4":  "Basketball players need good coordination",
		"doc5":  "Cats are independent pets",
		"doc6":  "JavaScript frameworks simplify web development",
		"doc7":  "Sushi is a traditional Japanese cuisine",
		"doc8":  "Tennis requires quick reflexes",
		"doc9":  "Elephants are intelligent mammals",
		"doc10": "Cloud computing revolutionizes infrastructure",
		"doc11": "Chocolate cake is a sweet dessert",
		"doc12": "Soccer is the world's most popular sport",
	}
	
	categories := map[string]string{
		"doc1": "animals", "doc2": "technology", "doc3": "food", "doc4": "sports",
		"doc5": "animals", "doc6": "technology", "doc7": "food", "doc8": "sports", 
		"doc9": "animals", "doc10": "technology", "doc11": "food", "doc12": "sports",
	}
	
	fmt.Printf("Adding %d documents to the database...\n", len(documents))
	
	// Add all documents with their embeddings and metadata
	start := time.Now()
	for docID, text := range documents {
		category := categories[docID]
		embedding := generateEmbedding(category, 128)
		
		metadata := serverlessVector.VectorMetadata{
			Tags: map[string]string{
				"category": category,
				"text":     text,
			},
		}
		
		err := db.Add(docID, embedding, metadata)
		if err != nil {
			log.Fatalf("Failed to add document %s: %v", docID, err)
		}
	}
	addTime := time.Since(start)
	
	fmt.Printf("Added all documents in %v\n\n", addTime)
	
	// Show database stats
	stats := db.GetStats()
	fmt.Printf("Database Statistics:\n")
	fmt.Printf("   Total vectors: %d\n", stats["total_vectors"])
	fmt.Printf("   Dimension: %d\n", stats["dimension"])
	fmt.Printf("   Memory usage: %d KB\n", stats["memory_usage_kb"])
	fmt.Printf("   Distance function: %s\n\n", stats["distance_function"])
	
	// Test different types of queries
	queries := []struct {
		name     string
		category string
		text     string
	}{
		{"Animal Query", "animals", "What about dogs and cats?"},
		{"Food Query", "food", "I'm hungry for something tasty"},
		{"Tech Query", "technology", "Software and programming topics"},
		{"Sports Query", "sports", "Athletic activities and games"},
	}
	
	fmt.Println("Testing Semantic Search:")
	fmt.Println("========================")
	
	for _, query := range queries {
		fmt.Printf("\nQuery: \"%s\" (%s)\n", query.text, query.name)
		
		// Generate query embedding
		queryEmbedding := generateEmbedding(query.category, 128)
		
		// Search for similar documents
		searchStart := time.Now()
		results, err := db.Search(queryEmbedding, 3)
		searchTime := time.Since(searchStart)
		
		if err != nil {
			log.Printf("Search failed: %v", err)
			continue
		}
		
		fmt.Printf("   Found %d results in %v:\n", len(results.Results), searchTime)
		
		for i, result := range results.Results {
			text := result.Metadata.Tags["text"]
			category := result.Metadata.Tags["category"]
			
			fmt.Printf("   %d. [%s] %s\n", i+1, category, text)
			fmt.Printf("      Similarity: %.3f\n", result.Score)
		}
	}
	
	// Demonstrate batch search
	fmt.Println("\nTesting Batch Search:")
	fmt.Println("=====================")
	
	batchQueries := map[string]interface{}{
		"multi_animal": generateEmbedding("animals", 128),
		"multi_tech":   generateEmbedding("technology", 128),
		"multi_food":   generateEmbedding("food", 128),
	}
	
	batchStart := time.Now()
	batchResults, err := db.BatchSearch(batchQueries, 2)
	batchTime := time.Since(batchStart)
	
	if err != nil {
		log.Printf("Batch search failed: %v", err)
	} else {
		fmt.Printf("Batch search completed in %v\n", batchTime)
		for queryID, result := range batchResults {
			fmt.Printf("Query %s: %d results\n", queryID, len(result.Results))
		}
	}
	
	// Performance test
	fmt.Println("\nPerformance Test:")
	fmt.Println("=================")
	
	testQuery := generateEmbedding("animals", 128)
	numSearches := 1000
	
	perfStart := time.Now()
	for i := 0; i < numSearches; i++ {
		_, err := db.Search(testQuery, 5)
		if err != nil {
			log.Printf("Performance test search %d failed: %v", i, err)
			break
		}
	}
	perfTime := time.Since(perfStart)
	
	avgTime := perfTime / time.Duration(numSearches)
	searchesPerSecond := float64(numSearches) / perfTime.Seconds()
	
	fmt.Printf("Completed %d searches in %v\n", numSearches, perfTime)
	fmt.Printf("Average search time: %v\n", avgTime)
	fmt.Printf("Searches per second: %.0f\n", searchesPerSecond)
	
	fmt.Println("\nDemo Summary:")
	fmt.Println("=============")
	fmt.Println("- Store high-dimensional vectors efficiently")
	fmt.Println("- Fast semantic similarity search")
	fmt.Println("- Batch operations support")
	fmt.Println("- Good performance characteristics")
	fmt.Println("- Suitable for serverless environments")
}