package main

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
)

const (
	numIndexes     = 10401 //401 // Number of image indexes (0 to 10401)
	numInjections  = 50    // Number of injections (0 to 100)
	maxGoroutines  = 128   // Maximum number of parallel goroutines
	outputFileName = "lenet_fault_injection_results.txt"
)

type Result struct {
	index  int
	output string
}

func runLenet(index int, injection int) (string, error) {
	// Run the command ./int-lenet with the specified index and injection
	cmd := exec.Command("./int-lenet", strconv.Itoa(index), strconv.Itoa(injection))
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return string(out), nil
}

func processIndex(index int, wg *sync.WaitGroup, results chan<- Result) {
	defer wg.Done()
	fmt.Printf("In index %d", index)

	// Store result for the current index
	var output string

	// Get the original result (0 injections)
	original, err := runLenet(index, 0)
	if err != nil {
		fmt.Printf("Error processing index %d, injection 0: %v\n", index, err)
		return
	}
	output += strings.Trim(original, "\n ")

	// Run with different numbers of fault injections (1 to 100)
	for i := 1; i <= numInjections; i++ {
		result, err := runLenet(index, i)
		if err != nil {
			fmt.Printf("Error processing index %d, injection %d: %v\n", index, i, err)
			return
		}
		output += strings.Trim(result, "\n")
	}

	// Send the result for the current index to the results channel
	results <- Result{index: index, output: output}
}

func writeResults(results []string) error {
	// Open the output file
	file, err := os.Create(outputFileName)
	if err != nil {
		return err
	}
	defer file.Close()

	// Write the results in order
	for _, result := range results {
		_, err := file.WriteString(result + "\n")
		if err != nil {
			return err
		}
	}

	return nil
}

func main() {
	// Create a channel to receive results from goroutines
	resultsChan := make(chan Result, numIndexes)
	var wg sync.WaitGroup

	// Limit the number of concurrent goroutines
	sem := make(chan struct{}, maxGoroutines)

	// Start the goroutines
	for i := 0; i <= numIndexes; i++ {
		sem <- struct{}{} // Block if there are maxGoroutines active goroutines
		wg.Add(1)
		go func(index int) {
			defer func() { <-sem }()
			processIndex(index, &wg, resultsChan)
		}(i)
	}

	// Wait for all goroutines to complete
	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	// Prepare a slice to store results in the original order
	results := make([]string, numIndexes+1)

	// Collect results from the resultsChan
	for result := range resultsChan {
		results[result.index] = strconv.Itoa(result.index) + "" + result.output
	}

	// Write the results to the output file
	err := writeResults(results)
	if err != nil {
		fmt.Printf("Error writing results to file: %v\n", err)
		return
	}

	fmt.Println("Testing completed. Results saved in", outputFileName)
}
