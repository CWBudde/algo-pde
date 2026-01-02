# algo-pde justfile

# Default recipe
default: test

# Run all tests
test:
    go test ./...

# Run tests with verbose output
test-v:
    go test -v ./...

# Run tests with race detector
test-race:
    go test -race ./...

# Run tests with coverage
test-cov:
    go test -coverprofile=coverage.txt ./...
    go tool cover -html=coverage.txt -o coverage.html
    @echo "Coverage report: coverage.html"

# Run benchmarks
bench:
    go test -bench=. -benchmem ./...

# Run benchmarks for a specific package
bench-pkg pkg:
    go test -bench=. -benchmem ./{{pkg}}/...

# Run linter
lint:
    golangci-lint run

# Run linter and fix issues
lint-fix:
    golangci-lint run --fix

# Format code using treefmt
fmt:
    treefmt . --allow-missing-formatter

# Check if code is formatted
fmt-check:
    treefmt --allow-missing-formatter --fail-on-change

# Tidy dependencies
tidy:
    go mod tidy

# Build (check compilation)
build:
    go build ./...

# Clean build artifacts
clean:
    rm -f coverage.txt coverage.html
    go clean ./...

# Run all checks (lint, test, build)
check: lint test build
    @echo "All checks passed!"

# Generate documentation
doc:
    @echo "Opening documentation in browser..."
    go doc -all ./... | less

# Watch for changes and run tests (requires entr)
watch:
    find . -name '*.go' | entr -c go test ./...

# Profile CPU for benchmarks
profile-cpu pkg bench:
    go test -bench={{bench}} -cpuprofile=cpu.pprof ./{{pkg}}
    go tool pprof -http=:8080 cpu.pprof

# Profile memory for benchmarks
profile-mem pkg bench:
    go test -bench={{bench}} -memprofile=mem.pprof ./{{pkg}}
    go tool pprof -http=:8080 mem.pprof
