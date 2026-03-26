.PHONY: help test-metadata test-vertex-batch-results

help:
	@echo "Targets:"
	@echo "  make test-metadata              - cargo tests with names matching 'metadata' (unit + integration)"
	@echo "  make test-vertex-batch-results  - slow Vertex batch test (sets VERTEX_BATCH_WAIT_RESULTS=1)"
	@echo ""
	@echo "Optional env for Vertex batch results test:"
	@echo "  VERTEX_BATCH_WAIT_RESULTS=1     - required by test_vertex_batch_metadata_get_results_echoed_labels"
	@echo "  VERTEX_PROJECT_ID, VERTEX_LOCATION, VERTEX_ACCESS_TOKEN, VERTEX_BATCH_BUCKET"

test-metadata:
	cargo test --features integration metadata

test-vertex-batch-results:
	VERTEX_BATCH_WAIT_RESULTS=1 cargo test --features integration test_vertex_batch_metadata_get_results_echoed_labels -- --test-threads=1
