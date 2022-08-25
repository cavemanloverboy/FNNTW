
# External Validation Tests
cargo run --release --example correctness_test --quiet
python examples/correctness_test.py
cargo run --release --example correctness_test_periodic --quiet
python examples/correctness_test_periodic.py
cargo run --release --example correctness_test_k --quiet
python examples/correctness_test_k.py

# Internal Tests
cargo test
