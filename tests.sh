
# External Validation Tests
cargo run --release --example correctness_test
python examples/correctness_test.py
cargo run --release --example correctness_test_periodic
python examples/correctness_test_periodic.py
cargo run --release --example correctness_test_k
python examples/correctness_test_k.py

# Internal Tests
cargo test
