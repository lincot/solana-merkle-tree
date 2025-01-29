# basic Merkle tree Solana program

## testing

```sh
cd program
cargo test-sbf
```

## running the client

```sh
cd program
cargo build-sbf
solana-test-validator --reset --bpf-program target/deploy/basic_merkle_tree-keypair.json target/deploy/basic_merkle_tree.so

cd ../client
SOLANA_RPC=http://127.0.0.1:8899 cargo run 52C9T2T7JRojtxumYnYZhyUmrN7kqzvCLc4Ksvjk7TxD
```
