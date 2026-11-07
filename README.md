# Atari Train DQN

Deep-RL training and evaluation using ALE environments and DQN-based agents for Atari games:

* Boxing
* Breakout
* Ice Hockey
* Pong
* Space Invaders
* Tennis

## Install

```sh
chmod +x crates/ale-bridge/native/setup.sh
./crates/ale-bridge/native/setup.sh
```

## Build

```sh
cargo build --locked --release
```

## Run

```sh
cargo run --locked --release -- [OPTIONS]
```

#### Train

```sh
cargo run --locked --release -- train --game <game>
```

#### Train with Config

```bash
cargo run --locked --release -- train --config configs/<game>.json
```

#### Train with GPU

```bash
cargo run --locked --release --features gpu -- train [OPTIONS] --device gpu
```

## Evaluate

```bash
cargo run --locked --release -- evaluate [OPTIONS]
```

## Contribute

Before making a Pull Request, ensure it addresses an Issue, and verify:

```bash
cargo fmt --all
cargo check --locked --workspace --all-features
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
```
