use anyhow::{Context, Result, ensure};
use burn::prelude::*;
use clap::{Parser, Subcommand, ValueEnum};
use atari_train_dqn::{
    checkpoint::{self, Metadata},
    config::{self, FRAME_HEIGHT, FRAME_STACK, FRAME_WIDTH, Padding, RunConfig},
    evaluate::Evaluation,
    model::{DQN, Weights},
    preprocess::FrameHistory,
    replay::ReplayMemory,
    state::State,
    train::TrainingResult,
};
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

const PRESET_DIRECTORY: &str = "configs";

#[derive(Parser)]
#[command(
    version,
    about = "Deep-RL training and evaluation using ALE environments and DQN-based agents for Atari games."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, ValueEnum)]
enum Backend {
    Cpu,
    Gpu,
}

impl Backend {
    fn device(self) -> Result<Device> {
        match self {
            Self::Cpu => Ok(Device::flex()),
            Self::Gpu => {
                #[cfg(feature = "gpu")]
                {
                    Ok(Device::cuda(0))
                }
                #[cfg(not(feature = "gpu"))]
                {
                    anyhow::bail!(
                        "GPU support is optional: cargo run --release --features gpu -- ... --device gpu"
                    )
                }
            }
        }
    }
}

#[derive(Subcommand)]
enum Command {
    List,
    Preset {
        #[arg(
            long,
            required_unless_present = "checkpoint",
            conflicts_with = "checkpoint"
        )]
        game: Option<String>,
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    Train {
        #[arg(long, conflicts_with_all = ["config", "all"])]
        game: Option<String>,
        #[arg(long, conflicts_with_all = ["game", "all"])]
        config: Option<PathBuf>,
        #[arg(long)]
        all: bool,
        #[arg(long)]
        steps: Option<u64>,
        #[arg(long)]
        envs: Option<usize>,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long)]
        batch_size: Option<usize>,
        #[arg(long)]
        learning_starts: Option<usize>,
        #[arg(long)]
        eval_steps: Option<u64>,
        #[arg(long)]
        eval_episodes: Option<usize>,
        #[arg(long, default_value = "runs")]
        out: PathBuf,
        #[arg(long, value_enum, default_value = "cpu")]
        device: Backend,
        #[arg(long)]
        init: Option<PathBuf>,
    },
    Evaluate {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 5)]
        episodes: usize,
        #[arg(long, default_value_t = 27_000)]
        max_steps: u64,
        #[arg(long, value_enum, default_value = "cpu")]
        device: Backend,
        #[arg(long)]
        report: Option<PathBuf>,
    },
    Export {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        q_values: bool,
    },
    Import {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Inspect {
        #[arg(long)]
        checkpoint: PathBuf,
    },
    Predict {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    ReplayBenchmark {
        #[arg(long, default_value_t = 10_000)]
        transitions: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

fn write_json(path: Option<&Path>, val: &impl serde::Serialize) -> Result<()> {
    let bytes: Vec<u8> = serde_json::to_vec_pretty(val)?;
    if let Some(path) = path {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, bytes)?;
    } else {
        println!("{}", String::from_utf8(bytes)?);
    }
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::List => {
            for key in config::GAMES {
                let cfg: RunConfig = RunConfig::named(key)?;
                println!(
                    "{key:14} actions={:2} arena={}",
                    cfg.game.actions.len(),
                    cfg.game.arena_compatible()
                );
            }
        }
        Command::Preset {
            game,
            checkpoint,
            output,
        } => {
            let cfg: RunConfig = if let Some(path) = checkpoint {
                checkpoint::read(&path)?.0.config
            } else {
                RunConfig::load_preset(
                    game.as_deref().expect("clap requires game or checkpoint"),
                    Path::new(PRESET_DIRECTORY),
                )?
            };
            if let Some(path) = output {
                cfg.save(&path)?;
            } else {
                write_json(None, &cfg)?;
            }
        }
        Command::Train {
            game,
            config: cfg_path,
            all,
            steps,
            envs,
            seed,
            batch_size,
            learning_starts,
            eval_steps,
            eval_episodes,
            out,
            device: backend,
            init,
        } => {
            let names: Vec<String> = if all {
                config::GAMES
                    .iter()
                    .map(|name: &&str| (*name).to_owned())
                    .collect()
            } else {
                vec![game.unwrap_or_else(|| "pong".to_owned())]
            };
            ensure!(!all || init.is_none(), "--init belongs to one game");

            for name in names {
                let mut cfg: RunConfig = if let Some(path) = &cfg_path {
                    RunConfig::load(path)?
                } else {
                    RunConfig::load_preset(&name, Path::new(PRESET_DIRECTORY))?
                };
                if let Some(val) = steps {
                    cfg.training.steps = val;
                }
                if let Some(val) = envs {
                    cfg.training.envs = val;
                }
                if let Some(val) = seed {
                    cfg.training.seed = val;
                }
                if let Some(val) = batch_size {
                    cfg.training.batch_size = val;
                }
                if let Some(val) = learning_starts {
                    cfg.training.learning_starts = val;
                }
                if let Some(val) = eval_steps {
                    cfg.training.evaluation_max_steps = val;
                }
                if let Some(val) = eval_episodes {
                    cfg.training.evaluation_episodes = val;
                }
                cfg.validate()?;

                let dev: Device = backend.device()?;
                let init_model: Option<DQN> = if let Some(path) = &init {
                    let (meta, model): (Metadata, DQN) = checkpoint::load(path, &dev)?;
                    ensure!(
                        meta.config.game.key == cfg.game.key
                            && meta.config.game.hidden == cfg.game.hidden
                            && meta.config.game.actions == cfg.game.actions
                            && meta.config.game.minimal_actions == cfg.game.minimal_actions
                            && meta.config.game.grayscale == cfg.game.grayscale
                            && meta.config.game.padding == cfg.game.padding,
                        "Warm-start game/action/preprocessing configuration differs"
                    );
                    ensure!(
                        meta.dueling == cfg.training.dueling,
                        "Warm-start head differs: set dueling to match checkpoint"
                    );
                    Some(model)
                } else {
                    None
                };
                let dest: PathBuf = out
                    .join(&cfg.game.key)
                    .join(format!("seed-{}", cfg.training.seed));
                let result: TrainingResult = atari_train_dqn::train::train(cfg, dev, &dest, init_model)?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
        }
        Command::Evaluate {
            checkpoint,
            seed,
            episodes,
            max_steps,
            device: backend,
            report,
        } => {
            ensure!(
                episodes > 0 && episodes <= 256 && max_steps > 0,
                "Invalid evaluation limits"
            );
            let dev: Device = backend.device()?;
            let (meta, model): (Metadata, DQN) = checkpoint::load(&checkpoint, &dev)?;
            let result: Evaluation =
                atari_train_dqn::evaluate::evaluate(&model, &meta.config.game, seed, episodes, max_steps)?;
            write_json(report.as_deref(), &result)?;
        }
        Command::Export {
            checkpoint,
            output,
            q_values,
        } => {
            let (meta, model): (Metadata, DQN) =
                checkpoint::load(&checkpoint, &Device::flex())?;
            atari_train_dqn::onnx::export(&output, &model, &meta, q_values)?;
            println!(
                "Exported {} (arena compatible: {})",
                output.display(),
                meta.config.game.arena_compatible()
            );
        }
        Command::Import { input, output } => {
            let data: Vec<u8> = std::fs::read(input)?;
            let (_, header) = safetensors::tensor::SafeTensors::read_metadata(&data)?;
            #[derive(Deserialize)]
            struct Legacy {
                game: String,
                actions: Vec<usize>,
                source: String,
                source_sha256: String,
                training_step: u64,
                environment: String,
            }
            let legacy: Legacy = serde_json::from_str(
                header
                    .metadata()
                    .as_ref()
                    .and_then(|metadata| metadata.get("legacy"))
                    .context("Missing legacy metadata")?,
            )?;
            let wts: Weights = checkpoint::read_weights(&data)?;
            let mut cfg: RunConfig = RunConfig::named(&legacy.game)?;
            cfg.game.actions = legacy.actions;
            cfg.training.dueling = false;
            cfg.training.n_step = 1;
            cfg.game.padding = Padding::Repeat;
            cfg.game.max_pool = false;
            cfg.game.noop_frame_skip = 4;
            cfg.game.sticky_probability = 0.25;
            cfg.game.reset_actions.clear();
            cfg.validate()?;

            let dev: Device = Device::flex();
            let model: DQN = DQN::new(&cfg.game, false, &dev).load(&wts, &dev)?;
            let meta: Metadata = Metadata {
                format_version: 1,
                config: cfg,
                dueling: false,
                repair_reset_padding: true,
                training_step: legacy.training_step,
                source: Some(format!(
                    "{}; SHA256 {}; original environment {}; historical optimizer/replay state not supplied",
                    legacy.source, legacy.source_sha256, legacy.environment
                )),
            };
            checkpoint::save(&output, &model, &meta)?;
            println!("Imported {}", output.display());
        }
        Command::Inspect { checkpoint } => {
            let (meta, wts) = checkpoint::read(&checkpoint)?;
            let parameters: usize = wts.values().map(|weight| weight.values.len()).sum();
            let tensors: Vec<serde_json::Value> = wts
                .iter()
                .map(|(name, weight)| serde_json::json!({"name": name, "shape": weight.shape}))
                .collect();
            write_json(
                None,
                &serde_json::json!({
                    "metadata": meta,
                    "parameters": parameters,
                    "tensors": tensors
                }),
            )?;
        }
        Command::Predict {
            checkpoint,
            input,
            output,
        } => {
            #[derive(Deserialize)]
            struct Input {
                values: Vec<f32>,
            }

            let dev: Device = Device::flex();
            let (meta, model): (Metadata, DQN) = checkpoint::load(&checkpoint, &dev)?;
            let mut data: Input = serde_json::from_slice(&std::fs::read(input)?)?;
            let shape: [usize; 3] = meta.config.game.shape();
            let elems: usize = shape.iter().product();
            ensure!(
                !data.values.is_empty()
                    && data.values.len().is_multiple_of(elems)
                    && data.values.iter().all(|val: &f32| val.is_finite()),
                "Invalid input dimensions/values"
            );
            let batch: usize = data.values.len() / elems;
            let frame_elems: usize = FRAME_WIDTH * FRAME_HEIGHT;
            if meta.repair_reset_padding {
                for obs in data.values.chunks_exact_mut(elems) {
                    for channel in (0..FRAME_STACK - 1).rev() {
                        let start: usize = channel * frame_elems;
                        if obs[start..start + frame_elems]
                            .iter()
                            .all(|&val: &f32| val == 0.0)
                        {
                            obs.copy_within(
                                start + frame_elems..start + 2 * frame_elems,
                                start,
                            );
                        }
                    }
                }
            }
            let tensor: Tensor<4> = Tensor::from_data(
                TensorData::new(
                    data.values,
                    [batch, FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH],
                ),
                &dev,
            );
            let vals: Vec<f32> = model
                .forward(tensor)
                .into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .map_err(|err| anyhow::anyhow!("{err:?}"))?;
            let actions: Vec<usize> = vals
                .chunks_exact(meta.config.game.actions.len())
                .map(|q_vals: &[f32]| {
                    let policy_idx: usize = q_vals
                        .iter()
                        .enumerate()
                        .max_by(|left, right| {
                            left.1
                                .total_cmp(right.1)
                                .then_with(|| right.0.cmp(&left.0))
                        })
                        .expect("actions nonempty")
                        .0;
                    meta.config.game.actions[policy_idx]
                })
                .collect();
            write_json(
                Some(&output),
                &serde_json::json!({"batch": batch, "q_values": vals, "actions": actions}),
            )?;
        }
        Command::ReplayBenchmark {
            transitions,
            output,
        } => {
            ensure!(
                transitions > 32 && transitions <= 100_000,
                "Benchmark count must be 33..100000"
            );
            let mut hist: FrameHistory = FrameHistory::new(Padding::Zero);
            let frame_elems: usize = FRAME_WIDTH * FRAME_HEIGHT;
            let mut prev: State = hist.push(vec![1_u8; frame_elems].into());
            let mut replay: ReplayMemory = ReplayMemory::new(transitions, 0.5);
            let start: Instant = Instant::now();
            for idx in 0..transitions {
                let px: u8 = u8::try_from(idx % 254 + 1).expect("pixel value fits u8");
                let next: State = hist.push(vec![px; frame_elems].into());
                replay.push(atari_train_dqn::replay::Transition {
                    state: prev,
                    action: 0,
                    reward: 1.0,
                    next: next.clone(),
                    discount: 0.99,
                });
                prev = next;
            }
            let fill_seconds: f64 = start.elapsed().as_secs_f64();
            use rand::SeedableRng;
            let mut rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(42);
            let start: Instant = Instant::now();
            for _ in 0..1_000 {
                std::hint::black_box(replay.sample(32, 0.4, &mut rng));
            }
            let sample_seconds: f64 = start.elapsed().as_secs_f64().max(f64::EPSILON);
            write_json(
                output.as_deref(),
                &serde_json::json!({
                    "transitions": transitions,
                    "estimated_unique_frame_bytes": (transitions + 2) * frame_elems,
                    "full_uint8_stacks_bytes": transitions * 2 * FRAME_STACK * frame_elems,
                    "full_float32_stacks_bytes": transitions * 2 * FRAME_STACK * frame_elems * 4,
                    "fill_seconds": fill_seconds,
                    "sample_batches_per_second": 1_000.0 / sample_seconds,
                    "note": "Pixel payload comparison; excludes Arc/transition/priority-tree overhead and is not an RSS measurement"
                }),
            )?;
        }
    }
    Ok(())
}
