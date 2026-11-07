mod file;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const FRAME_STACK: usize = 4;
pub const FRAME_WIDTH: usize = 84;
pub const FRAME_HEIGHT: usize = 84;
pub const ALE_ACTIONS: usize = 18;
pub const MAX_EPISODE_STEPS: u64 = 27_000;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Padding {
    Zero,
    Repeat,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Grayscale {
    BeforeResize,
    AfterResize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Scoring {
    Signed,
    Ram {
        addresses: [usize; 2],
        bcd: bool,
        overrides: Vec<[u16; 2]>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameSpec {
    pub key: String,
    pub rom: PathBuf,
    pub scoring: Scoring,
    pub actions: Vec<usize>,
    pub minimal_actions: Vec<i32>,
    pub grayscale: Grayscale,
    pub padding: Padding,
    pub frame_skip: u32,
    pub sticky_probability: f32,
    pub max_pool: bool,
    pub noop_max: u32,
    pub noop_frame_skip: u32,
    pub reset_actions: Vec<usize>,
    pub episodic_life: bool,
    pub max_episode_steps: u64,
    pub hidden: usize,
}

impl GameSpec {
    pub const fn shape(&self) -> [usize; 3] {
        [FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH]
    }

    pub fn arena_compatible(&self) -> bool {
        GAMES.contains(&self.key.as_str())
            && Self::named(&self.key).is_ok_and(|target: Self| {
                self.minimal_actions == target.minimal_actions
                    && self.grayscale == target.grayscale
                    && (self.padding == target.padding
                        || self.padding == Padding::Repeat && target.padding == Padding::Zero)
            })
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.key.is_empty()
                && self
                    .key
                    .chars()
                    .all(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
            "Invalid game key"
        );
        ensure!(
            !self.rom.as_os_str().is_empty(),
            "Missing Atari ROM path"
        );
        if let Scoring::Ram { addresses, .. } = &self.scoring {
            ensure!(
                addresses.iter().all(|&addr: &usize| addr < 128),
                "RAM score addresses must be 0..127"
            );
        }
        ensure!(
            !self.minimal_actions.is_empty()
                && self.minimal_actions.len() <= ALE_ACTIONS
                && self
                    .minimal_actions
                    .iter()
                    .all(|&action: &i32| (0..ALE_ACTIONS as i32).contains(&action)),
            "Invalid raw ALE action set"
        );
        ensure!(
            !self.actions.is_empty() && self.actions.len() <= ALE_ACTIONS,
            "Require 1..18 policy actions"
        );
        ensure!(
            self.actions
                .iter()
                .all(|&action: &usize| action < self.minimal_actions.len()),
            "Action map leaves minimal action set"
        );
        ensure!(
            self.actions.iter().enumerate().all(|(idx, action): (usize, &usize)| {
                !self.actions[..idx].contains(action)
            }),
            "Duplicate policy actions"
        );
        ensure!(
            (1..=60).contains(&self.frame_skip) && (1..=60).contains(&self.noop_frame_skip),
            "Frame repeat must be 1..60"
        );
        ensure!(
            self.sticky_probability.is_finite()
                && (0.0..=1.0).contains(&self.sticky_probability)
                && self.noop_max <= 100,
            "Invalid reset/sticky settings"
        );
        ensure!(
            self.reset_actions
                .iter()
                .all(|&action: &usize| action < self.minimal_actions.len()),
            "Invalid reset action"
        );
        ensure!(
            self.max_episode_steps > 0 && (1..=8192).contains(&self.hidden),
            "Invalid network/episode limits"
        );
        Ok(())
    }

    pub fn named(name: &str) -> Result<Self> {
        let normalized: String = name
            .to_ascii_lowercase()
            .replace(['-', '/', '_', ' '], "");
        let key: &'static str = match normalized.as_str() {
            "pong" | "alepongv5" | "pongv0" | "pongv4" => "pong",
            "breakout" | "alebreakoutv5" => "breakout",
            "boxing" | "aleboxingv5" => "boxing",
            "icehockey" | "aleicehockeyv5" => "icehockey",
            "spaceinvaders" | "alespaceinvadersv5" => "spaceinvaders",
            "tennis" | "aletennisv5" => "tennis",
            _ => bail!("Unknown Atari game {name}; use a full --config file for a custom ROM"),
        };

        let minimal_actions: Vec<i32> = match key {
            "pong" | "spaceinvaders" => vec![0, 1, 3, 4, 11, 12],
            "breakout" => vec![0, 1, 3, 4],
            _ => (0..ALE_ACTIONS as i32).collect(),
        };
        let actions: Vec<usize> = (0..minimal_actions.len()).collect();
        let sample_factory_compatible: bool = ["boxing", "icehockey", "tennis"].contains(&key);
        let rom_name: &'static str = match key {
            "icehockey" => "ice_hockey",
            "spaceinvaders" => "space_invaders",
            _ => key,
        };
        let scoring: Scoring = match key {
            "boxing" => Scoring::Ram {
                addresses: [18, 19],
                bcd: true,
                overrides: vec![[192, 100]],
            },
            "icehockey" => Scoring::Ram {
                addresses: [10, 11],
                bcd: true,
                overrides: Vec::new(),
            },
            "tennis" => Scoring::Ram {
                addresses: [71, 72],
                bcd: false,
                overrides: Vec::new(),
            },
            _ => Scoring::Signed,
        };
        let reset_actions: Vec<usize> = match key {
            "tennis" => vec![1],
            _ => vec![1, 2],
        };

        Ok(Self {
            key: key.to_owned(),
            rom: PathBuf::from(format!("assets/roms/{rom_name}.bin")),
            scoring,
            actions,
            minimal_actions,
            grayscale: if sample_factory_compatible {
                Grayscale::AfterResize
            } else {
                Grayscale::BeforeResize
            },
            padding: if sample_factory_compatible {
                Padding::Repeat
            } else {
                Padding::Zero
            },
            frame_skip: 4,
            sticky_probability: 0.0,
            max_pool: true,
            noop_max: 30,
            noop_frame_skip: 1,
            reset_actions,
            episodic_life: key == "breakout",
            max_episode_steps: MAX_EPISODE_STEPS,
            hidden: 512,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrainingConfig {
    pub seed: u64,
    pub steps: u64,
    pub envs: usize,
    pub replay_capacity: usize,
    pub batch_size: usize,
    pub learning_starts: usize,
    pub train_every: u64,
    pub gradient_steps: usize,
    pub target_every: u64,
    pub learning_rate: f64,
    pub gamma: f32,
    pub n_step: usize,
    pub epsilon_start: f32,
    pub epsilon_end: f32,
    pub epsilon_decay: u64,
    pub priority_alpha: f32,
    pub priority_beta: f32,
    pub clip_reward: bool,
    pub dueling: bool,
    pub evaluate_every: u64,
    pub evaluation_episodes: usize,
    pub evaluation_max_steps: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            steps: 50_000_000,
            envs: 4,
            replay_capacity: 100_000,
            batch_size: 32,
            learning_starts: 20_000,
            train_every: 4,
            gradient_steps: 1,
            target_every: 2_000,
            learning_rate: 6.25e-5,
            gamma: 0.99,
            n_step: 3,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 250_000,
            priority_alpha: 0.5,
            priority_beta: 0.4,
            clip_reward: true,
            dueling: true,
            evaluate_every: 250_000,
            evaluation_episodes: 5,
            evaluation_max_steps: MAX_EPISODE_STEPS,
        }
    }
}

impl TrainingConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.steps > 0 && (1..=256).contains(&self.envs),
            "Invalid steps/env count"
        );
        ensure!(
            self.batch_size > 0
                && self.replay_capacity >= self.batch_size
                && self.replay_capacity <= 10_000_000,
            "Invalid replay/batch size"
        );
        ensure!(
            self.learning_starts >= self.batch_size
                && self.learning_starts <= self.replay_capacity,
            "learning_starts must be between batch size and replay capacity"
        );
        ensure!(
            self.train_every > 0
                && self.target_every > 0
                && (1..=64).contains(&self.gradient_steps)
                && self.epsilon_decay > 0,
            "Invalid update/epsilon periods"
        );
        ensure!(
            self.learning_rate.is_finite()
                && self.learning_rate > 0.0
                && self.gamma.is_finite()
                && (0.0..=1.0).contains(&self.gamma),
            "Invalid learning rate/gamma"
        );
        ensure!(
            (1..=32).contains(&self.n_step)
                && self.epsilon_start.is_finite()
                && self.epsilon_end.is_finite()
                && (0.0..=1.0).contains(&self.epsilon_end)
                && self.epsilon_start >= self.epsilon_end
                && self.epsilon_start <= 1.0,
            "Invalid n-step/epsilon"
        );
        ensure!(
            self.priority_alpha.is_finite()
                && self.priority_beta.is_finite()
                && (0.0..=1.0).contains(&self.priority_alpha)
                && (0.0..=1.0).contains(&self.priority_beta),
            "Invalid replay priorities"
        );
        ensure!(
            self.evaluate_every > 0
                && (1..=256).contains(&self.evaluation_episodes)
                && self.evaluation_max_steps > 0,
            "Invalid evaluation settings"
        );
        Ok(())
    }

    pub fn epsilon(&self, step: u64) -> f32 {
        let warmup: u64 = u64::try_from(self.learning_starts)
            .expect("learning_starts fits u64 on supported targets");
        let decay_step: u64 = step.saturating_sub(warmup).min(self.epsilon_decay);
        let progress: f64 = decay_step as f64 / self.epsilon_decay as f64;
        let start: f64 = f64::from(self.epsilon_start);
        let end: f64 = f64::from(self.epsilon_end);
        (end + (start - end) * (1.0 - progress)) as f32
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    pub game: GameSpec,
    #[serde(default)]
    pub training: TrainingConfig,
}

impl RunConfig {
    pub fn named(name: &str) -> Result<Self> {
        Ok(Self {
            game: GameSpec::named(name)?,
            training: TrainingConfig::default(),
        })
    }

    pub fn load_preset(name: &str, dir: &Path) -> Result<Self> {
        let defs: Self = Self::named(name)?;
        let path: PathBuf = dir.join(format!("{}.json", defs.game.key));
        if !path
            .try_exists()
            .with_context(|| format!("Could not inspect preset {}", path.display()))?
        {
            return Ok(defs);
        }
        let cfg: Self = Self::load(&path)?;
        ensure!(
            cfg.game.key == defs.game.key,
            "Preset {} selects {}; expected {}",
            path.display(),
            cfg.game.key,
            defs.game.key
        );
        Ok(cfg)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let input: RunConfigInput = file::read(path)?;
        let game: GameSpec = match input.game {
            GameConfig::Named(name) => GameSpec::named(&name)?,
            GameConfig::Full(spec) => spec,
        };
        let cfg: Self = Self {
            game,
            training: input.training,
        };
        cfg.validate()
            .with_context(|| format!("Invalid configuration values in {}", path.display()))?;
        Ok(cfg)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        file::write(path, self)
    }

    pub fn validate(&self) -> Result<()> {
        self.game.validate()?;
        self.training.validate()
    }
}

pub const GAMES: [&str; 6] = [
    "boxing",
    "breakout",
    "icehockey",
    "pong",
    "spaceinvaders",
    "tennis",
];

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum GameConfig {
    Named(String),
    Full(GameSpec),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunConfigInput {
    game: GameConfig,
    #[serde(default)]
    training: TrainingConfig,
}
