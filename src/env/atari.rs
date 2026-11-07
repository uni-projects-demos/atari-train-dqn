use super::{Environment, Step};
use crate::{
    config::{GameSpec, Scoring},
    preprocess::{FrameHistory, Preprocessor},
    state::State,
};
use ale_bridge::{Ale, Info};
use anyhow::{Result, anyhow, ensure};
use std::sync::Arc;

pub struct AtariEnvironment {
    spec: GameSpec,
    ale: Ale,
    prep: Preprocessor,
    hist: FrameHistory,
    rng: u32,
    steps: u64,
    lives: i32,
    scores: [f32; 2],
}

impl AtariEnvironment {
    pub fn new(spec: GameSpec, seed: u64) -> Result<Self> {
        let ale_seed: u32 = u32::try_from(seed & u64::from(u32::MAX))
            .expect("masked seed always fits u32");
        let ale: Ale = Ale::new(&spec.rom, ale_seed, spec.sticky_probability)
            .map_err(|err: String| anyhow!(err))?;
        ensure!(
            ale.actions() == spec.minimal_actions,
            "{} minimal action set differs from config: {:?}",
            spec.key,
            ale.actions()
        );
        let hist: FrameHistory = FrameHistory::new(spec.padding);
        Ok(Self {
            spec,
            ale,
            prep: Preprocessor::default(),
            hist,
            rng: ale_seed.max(1),
            steps: 0,
            lives: 0,
            scores: [0.0; 2],
        })
    }

    fn random(&mut self, n: u32) -> u32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        self.rng % n.max(1)
    }

    fn obs(&mut self) -> State {
        let [w, h]: [usize; 2] = self.ale.dimensions();
        let frame: Arc<[u8]> = self
            .prep
            .process(self.ale.rgb(), w, h, self.spec.grayscale);
        self.hist.push(frame)
    }

    fn repeat(&mut self, action: usize) -> Result<Info> {
        let _: u32 = self.random(1);
        self.ale
            .step(action, self.spec.frame_skip, self.spec.max_pool)
            .map_err(|err: String| anyhow!(err))
    }

    fn update_scores(&mut self, reward: f32) {
        match &self.spec.scoring {
            Scoring::Signed => {
                self.scores[0] += reward.max(0.0);
                self.scores[1] += (-reward).max(0.0);
            }
            Scoring::Ram {
                addresses,
                bcd,
                overrides,
            } => {
                let ram: &[u8; 128] = self.ale.ram();
                self.scores = addresses.map(|addr: usize| {
                    let val: u8 = ram[addr];
                    if let Some(pair) = overrides
                        .iter()
                        .find(|pair: &&[u16; 2]| pair[0] == u16::from(val))
                    {
                        f32::from(pair[1])
                    } else if *bcd {
                        f32::from((val >> 4) * 10 + (val & 15))
                    } else {
                        f32::from(val)
                    }
                });
            }
        }
    }
}

impl Environment for AtariEnvironment {
    fn reset(&mut self) -> Result<State> {
        let mut last: Info = self.ale.reset().map_err(|err: String| anyhow!(err))?;
        self.hist.reset();
        self.steps = 0;
        self.scores = [0.0; 2];

        let noops: u32 = if self.spec.noop_max == 0 {
            0
        } else {
            self.random(self.spec.noop_max) + 1
        };
        for _ in 0..noops {
            let info: Info = self
                .ale
                .step(0, self.spec.noop_frame_skip, false)
                .map_err(|err: String| anyhow!(err))?;
            last = info;
            if info.done != 0 {
                last = self.ale.reset().map_err(|err: String| anyhow!(err))?;
            }
        }

        for action in self.spec.reset_actions.clone() {
            let info: Info = self.repeat(action)?;
            last = info;
            if info.done != 0 {
                last = self.ale.reset().map_err(|err: String| anyhow!(err))?;
            }
        }

        self.lives = last.lives;
        Ok(self.obs())
    }

    fn step(&mut self, action: usize) -> Result<Step> {
        ensure!(action < self.spec.actions.len(), "Invalid policy action");
        let raw_action: usize = self.spec.actions[action];
        let mut info: Info = self.repeat(raw_action)?;
        let reward: f32 = info.reward;
        self.update_scores(reward);

        let life_lost: bool = self.spec.episodic_life
            && info.done == 0
            && info.lives > 0
            && info.lives < self.lives;
        if life_lost {
            for reset_action in std::iter::once(0).chain(self.spec.reset_actions.clone()) {
                info = self.repeat(reset_action)?;
                if info.done != 0 {
                    break;
                }
            }
            self.hist.reset();
        }

        self.lives = info.lives;
        self.steps += 1;
        Ok(Step {
            state: self.obs(),
            reward,
            terminated: info.done != 0 && info.truncated == 0,
            truncated: info.truncated != 0 || self.steps >= self.spec.max_episode_steps,
            life_lost,
            scores: Some(self.scores),
        })
    }
}
