use crate::{config::GameSpec, state::State};
use anyhow::Result;

#[cfg(feature = "native")]
pub mod atari;

#[derive(Debug)]
pub struct Step {
    pub state: State,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
    pub life_lost: bool,
    pub scores: Option<[f32; 2]>,
}

impl Step {
    pub const fn done(&self) -> bool {
        self.terminated || self.truncated
    }
}

pub trait Environment: Send {
    fn reset(&mut self) -> Result<State>;
    fn step(&mut self, action: usize) -> Result<Step>;
}

pub fn create(spec: &GameSpec, seed: u64) -> Result<Box<dyn Environment>> {
    spec.validate()?;
    #[cfg(feature = "native")]
    {
        Ok(Box::new(atari::AtariEnvironment::new(spec.clone(), seed)?))
    }
    #[cfg(not(feature = "native"))]
    {
        let _ = (spec, seed);
        anyhow::bail!("This build has no native ALE; rebuild with the default native feature")
    }
}
