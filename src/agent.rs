use crate::{
    config::{RunConfig, TrainingConfig},
    model::{BatchBuffer, DQN},
    replay::Sample,
    state::State,
};
use anyhow::Result;
use burn::{
    optim::{AdamConfig, GradientsParams, ModuleOptimizer, grad_clipping::GradientClippingConfig},
    prelude::*,
};

pub struct Agent {
    pub model: DQN,
    pub target: DQN,
    opt: ModuleOptimizer,
    pub updates: u64,
    dev: Device,
    cfg: RunConfig,
    buf: BatchBuffer,
}

impl Agent {
    pub fn new(cfg: RunConfig, dev: Device, init: Option<DQN>) -> Self {
        let dev: Device = dev.autodiff();
        dev.seed(cfg.training.seed);
        let model: DQN = init
            .map(|model: DQN| model.train().fork(&dev))
            .unwrap_or_else(|| DQN::new(&cfg.game, cfg.training.dueling, &dev));
        let target: DQN = model.valid();
        let opt: ModuleOptimizer = AdamConfig::new()
            .with_epsilon(1.5e-4)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(10.0)))
            .init();
        Self {
            model,
            target,
            opt,
            updates: 0,
            dev,
            cfg,
            buf: BatchBuffer::default(),
        }
    }

    pub fn actions(&mut self, states: &[State]) -> Result<Vec<usize>> {
        let valid: DQN = self.model.valid();
        let inp: Tensor<4> = self
            .buf
            .tensor(states, self.cfg.game.shape(), &valid.devices()[0]);
        let out: Vec<i64> = valid
            .forward(inp)
            .argmax(1)
            .reshape([states.len()])
            .into_data()
            .convert::<i64>()
            .try_to_vec::<i64>()
            .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        out.into_iter()
            .map(|val: i64| {
                usize::try_from(val).map_err(|_| anyhow::anyhow!("Negative policy action index"))
            })
            .collect()
    }

    pub fn update(mut self, sample: &Sample) -> Result<(Self, f32, Vec<f32>)> {
        let n: usize = sample.transitions.len();
        let shape: [usize; 3] = self.cfg.game.shape();
        let dev: &Device = &self.dev;
        let inp: Tensor<4> = self
            .buf
            .tensor(sample.transitions.iter().map(|t| &t.state), shape, dev);
        let action_vals: Vec<i64> = sample
            .transitions
            .iter()
            .map(|t| i64::try_from(t.action).expect("policy action index fits i64"))
            .collect();
        let acts: Tensor<2, Int> = Tensor::<2, Int>::from_data(
            TensorData::new(action_vals, [n, 1]),
            dev,
        );
        let q: Tensor<1> = self.model.forward(inp).gather(1, acts).reshape([n]);

        let valid: DQN = self.model.valid();
        let raw_dev: Device = valid.devices()[0].clone();
        let next: Tensor<4> = self.buf.tensor(
            sample.transitions.iter().map(|t| &t.next),
            shape,
            &raw_dev,
        );
        let best = valid.forward(next.clone()).argmax(1);
        let next_q: Tensor<1> = self.target.forward(next).gather(1, best).reshape([n]);
        let rewards: Tensor<1> = Tensor::<1>::from_data(
            TensorData::new(
                sample
                    .transitions
                    .iter()
                    .map(|t| t.reward)
                    .collect::<Vec<f32>>(),
                [n],
            ),
            &raw_dev,
        );
        let discounts: Tensor<1> = Tensor::<1>::from_data(
            TensorData::new(
                sample
                    .transitions
                    .iter()
                    .map(|t| t.discount)
                    .collect::<Vec<f32>>(),
                [n],
            ),
            &raw_dev,
        );
        let targets: Tensor<1> = (rewards + discounts * next_q).to_device(dev).detach();
        let diff: Tensor<1> = q - targets;
        let abs: Tensor<1> = diff.clone().abs();
        let quad: Tensor<1> = abs.clone().clamp_max(1.0);
        let lin: Tensor<1> = abs.clone() - quad.clone();
        let huber: Tensor<1> = quad.powf_scalar(2.0) * 0.5 + lin;
        let wts: Tensor<1> =
            Tensor::<1>::from_data(TensorData::new(sample.weights.clone(), [n]), dev);
        let loss: Tensor<1> = (huber * wts).mean();
        let val: f32 = loss.clone().into_scalar::<f32>();
        let errs: Vec<f32> = abs
            .detach()
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        anyhow::ensure!(
            val.is_finite() && errs.iter().all(|err: &f32| err.is_finite()),
            "Nonfinite TD loss; inspect learning-rate/reward settings"
        );
        let grads: GradientsParams = GradientsParams::from_grads(loss.backward(), &self.model);
        self.model = self
            .opt
            .step(self.cfg.training.learning_rate, self.model, grads);
        self.updates += 1;
        if self
            .updates
            .is_multiple_of(self.cfg.training.target_every)
        {
            self.target = self.model.valid();
        }
        Ok((self, val, errs))
    }
}

pub fn beta(cfg: &TrainingConfig, step: u64) -> f32 {
    let warmup: u64 = u64::try_from(cfg.learning_starts)
        .expect("learning_starts fits u64 on supported targets");
    let horizon: u64 = cfg.steps.saturating_sub(warmup).max(1);
    let elapsed: u64 = step.saturating_sub(warmup).min(horizon);
    let progress: f64 = elapsed as f64 / horizon as f64;
    let start: f64 = f64::from(cfg.priority_beta);
    (start + (1.0 - start) * progress) as f32
}
