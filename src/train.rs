use crate::{
    agent::{Agent, beta},
    checkpoint::{self, Metadata},
    config::{GameSpec, RunConfig, TrainingConfig},
    env,
    evaluate::{self, Episode, Evaluation},
    model::DQN,
    onnx,
    replay::{NStep, RawTransition, ReplayMemory, Sample, Transition},
    state::State,
};
use anyhow::Result;
use burn::tensor::Device;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

struct Worker {
    env: Box<dyn env::Environment>,
    state: State,
    ep_return: f32,
    ep_steps: u64,
    nstep: NStep,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingResult {
    pub game: String,
    pub decisions: u64,
    pub updates: u64,
    pub episodes: u64,
    pub wall_seconds: f64,
    pub decisions_per_second: f64,
    pub final_loss: Option<f32>,
    pub exported_checkpoint: String,
    pub evaluation: Evaluation,
}

pub fn train(
    run_cfg: RunConfig,
    dev: Device,
    out: &Path,
    init: Option<DQN>,
) -> Result<TrainingResult> {
    run_cfg.validate()?;
    anyhow::ensure!(
        !out.exists() || std::fs::read_dir(out)?.next().is_none(),
        "Run directory is not empty: {}; choose another --out or seed",
        out.display()
    );
    std::fs::create_dir_all(out)?;
    run_cfg.save(&out.join("config.json"))?;

    let cfg: TrainingConfig = run_cfg.training.clone();
    let spec: GameSpec = run_cfg.game.clone();
    let start: Instant = Instant::now();
    let mut agent: Agent = Agent::new(run_cfg.clone(), dev, init);
    let mut explore_rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(cfg.seed ^ 0x12ab_93ef);
    let mut sample_rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(cfg.seed ^ 0x89de_6732);
    let mut workers: Vec<Worker> = (0..cfg.envs)
        .map(|idx: usize| {
            let offset: u64 = u64::try_from(idx).expect("environment index fits u64");
            let mut environment: Box<dyn env::Environment> =
                env::create(&spec, cfg.seed.wrapping_add(offset))?;
            let state: State = environment.reset()?;
            Ok(Worker {
                env: environment,
                state,
                ep_return: 0.0,
                ep_steps: 0,
                nstep: NStep::new(cfg.n_step, cfg.gamma),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut replay: ReplayMemory = ReplayMemory::new(cfg.replay_capacity, cfg.priority_alpha);
    let mut steps: u64 = 0;
    let mut episodes: u64 = 0;
    let mut update_tokens: u64 = 0;
    let mut next_eval: u64 = cfg.evaluate_every;
    let mut last_eval: u64 = 0;
    let mut best: f32 = f32::NEG_INFINITY;
    let mut final_loss: Option<f32> = None;
    let mut latest: Option<Evaluation> = None;
    let mut returns: Vec<(u64, f32)> = Vec::new();
    let mut losses: Vec<(u64, f32)> = Vec::new();

    let mut episode_log: BufWriter<File> =
        BufWriter::new(File::create(out.join("episodes.csv"))?);
    writeln!(
        episode_log,
        "decision,environment,episode,steps,raw_return,terminated,truncated,model_score,game_score"
    )?;
    let mut train_log: BufWriter<File> =
        BufWriter::new(File::create(out.join("training.csv"))?);
    writeln!(
        train_log,
        "decision,updates,loss,epsilon,replay,decisions_per_second"
    )?;
    let mut eval_log: BufWriter<File> =
        BufWriter::new(File::create(out.join("evaluations.jsonl"))?);

    while steps < cfg.steps {
        let remaining: u64 = cfg.steps - steps;
        let batch_envs: u64 = remaining.min(u64::try_from(cfg.envs).expect("env count fits u64"));
        let n: usize = usize::try_from(batch_envs).expect("bounded env count fits usize");
        let epsilon: f32 = if replay.len() < cfg.learning_starts {
            1.0
        } else {
            cfg.epsilon(steps)
        };
        let random: Vec<bool> = (0..n)
            .map(|_: usize| explore_rng.random::<f32>() < epsilon)
            .collect();
        let mut actions: Vec<usize> = if random.iter().any(|val: &bool| !*val) {
            let states: Vec<State> = workers[..n]
                .iter()
                .map(|worker: &Worker| worker.state.clone())
                .collect();
            agent.actions(&states)?
        } else {
            vec![0; n]
        };
        for (idx, &is_random) in random.iter().enumerate() {
            if is_random {
                actions[idx] = explore_rng.random_range(0..spec.actions.len());
            }
        }

        let collected: Vec<Result<(Vec<Transition>, Option<Episode>)>> = workers[..n]
            .par_iter_mut()
            .zip(actions.par_iter())
            .enumerate()
            .map(|(idx, (worker, &action))| {
                let state: State = worker.state.clone();
                let step: env::Step = worker.env.step(action)?;
                let done: bool = step.done();
                worker.ep_return += step.reward;
                worker.ep_steps += 1;
                let reward: f32 = if cfg.clip_reward {
                    step.reward.clamp(-1.0, 1.0)
                } else {
                    step.reward
                };
                let transitions: Vec<Transition> = worker.nstep.push(RawTransition {
                    state,
                    action,
                    reward,
                    next: step.state.clone(),
                    terminated: step.terminated || step.life_lost,
                    episode_end: done || step.life_lost,
                });
                let episode: Option<Episode> = done.then_some(Episode {
                    seed: cfg
                        .seed
                        .wrapping_add(u64::try_from(idx).expect("environment index fits u64")),
                    steps: worker.ep_steps,
                    raw_return: worker.ep_return,
                    terminated: step.terminated,
                    truncated: step.truncated,
                    evaluation_cutoff: false,
                    scores: step.scores,
                });
                worker.state = if done {
                    worker.ep_return = 0.0;
                    worker.ep_steps = 0;
                    worker.env.reset()?
                } else {
                    step.state
                };
                Ok((transitions, episode))
            })
            .collect();

        for (idx, result) in collected.into_iter().enumerate() {
            let (transitions, episode): (Vec<Transition>, Option<Episode>) = result?;
            for transition in transitions {
                replay.push(transition);
            }
            if let Some(episode) = episode {
                episodes += 1;
                let decision: u64 = steps
                    + u64::try_from(idx).expect("environment index fits u64")
                    + 1;
                returns.push((decision, episode.raw_return));
                let scores: [String; 2] = episode
                    .scores
                    .map(|score: [f32; 2]| [score[0].to_string(), score[1].to_string()])
                    .unwrap_or_default();
                writeln!(
                    episode_log,
                    "{decision},{idx},{episodes},{},{},{},{},{},{}",
                    episode.steps,
                    episode.raw_return,
                    episode.terminated,
                    episode.truncated,
                    scores[0],
                    scores[1]
                )?;
            }
        }

        steps += batch_envs;
        update_tokens += batch_envs;
        let due: u64 = update_tokens / cfg.train_every;
        update_tokens %= cfg.train_every;
        if replay.len() >= cfg.learning_starts {
            let due_updates: usize = usize::try_from(due)
                .expect("update count per environment batch fits usize")
                .checked_mul(cfg.gradient_steps)
                .expect("gradient update count overflow");
            for _ in 0..due_updates {
                let sample: Sample =
                    replay.sample(cfg.batch_size, beta(&cfg, steps), &mut sample_rng);
                let (next_agent, loss, errs): (Agent, f32, Vec<f32>) = agent.update(&sample)?;
                agent = next_agent;
                replay.update(&sample.idxs, &errs);
                final_loss = Some(loss);
                if agent.updates.is_multiple_of(100) || losses.is_empty() {
                    losses.push((steps, loss));
                    let elapsed: f64 = start.elapsed().as_secs_f64().max(f64::EPSILON);
                    writeln!(
                        train_log,
                        "{steps},{},{loss},{epsilon},{},{}",
                        agent.updates,
                        replay.len(),
                        steps as f64 / elapsed
                    )?;
                }
            }
        }

        if steps >= next_eval || steps == cfg.steps {
            let report: Evaluation = evaluate::evaluate(
                &agent.model,
                &spec,
                cfg.seed.wrapping_add(1_000_000),
                cfg.evaluation_episodes,
                cfg.evaluation_max_steps,
            )?;
            let meta: Metadata = Metadata {
                training_step: steps,
                ..Metadata::fresh(run_cfg.clone())
            };
            checkpoint::save(&out.join("last.safetensors"), &agent.model, &meta)?;
            if report.completed == cfg.evaluation_episodes
                && let Some(mean) = report.mean_completed_return
                && mean > best
            {
                best = mean;
                checkpoint::save(&out.join("best.safetensors"), &agent.model, &meta)?;
            }
            writeln!(
                eval_log,
                "{}",
                serde_json::json!({
                    "decisions": steps,
                    "updates": agent.updates,
                    "evaluation": report
                })
            )?;
            eval_log.flush()?;
            episode_log.flush()?;
            train_log.flush()?;
            std::fs::write(out.join("learning.svg"), chart(&returns, &losses))?;
            println!(
                "{} decisions={} updates={} completed_eval={}/{} mean_completed={:?}",
                spec.key,
                steps,
                agent.updates,
                report.completed,
                cfg.evaluation_episodes,
                report.mean_completed_return
            );
            latest = Some(report);
            last_eval = steps;
            while next_eval <= steps {
                next_eval = next_eval.saturating_add(cfg.evaluate_every);
            }
        }
    }

    debug_assert_eq!(last_eval, steps);
    let exported_checkpoint: &str = if best.is_finite() {
        "best.safetensors"
    } else {
        "last.safetensors"
    };
    let (meta, model): (Metadata, DQN) = checkpoint::load(
        &out.join(exported_checkpoint),
        &burn::tensor::Device::flex(),
    )?;
    onnx::export(&out.join("policy.onnx"), &model, &meta, false)?;

    let secs: f64 = start.elapsed().as_secs_f64().max(f64::EPSILON);
    let result: TrainingResult = TrainingResult {
        game: spec.key,
        decisions: steps,
        updates: agent.updates,
        episodes,
        wall_seconds: secs,
        decisions_per_second: steps as f64 / secs,
        final_loss,
        exported_checkpoint: exported_checkpoint.to_owned(),
        evaluation: latest.expect("final evaluation is mandatory"),
    };
    std::fs::write(
        out.join("result.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    Ok(result)
}

fn chart(returns: &[(u64, f32)], losses: &[(u64, f32)]) -> String {
    let mut svg: String = String::from(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="960" height="500" viewBox="0 0 960 500"><rect width="960" height="500" fill="white"/><g font-family="sans-serif" font-size="14" fill="#222">"##,
    );
    for (data, top, title, color) in [
        (returns, 40.0_f64, "Training episode return (raw)", "#2563eb"),
        (
            losses,
            280.0_f64,
            "Weighted Huber loss (sampled every 100 updates)",
            "#dc2626",
        ),
    ] {
        svg += &format!(
            r##"<text x="70" y="{}">{title}</text><path d="M70 {}V{}H920" fill="none" stroke="#777"/>"##,
            top - 15.0,
            top,
            top + 160.0
        );
        if !data.is_empty() {
            let max_x: f64 = data.last().map_or(1, |val: &(u64, f32)| val.0).max(1) as f64;
            let lo: f32 = data
                .iter()
                .map(|val: &(u64, f32)| val.1)
                .fold(f32::INFINITY, f32::min);
            let hi: f32 = data
                .iter()
                .map(|val: &(u64, f32)| val.1)
                .fold(f32::NEG_INFINITY, f32::max);
            let points: String = data
                .iter()
                .map(|&(x, y): &(u64, f32)| {
                    format!(
                        "{:.2},{:.2}",
                        70.0 + 850.0 * x as f64 / max_x,
                        top + 160.0 - 160.0 * f64::from((y - lo) / (hi - lo).max(1e-6))
                    )
                })
                .collect::<Vec<String>>()
                .join(" ");
            svg += &format!(
                r##"<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"/><text x="5" y="{}">{hi:.2}</text><text x="5" y="{}">{lo:.2}</text><text x="800" y="{}">{max_x:.0} decisions</text>"##,
                top + 10.0,
                top + 160.0,
                top + 182.0
            );
        }
    }
    svg += "</g></svg>";
    svg
}
