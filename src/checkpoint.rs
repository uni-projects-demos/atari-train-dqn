use crate::{
    config::RunConfig,
    model::{DQN, Weight, Weights},
};
use anyhow::{Context, Result, ensure};
use burn::tensor::Device;
use safetensors::tensor::{Dtype, SafeTensors, TensorView, serialize_to_file};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

const MAX_CHECKPOINT_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub format_version: u32,
    pub config: RunConfig,
    pub dueling: bool,
    pub repair_reset_padding: bool,
    pub training_step: u64,
    pub source: Option<String>,
}

impl Metadata {
    pub fn fresh(cfg: RunConfig) -> Self {
        let dueling: bool = cfg.training.dueling;
        let repair_reset_padding: bool = cfg.game.padding == crate::config::Padding::Repeat
            && ["pong", "breakout"].contains(&cfg.game.key.as_str());
        Self {
            format_version: 1,
            config: cfg,
            dueling,
            repair_reset_padding,
            training_step: 0,
            source: None,
        }
    }
}

pub fn save(path: &Path, model: &DQN, meta: &Metadata) -> Result<()> {
    save_weights(path, &model.weights()?, meta)
}

pub fn save_weights(path: &Path, weights: &Weights, meta: &Metadata) -> Result<()> {
    meta.config.validate()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let storage: Vec<(String, Vec<usize>, Vec<u8>)> = weights
        .iter()
        .map(|(name, weight)| {
            let bytes: Vec<u8> = weight
                .values
                .iter()
                .flat_map(|val: &f32| val.to_le_bytes())
                .collect();
            (name.clone(), weight.shape.clone(), bytes)
        })
        .collect();
    let views: Vec<(&str, TensorView<'_>)> = storage
        .iter()
        .map(|(name, shape, bytes)| {
            Ok((
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let metadata: Option<HashMap<String, String>> = Some(HashMap::from([(
        "rl".to_owned(),
        serde_json::to_string(meta)?,
    )]));
    let tmp: PathBuf = path.with_extension("safetensors.tmp");
    serialize_to_file(views, metadata, &tmp).context("Writing SafeTensors")?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

pub fn read(path: &Path) -> Result<(Metadata, Weights)> {
    let file_len: u64 = std::fs::metadata(path)?.len();
    ensure!(
        file_len <= MAX_CHECKPOINT_BYTES,
        "Checkpoint exceeds 256 MiB"
    );
    let data: Vec<u8> = std::fs::read(path)?;
    let (_, header) = SafeTensors::read_metadata(&data)?;
    let txt: &String = header
        .metadata()
        .as_ref()
        .and_then(|metadata| metadata.get("rl"))
        .context("Missing RL checkpoint metadata")?;
    let meta: Metadata = serde_json::from_str(txt)?;
    ensure!(meta.format_version == 1, "Unsupported checkpoint format");
    meta.config.validate()?;
    ensure!(
        meta.dueling == meta.config.training.dueling,
        "Checkpoint head metadata is inconsistent"
    );
    let wts: Weights = read_weights(&data)?;
    Ok((meta, wts))
}

pub fn load(path: &Path, dev: &Device) -> Result<(Metadata, DQN)> {
    let (meta, wts): (Metadata, Weights) = read(path)?;
    let model: DQN = DQN::new(&meta.config.game, meta.dueling, dev).load(&wts, dev)?;
    Ok((meta, model))
}

pub fn read_weights(data: &[u8]) -> Result<Weights> {
    let tensors: SafeTensors<'_> = SafeTensors::deserialize(data)?;
    let mut wts: Weights = Weights::new();
    for (name, tensor) in tensors.tensors() {
        ensure!(
            tensor.dtype() == Dtype::F32,
            "Only float32 weights are supported"
        );
        let (chunks, remainder): (&[[u8; 4]], &[u8]) = tensor.data().as_chunks::<4>();
        ensure!(remainder.is_empty(), "Invalid float32 tensor byte length");
        let vals: Vec<f32> = chunks
            .iter()
            .map(|bytes: &[u8; 4]| f32::from_le_bytes(*bytes))
            .collect();
        ensure!(
            vals.iter().all(|val: &f32| val.is_finite()),
            "Nonfinite checkpoint weights"
        );
        wts.insert(
            name,
            Weight {
                shape: tensor.shape().to_vec(),
                values: vals,
            },
        );
    }
    Ok(wts)
}
