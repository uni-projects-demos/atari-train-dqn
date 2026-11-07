use crate::{config::GameSpec, state::State};
use anyhow::{Result, ensure};
use burn::{
    module::Param,
    nn::{
        Linear, LinearConfig,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
    tensor::activation::relu,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const PIXEL_SCALE: f32 = 255.0;
pub const CONVS: [(usize, usize, usize, usize); 3] =
    [(4, 32, 8, 4), (32, 64, 4, 2), (64, 64, 3, 1)];

#[derive(Module, Debug)]
pub struct DQN {
    convs: Vec<Conv2d>,
    hidden: Linear,
    output: Linear,
    val: Option<Linear>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Weight {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

pub type Weights = BTreeMap<String, Weight>;

impl DQN {
    pub fn new(spec: &GameSpec, dueling: bool, dev: &Device) -> Self {
        let convs: Vec<Conv2d> = CONVS
            .iter()
            .map(|&(inp, out, kernel, stride): &(usize, usize, usize, usize)| {
                Conv2dConfig::new([inp, out], [kernel, kernel])
                    .with_stride([stride, stride])
                    .init(dev)
            })
            .collect();
        let hidden: Linear = LinearConfig::new(3136, spec.hidden).init(dev);
        let output: Linear = LinearConfig::new(spec.hidden, spec.actions.len()).init(dev);
        let val: Option<Linear> = dueling.then(|| LinearConfig::new(spec.hidden, 1).init(dev));

        Self {
            convs,
            hidden,
            output,
            val,
        }
    }

    pub fn forward(&self, inp: Tensor<4>) -> Tensor<2> {
        let mut x: Tensor<4> = inp / PIXEL_SCALE;
        for conv in &self.convs {
            x = relu(conv.forward(x));
        }
        let feat: Tensor<2> = relu(self.hidden.forward(x.flatten::<2>(1, 3)));
        let adv: Tensor<2> = self.output.forward(feat.clone());
        match &self.val {
            Some(val) => {
                let mean: Tensor<2> = adv.clone().mean_dim(1);
                val.forward(feat) + adv - mean
            }
            None => adv,
        }
    }

    pub fn weights(&self) -> Result<Weights> {
        let mut out: Weights = Weights::new();
        for (idx, conv) in self.convs.iter().enumerate() {
            let name: String = format!("conv{}", idx + 1);
            put(&mut out, &format!("{name}.weight"), conv.weight.val())?;
            if let Some(bias) = &conv.bias {
                put(&mut out, &format!("{name}.bias"), bias.val())?;
            }
        }
        for (name, linear) in [
            ("hidden", Some(&self.hidden)),
            ("output", Some(&self.output)),
            ("value", self.val.as_ref()),
        ] {
            if let Some(layer) = linear {
                put(
                    &mut out,
                    &format!("{name}.weight"),
                    layer.weight.val().transpose(),
                )?;
                if let Some(bias) = &layer.bias {
                    put(&mut out, &format!("{name}.bias"), bias.val())?;
                }
            }
        }
        Ok(out)
    }

    pub fn load(mut self, wts: &Weights, dev: &Device) -> Result<Self> {
        let mut used: usize = 0;
        for (idx, conv) in self.convs.iter_mut().enumerate() {
            let name: String = format!("conv{}", idx + 1);
            conv.weight = Param::from_tensor(get(
                wts,
                &format!("{name}.weight"),
                conv.weight.val().dims(),
                dev,
            )?);
            used += 1;
            if let Some(bias) = &mut conv.bias {
                *bias = Param::from_tensor(get(
                    wts,
                    &format!("{name}.bias"),
                    bias.val().dims(),
                    dev,
                )?);
                used += 1;
            }
        }
        for (name, linear) in [
            ("hidden", Some(&mut self.hidden)),
            ("output", Some(&mut self.output)),
            ("value", self.val.as_mut()),
        ] {
            if let Some(layer) = linear {
                let dims: [usize; 2] = layer.weight.val().dims();
                let tensor: Tensor<2> = get(
                    wts,
                    &format!("{name}.weight"),
                    [dims[1], dims[0]],
                    dev,
                )?;
                layer.weight = Param::from_tensor(tensor.transpose());
                used += 1;
                if let Some(bias) = &mut layer.bias {
                    *bias = Param::from_tensor(get(
                        wts,
                        &format!("{name}.bias"),
                        bias.val().dims(),
                        dev,
                    )?);
                    used += 1;
                }
            }
        }
        ensure!(used == wts.len(), "Checkpoint contains unexpected tensors");
        Ok(self)
    }
}

fn put<const D: usize>(out: &mut Weights, name: &str, tensor: Tensor<D>) -> Result<()> {
    let shape: Vec<usize> = tensor.dims().to_vec();
    let vals: Vec<f32> = tensor
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .map_err(|err| anyhow::anyhow!("{err:?}"))?;
    ensure!(
        vals.iter().all(|val: &f32| val.is_finite()),
        "Nonfinite tensor {name}"
    );
    out.insert(
        name.to_owned(),
        Weight {
            shape,
            values: vals,
        },
    );
    Ok(())
}

fn get<const D: usize>(
    wts: &Weights,
    name: &str,
    shape: [usize; D],
    dev: &Device,
) -> Result<Tensor<D>> {
    let weight: &Weight = wts
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("Missing tensor {name}"))?;
    ensure!(
        weight.shape == shape
            && weight.values.len() == shape.iter().product::<usize>()
            && weight.values.iter().all(|val: &f32| val.is_finite()),
        "Invalid tensor {name}"
    );
    Ok(Tensor::from_data(
        TensorData::new(weight.values.clone(), shape),
        dev,
    ))
}

#[derive(Default)]
pub struct BatchBuffer {
    vals: Vec<f32>,
}

impl BatchBuffer {
    pub fn tensor<'a>(
        &mut self,
        states: impl IntoIterator<Item = &'a State>,
        shape: [usize; 3],
        dev: &Device,
    ) -> Tensor<4> {
        self.vals.clear();
        let elems: usize = shape.iter().product();
        let iter = states.into_iter();
        self.vals.reserve(iter.size_hint().0 * elems);
        let mut n: usize = 0;
        for state in iter {
            assert_eq!(state.elements(), elems);
            state.append(&mut self.vals);
            n += 1;
        }
        Tensor::from_data(
            TensorData::new(
                std::mem::take(&mut self.vals),
                [n, shape[0], shape[1], shape[2]],
            ),
            dev,
        )
    }
}
