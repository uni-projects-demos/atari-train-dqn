use crate::{
    config::{FRAME_HEIGHT, FRAME_STACK, FRAME_WIDTH, Grayscale, Padding},
    state::State,
};
use std::{collections::VecDeque, sync::Arc};

#[derive(Default)]
pub struct Preprocessor {
    plan: Option<Plan>,
    gray: Vec<u8>,
    rows: Vec<f32>,
    resized: Vec<u8>,
}

struct Plan {
    shape: [usize; 4],
    horiz: Vec<Vec<(usize, f32)>>,
    vert: Vec<Vec<(usize, f32)>>,
}

impl Preprocessor {
    pub fn process(&mut self, rgb: &[u8], w: usize, h: usize, order: Grayscale) -> Arc<[u8]> {
        let shape: [usize; 4] = [w, h, FRAME_WIDTH, FRAME_HEIGHT];
        if self.plan.as_ref().is_none_or(|plan: &Plan| plan.shape != shape) {
            self.plan = Some(Plan {
                shape,
                horiz: weights(w, FRAME_WIDTH),
                vert: weights(h, FRAME_HEIGHT),
            });
        }
        let plan: &Plan = self.plan.as_ref().expect("resize plan initialized");
        match order {
            Grayscale::BeforeResize => {
                self.gray.clear();
                self.gray.extend(
                    rgb.as_chunks::<3>()
                        .0
                        .iter()
                        .map(|px: &[u8; 3]| luma(px, order)),
                );
                resize(&self.gray, plan, 1, &mut self.rows, &mut self.resized);
                Arc::from(self.resized.as_slice())
            }
            Grayscale::AfterResize => {
                resize(rgb, plan, 3, &mut self.rows, &mut self.resized);
                self.gray.clear();
                self.gray.extend(
                    self.resized
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .map(|px: &[u8; 3]| luma(px, order)),
                );
                Arc::from(self.gray.as_slice())
            }
        }
    }
}

fn luma(px: &[u8], order: Grayscale) -> u8 {
    let (r, g, b, shift): (u32, u32, u32, u32) = match order {
        Grayscale::BeforeResize => (4899, 9617, 1868, 14),
        Grayscale::AfterResize => (9798, 19235, 3735, 15),
    };
    let rounded: u32 = u32::from(px[0]) * r
        + u32::from(px[1]) * g
        + u32::from(px[2]) * b
        + (1_u32 << (shift - 1));
    u8::try_from(rounded >> shift).expect("luma is bounded by u8")
}

fn weights(src: usize, dest: usize) -> Vec<Vec<(usize, f32)>> {
    let scale: f64 = src as f64 / dest as f64;
    (0..dest)
        .map(|idx: usize| {
            let start: f64 = idx as f64 * scale;
            let end: f64 = start + scale;
            let cell: f64 = scale.min(src as f64 - start);
            let last: usize = (end.floor() as usize).min(src - 1);
            let first: usize = (start.ceil() as usize).min(last);
            let mut out: Vec<(usize, f32)> = Vec::new();
            if first as f64 - start > 1e-3 {
                out.push((first - 1, ((first as f64 - start) / cell) as f32));
            }
            for x in first..last {
                out.push((x, (1.0 / cell) as f32));
            }
            if end - last as f64 > 1e-3 {
                let weight: f64 = (end - last as f64).min(1.0).min(cell) / cell;
                out.push((last, weight as f32));
            }
            out
        })
        .collect()
}

fn resize(inp: &[u8], plan: &Plan, channels: usize, rows: &mut Vec<f32>, out: &mut Vec<u8>) {
    let [src_w, src_h, width, height]: [usize; 4] = plan.shape;
    rows.resize(src_h * width * channels, 0.0);
    for src_y in 0..src_h {
        for (x, weights) in plan.horiz.iter().enumerate() {
            for channel in 0..channels {
                let val: f32 = weights.iter().fold(0.0, |sum: f32, &(src_x, alpha)| {
                    sum + f32::from(inp[(src_y * src_w + src_x) * channels + channel]) * alpha
                });
                rows[(src_y * width + x) * channels + channel] = val;
            }
        }
    }

    out.resize(width * height * channels, 0);
    for (y, weights) in plan.vert.iter().enumerate() {
        for x in 0..width * channels {
            let val: f32 = weights.iter().fold(0.0, |sum: f32, &(src_y, beta)| {
                sum + rows[src_y * width * channels + x] * beta
            });
            out[y * width * channels + x] = val.round_ties_even().clamp(0.0, 255.0) as u8;
        }
    }
}

pub struct FrameHistory {
    frames: VecDeque<Arc<[u8]>>,
    padding: Padding,
    zero: Arc<[u8]>,
}

impl FrameHistory {
    pub fn new(padding: Padding) -> Self {
        Self {
            frames: VecDeque::with_capacity(FRAME_STACK),
            padding,
            zero: vec![0; FRAME_WIDTH * FRAME_HEIGHT].into(),
        }
    }

    pub fn reset(&mut self) {
        self.frames.clear();
    }

    pub fn push(&mut self, frame: Arc<[u8]>) -> State {
        if self.frames.is_empty() {
            for _ in 0..FRAME_STACK - 1 {
                let padding: Arc<[u8]> = match self.padding {
                    Padding::Zero => self.zero.clone(),
                    Padding::Repeat => frame.clone(),
                };
                self.frames.push_back(padding);
            }
        }
        if self.frames.len() == FRAME_STACK {
            let _ = self.frames.pop_front();
        }
        self.frames.push_back(frame);
        State::new(self.frames.iter().cloned().collect::<Vec<Arc<[u8]>>>().into())
    }
}
