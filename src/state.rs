use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct State {
    frames: Arc<[Arc<[u8]>]>,
}

impl State {
    pub fn new(frames: Arc<[Arc<[u8]>]>) -> Self {
        Self { frames }
    }

    pub fn append(&self, out: &mut Vec<f32>) {
        out.extend(
            self.frames
                .iter()
                .flat_map(|frame: &Arc<[u8]>| frame.iter().map(|&px: &u8| f32::from(px))),
        );
    }

    pub fn elements(&self) -> usize {
        self.frames.iter().map(|frame: &Arc<[u8]>| frame.len()).sum()
    }
}
