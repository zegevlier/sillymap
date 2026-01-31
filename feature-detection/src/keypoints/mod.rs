mod determine;
mod extrema;
mod octaves;
mod orientation;

pub use determine::find_keypoints;
pub use extrema::find_extrema;
pub use octaves::build_octaves;
pub use orientation::assign_orientations;

use octaves::Octave;

#[derive(Debug)]
pub(crate) struct OrientedKeypoint {
    pub octave: usize,
    pub layer: f32,
    pub x: f32,
    pub y: f32,
    pub orientation: f32,
}

#[derive(Debug)]
pub(crate) struct DiscreteKeyPoint {
    octave: usize,
    // This is the DoG image index within the octave.
    layer: usize,
    x: u32,
    y: u32,
}

#[derive(Debug)]
pub(crate) struct Keypoint {
    octave: usize,
    layer: f32,
    x: f32,
    y: f32,
}
