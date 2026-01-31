use std::cmp::Ordering;

use super::{Octave, DiscreteKeyPoint};
use crate::pixel_image::Image;

fn is_extrema(dogs: &[Image], layer_idx: usize, x: u32, y: u32) -> bool {
    let current = dogs[layer_idx].get_pixel(x, y);
    let ordering = current.total_cmp(&dogs[layer_idx].get_pixel(x + 1, y));
    if ordering == Ordering::Equal {
        return false;
    }
    for dl in -1..=1 {
        for dx in -1..=1 {
            for dy in -1..=1 {
                if dl == 0 && dx == 0 && dy == 0 {
                    continue;
                }
                if current.total_cmp(
                    &dogs[(layer_idx as isize + dl) as usize]
                        .get_pixel((x as i32 + dx) as u32, (y as i32 + dy) as u32),
                ) != ordering
                {
                    return false;
                }
            }
        }
    }
    true
}

pub fn find_extrema(octaves: &[Octave]) -> Vec<DiscreteKeyPoint> {
    let mut keypoint_candidates = vec![];

    for (octave_idx, octave) in octaves.iter().enumerate() {
        // We can only look at the layers that have both a top and a bottom neighbour
        for layer_idx in 1..octave.dogs.len() - 1 {
            let width = octave.dogs[layer_idx].get_width();
            let height = octave.dogs[layer_idx].get_height();
            // Similarly here, we need to avoid the borders
            for x in 1..(width - 1) {
                for y in 1..(height - 1) {
                    if is_extrema(&octave.dogs, layer_idx, x, y) {
                        keypoint_candidates.push(DiscreteKeyPoint {
                            octave: octave_idx,
                            layer: layer_idx,
                            x,
                            y,
                        });
                    }
                }
            }
        }
    }

    keypoint_candidates
}
