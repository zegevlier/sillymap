use nalgebra::{Matrix3, Vector3};

use super::{DiscreteKeyPoint, Keypoint, Octave};

fn determine_derivatives(
    p: &DiscreteKeyPoint,
    octaves: &[Octave],
) -> (Matrix3<f32>, Vector3<f32>, f32) {
    let dog = &octaves[p.octave].dogs[p.layer];
    let dog_prev = &octaves[p.octave].dogs[p.layer - 1];
    let dog_next = &octaves[p.octave].dogs[p.layer + 1];

    let x = p.x as i32;
    let y = p.y as i32;

    let val_c = dog.get_pixeli(x, y);
    let val_rx = dog.get_pixeli(x + 1, y);
    let val_lx = dog.get_pixeli(x - 1, y);

    let val_uy = dog.get_pixeli(x, y + 1);
    let val_dy = dog.get_pixeli(x, y - 1);

    let val_prev = dog_prev.get_pixeli(x, y);
    let val_next = dog_next.get_pixeli(x, y);

    let dxx = val_rx + val_lx - 2.0 * val_c;
    let dyy = val_uy + val_dy - 2.0 * val_c;
    let dss = val_next + val_prev - 2.0 * val_c;

    let dxy = (dog.get_pixeli(x + 1, y + 1)
        - dog.get_pixeli(x - 1, y + 1)
        - dog.get_pixeli(x + 1, y - 1)
        + dog.get_pixeli(x - 1, y - 1))
        / 4.0;

    let dxs = (dog_next.get_pixeli(x + 1, y)
        - dog_next.get_pixeli(x - 1, y)
        - dog_prev.get_pixeli(x + 1, y)
        + dog_prev.get_pixeli(x - 1, y))
        / 4.0;

    let dys = (dog_next.get_pixeli(x, y + 1)
        - dog_next.get_pixeli(x, y - 1)
        - dog_prev.get_pixeli(x, y + 1)
        + dog_prev.get_pixeli(x, y - 1))
        / 4.0;

    let hessian = Matrix3::new(dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss);
    let gradient = Vector3::new(
        (val_rx - val_lx) / 2.0,
        (val_uy - val_dy) / 2.0,
        (val_next - val_prev) / 2.0,
    );

    (hessian, gradient, val_c)
}

const R: f32 = 10.;
const EDGE_CHECK_THRESHOLD: f32 = ((R + 1.) * (R + 1.)) / R;
const CONTRAST_THRESHOLD: f32 = 0.015;

pub fn find_keypoints(octaves: &[Octave], extrema: Vec<DiscreteKeyPoint>) -> Vec<Keypoint> {
    let mut keypoints = vec![];
    let mut convergent_count = 0;
    let mut nonconvergent_count = 0;
    let mut discarded_low_contrast = 0;
    let mut discarded_edge = 0;

    for extrema_point in extrema {
        // For each extrema point, we determine the derivatives.
        let mut point = extrema_point;
        let mut loop_counter = 0;
        let result = loop {
            let (hessian, gradient, val_c) = determine_derivatives(&point, octaves);

            // Then, we can solve for the offset.
            let hessian_inv = match hessian.try_inverse() {
                Some(inv) => inv,
                None => break None, // We discard the degenerate case, where the hessian is not invertible.
            };
            let offset = -hessian_inv * gradient;

            if offset.x.abs() > 0.5 || offset.y.abs() > 0.5 || offset.z.abs() > 0.5 {
                // We need to look at a neighbouring point, since the offset is > 0.5 in some dimension.
                point = DiscreteKeyPoint {
                    octave: point.octave,
                    layer: (point.layer as isize + if offset[2] > 0.5 { 1 } else { 0 }) as usize
                        - if offset[2] < -0.5 { 1 } else { 0 },
                    x: (point.x as i32 + if offset[0] > 0.5 { 1 } else { 0 }
                        - if offset[0] < -0.5 { 1 } else { 0 }) as u32,
                    y: (point.y as i32 + if offset[1] > 0.5 { 1 } else { 0 }
                        - if offset[1] < -0.5 { 1 } else { 0 }) as u32,
                };
                loop_counter += 1;
                // If any are out of bounds, or we've been going for a while, we give up.
                let octave = &octaves[point.octave];
                if point.layer == 0
                    || point.layer >= octave.dogs.len() - 1
                    || point.x == 0
                    || point.x >= octave.dogs[0].get_width() - 1
                    || point.y == 0
                    || point.y >= octave.dogs[0].get_height() - 1
                    || loop_counter > 5
                {
                    break None;
                }
            } else {
                break Some((hessian, gradient, val_c, offset));
            }
        };

        if result.is_none() {
            nonconvergent_count += 1;
            continue;
        }
        convergent_count += 1;
        let (hessian, gradient, val_c, offset) = result.unwrap();

        let d_hat = val_c + 0.5 * gradient.dot(&offset);

        if d_hat.abs() < CONTRAST_THRESHOLD {
            discarded_low_contrast += 1;
            continue;
        }

        // Now, we need to make sure the keypoint is not on an edge.
        // Hessian is
        // xx xy ..
        // yx yy ..
        // .. .. ..
        let dxx = hessian[0];
        let dyy = hessian[4];
        let dxy = hessian[1];

        let tr = dxx + dyy;
        let det = (dxx * dyy) - dxy.powi(2);

        if det < 0. {
            // Discard this point, it's not an extrema.
            continue;
        }

        if (tr.powi(2) / det) >= EDGE_CHECK_THRESHOLD {
            // Discard this point, it's on an edge.
            discarded_edge += 1;
            continue;
        }

        // We now have a keypoint!
        let keypoint = Keypoint {
            octave: point.octave,
            layer: point.layer as f32 + offset[2],
            x: point.x as f32 + offset[0],
            y: point.y as f32 + offset[1],
        };
        keypoints.push(keypoint);
    }

    println!(
        "Converged keypoints: {}, non-converged keypoints: {}, discarded low-contrast keypoints: {}, discarded edge keypoints: {}",
        convergent_count, nonconvergent_count, discarded_low_contrast, discarded_edge
    );

    keypoints
}
