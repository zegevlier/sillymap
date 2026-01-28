use std::cmp::Ordering;

use image::{GenericImageView, ImageReader};

mod pixel_image;
use nalgebra::{Matrix3, Vector3};
use pixel_image::Image;

const S: u32 = 3;
const SIGMA_0: f32 = 1.6;
const IMAGES_PER_OCTAVE: u32 = S + 3;

struct Octave {
    // The S+3 blurred images
    gaussians: Vec<Image>,
    // Difference-of-gaussians, the S+2 differences of these blurred images
    dogs: Vec<Image>,
}

fn build_octaves(input_image: Image, num_octaves: u32) -> Vec<Octave> {
    let mut octaves = Vec::with_capacity(num_octaves as usize);

    // We need to double the input image size first, as per the SIFT paper.
    let initial_resized = input_image.resize_double();

    // We then need to make sure the initial sigma is correct.
    // The paper assumes that the input image has sigma 0.5, doubled that would be 1.0,
    // and it needs to reach SIGMA_0 (1.6).
    let sigma_diff = (SIGMA_0 * SIGMA_0 - 1.0 * 1.0).sqrt();
    let mut current_base_image = initial_resized.gaussian_blur(sigma_diff);

    let k = 2_f32.powf(1. / S as f32);

    for _ in 0..num_octaves {
        let mut octave_gaussians = vec![];
        octave_gaussians.push(current_base_image.clone());

        let mut current_sigma = SIGMA_0;

        for _ in 1..IMAGES_PER_OCTAVE {
            // We first need to calculate the sigma ew need to blur the previous image with.
            // We want the current sigma to be the previous sigma * k.
            // So, the sigma we need to use = sqrt((current sigma * k)^2 - (current sigma)^2)
            // This works out to current sigma * sqrt(k^2-1)
            let sigma = current_sigma * (k.powi(2) - 1.).sqrt();

            let previous_image = octave_gaussians.last().unwrap();
            let new_image = previous_image.gaussian_blur(sigma);

            octave_gaussians.push(new_image);

            // Update the current sigma value.
            current_sigma *= k;
        }

        let mut octave_dogs = vec![];
        for i in 0..(IMAGES_PER_OCTAVE as usize - 1) {
            let dog = octave_gaussians[i + 1].subtract(&octave_gaussians[i]);
            octave_dogs.push(dog)
        }

        // Now, we downsample the image, as per the paper.
        // We need to that the Sth image from the octave gaussians, and downsample that.
        let image_to_downsample = octave_gaussians[S as usize].clone();
        current_base_image = image_to_downsample.resize_half();

        octaves.push(Octave {
            gaussians: octave_gaussians,
            dogs: octave_dogs,
        });
    }

    // We first need to double the size of the input image.
    octaves
}

#[derive(Debug)]
struct DiscreteKeyPoint {
    octave: usize,
    // This is the DoG image index within the octave.
    layer: usize,
    x: u32,
    y: u32,
}

#[derive(Debug)]
struct Keypoint {
    octave: usize,
    layer: f32,
    x: f32,
    y: f32,
}

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

fn detect_extrema(octaves: &[Octave]) -> Vec<DiscreteKeyPoint> {
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

fn sift(pixels: &[u8], width: u32, height: u32) -> Vec<(u32, u32)> {
    assert_eq!(pixels.len() as u32, width * height);
    println!(
        "Feature detection would be performed on an image of dimensions {}x{} with {} pixels.",
        width,
        height,
        pixels.len()
    );

    let image = Image::new_u8(width, height, pixels.to_vec());

    let resized_image = image.resize_half();

    resized_image.write_debug_out("../output/feature-detection/resized.png");

    let octaves = build_octaves(image, 4);

    // We write them all out to files, for debugging purposes.
    for (octave_index, octave) in octaves.iter().enumerate() {
        for (gaussian_index, gaussian) in octave.gaussians.iter().enumerate() {
            let path = format!(
                "../output/feature-detection/octave_{}_gaussian_{}.png",
                octave_index, gaussian_index
            );
            gaussian.write_debug_out(&path);
        }
        for (dog_index, dog) in octave.dogs.iter().enumerate() {
            let path = format!(
                "../output/feature-detection/octave_{}_dog_{}.png",
                octave_index, dog_index
            );
            dog.write_debug_out(&path);
        }
    }

    let extrema = detect_extrema(&octaves);

    let mut keypoints = vec![];
    let mut convergent_count = 0;
    let mut nonconvergent_count = 0;

    for extrema_point in extrema {
        // For each extrema point, we determine the derivatives.
        let mut point = extrema_point;
        let mut loop_counter = 0;
        let result = loop {
            let (hessian, gradient, val_c) = determine_derivatives(&point, &octaves);

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

        if d_hat.abs() < 0.03 {
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
        "Converged keypoints: {}, non-converged keypoints: {}",
        convergent_count, nonconvergent_count
    );

    dbg!(&keypoints.len());

    vec![]
}

fn main() {
    // First, we create a directory for storing the intermediate and output images, if it doesn't exist.
    println!("Creating output directory...");
    std::fs::create_dir_all("../output/feature-detection/").unwrap();

    // We load the image from the data directory.
    println!("Loading image...");
    let img = ImageReader::open("../data/images/500px-Writing_desk.jpg")
        .unwrap()
        .decode()
        .unwrap();

    // We convert the image to grayscale, since we don't want colours for the feature detection.
    println!("Converting image to grayscale...");
    let gray_image = img.to_luma8();
    // We save the grayscale image for reference.
    gray_image
        .save("../output/feature-detection/grayscale.png")
        .unwrap();

    println!("Extracting pixels...");
    // Now, we extract all the pixels from the grayscale image, so that we can process them.
    let pixels: Vec<u8> = gray_image.into_raw();
    let (width, height) = img.dimensions();

    // Now we do the actual feature detection.
    println!("Detecting features...");
    let features = sift(&pixels, width, height);

    // We write out the features to a file, so that we can inspect them later.
    println!("Writing features to file...");
    let mut feature_file =
        std::fs::File::create("../output/feature-detection/features.txt").unwrap();
    for (x, y) in &features {
        use std::io::Write;
        writeln!(feature_file, "{},{}", x, y).unwrap();
    }

    // Finally, we create an output image with the detected features drawn on top of the original image.
    println!("Creating output image with features...");
    let mut output_image = img.to_rgba8();
    let marker_size = (width / 200) as i32; // Size of the feature marker.
    for (x, y) in &features {
        // Draw a small red dot at each feature location.
        for dx in -marker_size..=marker_size {
            for dy in -marker_size..=marker_size {
                let px = x.saturating_add(dx as u32);
                let py = y.saturating_add(dy as u32);
                if px < width && py < height {
                    output_image.put_pixel(px, py, image::Rgba([255, 0, 0, 255]));
                }
            }
        }
    }
    output_image
        .save("../output/feature-detection/detected_features.png")
        .unwrap();
}
