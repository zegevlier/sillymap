use crate::pixel_image::Image;

pub(crate) struct Octave {
    // The S+3 blurred images
    pub(crate) gaussians: Vec<Image>,
    // Difference-of-gaussians, the S+2 differences of these blurred images
    pub(crate) dogs: Vec<Image>,
}

const S: u32 = 3;
const SIGMA_0: f32 = 1.6;
const IMAGES_PER_OCTAVE: u32 = S + 3;

pub fn build_octaves(input_image: Image, num_octaves: u32) -> Vec<Octave> {
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

    octaves
}
