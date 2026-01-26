use image::{GenericImageView, ImageReader};

fn convolve(input: &[f32], width: u32, height: u32, kernel: &[&[f32]], output: &mut [f32]) {
    // The kernel should always be odd-sized, and thus have a midpoint.
    assert!(kernel.len() / 2 * 2 != kernel.len());
    // The kernel should always be square
    assert!(kernel.len() == kernel[0].len());

    let kernel_midpoint = (kernel.len() / 2) as i32;
    // For a 3x3 kernel, we want to go from -1, 0, 1,
    // kernel_midpoint would be 1, so we want to go from -kernel_midpoint..=kernel_midpoint

    for x in 0..width {
        for y in 0..height {
            let idx = (y * width + x) as usize;
            let mut result = 0.;
            for kernel_x in -kernel_midpoint..=kernel_midpoint {
                for kernel_y in -kernel_midpoint..=kernel_midpoint {
                    let val_x = (x as i32 + kernel_y).clamp(0, width as i32 - 1) as u32;
                    let val_y = (y as i32 + kernel_x).clamp(0, height as i32 - 1) as u32;
                    let val_idx = (val_y * width + val_x) as usize;

                    // If it is in range, we find out the multiplier.
                    let multiplier = kernel[(kernel_x + kernel_midpoint) as usize]
                        [(kernel_y + kernel_midpoint) as usize];
                    result += input[val_idx] * multiplier;
                }
            }
            output[idx] = result;
        }
    }
}

fn debug_output_image(pixels: &[f32], width: u32, height: u32, path: &str) {
    let mut debug_image = image::GrayImage::new(width, height);
    for x in 0..width {
        for y in 0..height {
            let idx = (y * width + x) as usize;
            let value = pixels[idx].abs().min(255.) as u8;
            debug_image.put_pixel(x, y, image::Luma([value]));
        }
    }
    debug_image.save(path).unwrap();
}

fn gaussian_convolution(input: Vec<f32>, width: u32, height: u32, sigma: f32) -> Vec<f32> {
    // We set the radius of the Gaussian kernel to be 3 times the sigma, rounded up.
    // This is apparently a common choice.
    let radius = (sigma * 3.).ceil() as isize;
    let center = radius;
    // The gaussian kernel size should be 2*radius, but because we want the center pixel, we add 1.
    let gaussian_size = (radius * 2 + 1) as usize;

    // We can pre-compute the inverse coefficient, as it does not depend on x or y.
    // Same with 2*sigma^2
    let inv_coeff = 1. / 2. * std::f32::consts::PI * sigma.powi(2);
    let two_sigma_sq = 2. * sigma.powi(2);

    // Now we create the Gaussian kernel.
    let mut kernel = vec![vec![0.; gaussian_size]; gaussian_size];
    let mut sum = 0.0;

    // Fill in the kernel values.
    #[allow(clippy::needless_range_loop)]
    for x in 0..gaussian_size {
        for y in 0..gaussian_size {
            // We need to use dx and dy as the function is centered around (0,0),
            // but we need to center it around (center, center).
            let dx = (x as isize - center) as f32;
            let dy = (y as isize - center) as f32;
            let exp = -(dx.powi(2) + dy.powi(2)) / two_sigma_sq;
            let val = inv_coeff * exp.exp();
            kernel[x][y] = val;
            sum += val;
        }
    }

    // Normalize the kernel so that the sum of all elements is 1.
    #[allow(clippy::needless_range_loop)]
    for x in 0..gaussian_size {
        for y in 0..gaussian_size {
            kernel[x][y] /= sum;
        }
    }

    // Our output will be the same size as our input, initialize it.
    let mut output = vec![0.; input.len()];

    // Now we perform the convolution.
    convolve(
        &input,
        width,
        height,
        // TODO: There is probably a better way to do this conversion.
        &kernel.iter().map(|v| v.as_slice()).collect::<Vec<&[f32]>>(),
        &mut output,
    );

    output
}

fn detect_features(pixels: &[u8], width: u32, height: u32) -> Vec<(u32, u32)> {
    assert_eq!(pixels.len() as u32, width * height);
    println!(
        "Feature detection would be performed on an image of dimensions {}x{} with {} pixels.",
        width,
        height,
        pixels.len()
    );

    let float_pixels: Vec<f32> = pixels.iter().map(|&p| p as f32).collect();
    let blurred_pixels = gaussian_convolution(float_pixels, width, height, 2.);

    debug_output_image(
        &blurred_pixels,
        width,
        height,
        "../output/feature-detection/blurred.png",
    );

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
    let features = detect_features(&pixels, width, height);

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
