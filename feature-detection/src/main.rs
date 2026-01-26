use image::{GenericImageView, ImageReader};

fn convolve<const N: usize>(
    input: &[u8],
    width: u32,
    height: u32,
    kernel: &[[f32; N]; N],
    output: &mut [f32],
) {
    // The kernel should always be odd-sized, and thus have a midpoint.
    assert!(N / 2 * 2 != N);

    let kernel_midpoint = (N / 2) as i32;
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
                    result += input[val_idx] as f32 * multiplier;
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

fn detect_features(pixels: &[u8], width: u32, height: u32) -> Vec<(u32, u32)> {
    assert_eq!(pixels.len() as u32, width * height);
    println!(
        "Feature detection would be performed on an image of dimensions {}x{} with {} pixels.",
        width,
        height,
        pixels.len()
    );

    // First, we will convolve the image with both the Sobel X and Y kernels to get the gradients.

    let sobel_x: [[f32; 3]; 3] = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]];
    let mut gradients_x = vec![0f32; pixels.len()];
    convolve(pixels, width, height, &sobel_x, &mut gradients_x);

    debug_output_image(
        &gradients_x,
        width,
        height,
        "../output/feature-detection/gradient_x.png",
    );

    let sobel_y: [[f32; 3]; 3] = [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]];
    let mut gradients_y = vec![0f32; pixels.len()];
    convolve(pixels, width, height, &sobel_y, &mut gradients_y);

    debug_output_image(
        &gradients_y,
        width,
        height,
        "../output/feature-detection/gradient_y.png",
    );

    // Now, we combine the gradients to get the gradient magnitude.
    let mut gradient_magnitude = vec![0f32; pixels.len()];
    for i in 0..pixels.len() {
        gradient_magnitude[i] =
            (gradients_x[i].powi(2) + gradients_y[i].powi(2)).sqrt();
    }

    debug_output_image(&gradient_magnitude, width, height, "../output/feature-detection/gradient_magnitude.png");





    unimplemented!("Feature detection logic not yet implemented");
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
