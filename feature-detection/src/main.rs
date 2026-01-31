use image::{GenericImageView, ImageReader};

mod keypoints;
mod pixel_image;
use pixel_image::Image;

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

    let octaves = keypoints::build_octaves(image, 4);

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

    let extrema = keypoints::find_extrema(&octaves);
    let keypoints = keypoints::find_keypoints(&octaves, extrema);
    dbg!(&keypoints[..10]);
    dbg!(&keypoints.len());
    let oriented_keypoints = keypoints::assign_orientations(&keypoints, &octaves);
    dbg!(&oriented_keypoints[..10]);

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
