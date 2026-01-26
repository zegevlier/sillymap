#[derive(Clone)]
pub struct Image {
    width: u32,
    height: u32,
    data: Vec<f32>,
}

impl Image {
    pub fn new(width: u32, height: u32, data: Vec<f32>) -> Self {
        assert_eq!(data.len() as u32, width * height);
        Image {
            width,
            height,
            data,
        }
    }

    pub fn new_u8(width: u32, height: u32, data: Vec<u8>) -> Self {
        assert_eq!(data.len() as u32, width * height);
        Image {
            width,
            height,
            data: data.iter().map(|&p| p as f32).collect(),
        }
    }

    pub fn empty(width: u32, height: u32) -> Self {
        Image {
            width,
            height,
            data: vec![0.0; (width * height) as usize],
        }
    }

    pub fn write_debug_out(&self, path: &str) {
        let mut debug_image = image::GrayImage::new(self.width, self.height);
        for x in 0..self.width {
            for y in 0..self.height {
                let idx = (y * self.width + x) as usize;
                let pixel_value = self.data[idx].clamp(0.0, 255.0) as u8;
                debug_image.put_pixel(x, y, image::Luma([pixel_value]));
            }
        }
        debug_image
            .save(path)
            .expect("Failed to save debug output image");
    }

    pub fn resize_double(&self) -> Self {
        let new_width = self.width * 2;
        let new_height = self.height * 2;
        let mut new_data = vec![0.0; (new_width * new_height) as usize];

        // First, we copy over the existing pixels.
        for x in 0..self.width {
            for y in 0..self.height {
                let old_idx = (y * self.width + x) as usize;
                let new_x = x * 2;
                let new_y = y * 2;
                let new_idx = (new_y * new_width + new_x) as usize;
                new_data[new_idx] = self.data[old_idx];
            }
        }

        // Next, we interpolate the missing pixels.
        for x in 0..new_width {
            for y in 0..new_height {
                let idx = (y * new_width + x) as usize;
                // We only need to interpolate if the pixel would not have been copied over.
                if x % 2 == 1 || y % 2 == 1 {
                    // We average the surrounding pixels.
                    let mut sum = 0.0;
                    let mut count = 0;

                    for dx in -1..=1 {
                        for dy in -1..=1 {
                            let px = x as isize + dx;
                            let py = y as isize + dy;
                            if px >= 0
                                && px < new_width as isize
                                && py >= 0
                                && py < new_height as isize
                                && (px % 2 == 0)
                                && (py % 2 == 0)
                            {
                                let neighbor_idx = (py as u32 * new_width + px as u32) as usize;
                                sum += new_data[neighbor_idx];
                                count += 1;
                            }
                        }
                    }

                    if count > 0 {
                        new_data[idx] = sum / count as f32;
                    }
                }
            }
        }

        Image {
            width: new_width,
            height: new_height,
            data: new_data,
        }
    }

    pub fn resize_half(&self) -> Self {
        let new_width = self.width / 2;
        let new_height = self.height / 2;
        let mut new_data = vec![0.0; (new_width * new_height) as usize];

        for x in 0..new_width {
            for y in 0..new_height {
                // We just take every second pixel.
                let old_x = x * 2;
                let old_y = y * 2;
                let old_idx = (old_y * self.width + old_x) as usize;
                let new_idx = (y * new_width + x) as usize;
                new_data[new_idx] = self.data[old_idx];
            }
        }

        Image {
            width: new_width,
            height: new_height,
            data: new_data,
        }
    }

    pub fn convolve(&self, kernel: &[&[f32]]) -> Self {
        // The kernel should always be odd-sized, and thus have a midpoint.
        assert!(kernel.len() % 2 == 1);
        // The kernel should always be square
        assert!(kernel.len() == kernel[0].len());

        let kernel_midpoint = (kernel.len() / 2) as i32;
        // For a 3x3 kernel, we want to go from -1, 0, 1,
        // kernel_midpoint would be 1, so we want to go from -kernel_midpoint..=kernel_midpoint

        let mut output_image = Image::empty(self.width, self.height);
        let output = &mut output_image.data;
        for x in 0..self.width {
            for y in 0..self.height {
                let idx = (y * self.width + x) as usize;
                let mut result = 0.;
                for kernel_x in -kernel_midpoint..=kernel_midpoint {
                    for kernel_y in -kernel_midpoint..=kernel_midpoint {
                        let val_x = (x as i32 + kernel_y).clamp(0, self.width as i32 - 1) as u32;
                        let val_y = (y as i32 + kernel_x).clamp(0, self.height as i32 - 1) as u32;
                        let val_idx = (val_y * self.width + val_x) as usize;

                        // If it is in range, we find out the multiplier.
                        let multiplier = kernel[(kernel_x + kernel_midpoint) as usize]
                            [(kernel_y + kernel_midpoint) as usize];
                        result += self.data[val_idx] * multiplier;
                    }
                }
                output[idx] = result;
            }
        }
        output_image
    }

    pub fn gaussian_blur(&self, sigma: f32) -> Self {
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

        // Now we perform the convolution.
        self.convolve(
            // TODO: There is probably a better way to do this conversion.
            &kernel.iter().map(|v| v.as_slice()).collect::<Vec<&[f32]>>(),
        )
    }

    pub fn subtract(&self, other: &Self) -> Self {
        assert_eq!(self.width, other.width);
        assert_eq!(self.height, other.height);

        let mut result = Image::empty(self.width, self.height);

        for i in 0..result.data.len() {
            result.data[i] = self.data[i] - other.data[i];
        }

        result
    }

    pub fn get_width(&self) -> u32 {
        self.width
    }

    pub fn get_height(&self) -> u32 {
        self.height
    }

    pub(crate) fn get_pixel(&self, x: u32, y: u32) -> f32 {
        self.data[(self.width * x + y) as usize]
    }
}
