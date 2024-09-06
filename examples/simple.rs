use std::{error::Error, sync::Arc};
use vkfft::config::Config;
use vkfft::context::{Context, FftType};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::instance::{Instance, InstanceExtensions};

const DEFAULT_BUFFER_USAGE: BufferUsage = BufferUsage {
  storage_buffer: true,
  transfer_source: true,
  transfer_destination: true,
  ..BufferUsage::none()
};

fn real_to_complex_2d(instance: &Arc<Instance>) -> Result<(), Box<dyn Error>> {
  let k_x_coord = 2;
  let k_y_coord = 3;
  println!("------------");
  println!("Performing 2D real-to-complex FFT. The plane wave should localize to position [{k_x_coord}, {k_y_coord}]:");

  let mut context = Context::new(instance)?;

  let size = [8, 8];
  let size_fft = [2 * (size[0] / 2 + 1), size[1]];
  let buffer_size = size_fft[0] * size_fft[1];

  let data = CpuAccessibleBuffer::from_iter(
    context.device.clone(),
    DEFAULT_BUFFER_USAGE,
    false,
    (0..buffer_size).map(|_| 0.0f32),
  )?;

  let k_x = k_x_coord as f32 * std::f32::consts::TAU / size[0] as f32;
  let k_y = k_y_coord as f32 * std::f32::consts::TAU / size[1] as f32;
  data.write()?.iter_mut().enumerate().for_each(|(i, val)| {
    let x = (i % size[0] as usize) as f32;
    let y = (i / size[0] as usize) as f32;
    *val = (k_x * x + k_y * y).cos()
  });
  println!("Data:");
  print_matrix_buffer(&data, &size);

  let config_builder = Config::builder()
    .input_buffer(data.clone())
    .buffer(data.clone())
    .input_formatted(true)
    .r2c()
    .dim(&size);

  context.single_fft(config_builder, FftType::Forward)?;

  println!("Transformed data:");
  print_complex_matrix_buffer(&data, &size_fft);
  Ok(())
}

fn complex_to_complex_1d(instance: &Arc<Instance>) -> Result<(), Box<dyn Error>> {
  let k_coord = 2;
  println!("------------");
  println!(
    "Performing 1D complex-to-complex FFT. The plane wave should localize to position [{k_coord}]:"
  );

  let mut context = Context::new(instance)?;

  let size = [8];
  let buffer_size = 2 * size[0];
  let printing_size = [buffer_size, 1];

  let data = CpuAccessibleBuffer::from_iter(
    context.device.clone(),
    DEFAULT_BUFFER_USAGE,
    false,
    (0..buffer_size).map(|_| 0.0f32),
  )?;

  let k_x = k_coord as f32 * std::f32::consts::TAU / size[0] as f32;
  data.write()?.iter_mut().enumerate().for_each(|(i, val)| {
    let x = (i as usize / 2usize) as f32;
    if i % 2 == 0 {
      *val = (k_x * x).cos()
    } else {
      *val = (k_x * x).sin()
    }
  });
  println!("Data:");
  print_complex_matrix_buffer(&data, &printing_size);

  let config_builder = Config::builder()
    .input_buffer(data.clone())
    .buffer(data.clone())
    .dim(&size);

  context.single_fft(config_builder, FftType::Forward)?;

  println!("Transformed data:");
  print_complex_matrix_buffer(&data, &printing_size);

  let config_builder = Config::builder()
    .input_buffer(data.clone())
    .buffer(data.clone())
    .normalize()
    .dim(&size);

  //note: in a real application, multiple calls to single_fft are not recommended.
  // instead, re-use an existing vkfft app and vulkan command buffer as they are
  // defined within this function.
  context.single_fft(config_builder, FftType::Inverse)?;
  println!("After inverse transform:");
  print_complex_matrix_buffer(&data, &printing_size);
  Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
  println!("VkFFT version: {}", vkfft::version());

  let instance = Instance::new(
    None,
    vulkano::Version { major: 14, minor: 0, patch: 0 },
    &InstanceExtensions {
      ext_debug_utils: false,
      ..InstanceExtensions::none()
    },
    None,
  )?;

  complex_to_complex_1d(&instance)?;
  real_to_complex_2d(&instance)?;

  Ok(())
}

/// Prints a 2D matrix contained in a Vulkano buffer
fn print_matrix_buffer(buffer: &Arc<CpuAccessibleBuffer<[f32]>>, shape: &[u32; 2]) {
  buffer
    .read()
    .unwrap()
    .iter()
    .take((shape[0] * shape[1]) as usize)
    .enumerate()
    .for_each(|(i, &value)| {
      print!("{:>5.1} ", value);
      if (i + 1) as u32 % shape[0] == 0 {
        println!();
      }
    });
}

/// Prints a 2D matrix contained in a Vulkano buffer representing complex numbers in the
/// format (re, im) (re, im) ...
/// It is assumed that the even indicies are the real parts, and odd indicies are
/// the imaginary parts.
fn print_complex_matrix_buffer(buffer: &Arc<CpuAccessibleBuffer<[f32]>>, shape: &[u32; 2]) {
  buffer
    .read()
    .unwrap()
    .iter()
    .take((shape[0] * shape[1]) as usize)
    .enumerate()
    .for_each(|(i, &value)| {
      if i % 2 == 0 {
        print!("({:>5.1},", value);
      } else {
        print!("{:>5.1}) ", value);
      }
      if (i + 1) as u32 % shape[0] == 0 {
        println!();
      }
    });
}
