use vkfft::app::App;
use vkfft::app::LaunchParams;
use vkfft::config::Config;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{
  sys::{Flags, UnsafeCommandBufferBuilder},
  Kind,
};

use vulkano::instance::{Instance, InstanceExtensions};

use std::{error::Error, sync::Arc};

use util::{Context, MatrixFormatter, SizeIterator};

const DEFAULT_BUFFER_USAGE: BufferUsage = BufferUsage {
  storage_buffer: true,
  transfer_source: true,
  transfer_destination: true,
  ..BufferUsage::none()
};

fn main() -> Result<(), Box<dyn Error>> {
  println!("VkFFT version: {}", vkfft::version());

  let instance = Instance::new(
    None,
    &InstanceExtensions {
      ext_debug_utils: false,
      ..InstanceExtensions::none()
    },
    None,
  )?;

  let mut context = Context::new(&instance)?;
  let size = [16, 16];
  let size_fft = [2 * (size[0] / 2 + 1), size[1]];
  let buffer_size = size_fft[0] * size_fft[1];

  let data = CpuAccessibleBuffer::from_iter(
    context.device.clone(),
    DEFAULT_BUFFER_USAGE,
    false,
    (0..buffer_size).map(|_| 0.0f32),
  )?;

  let output_data = CpuAccessibleBuffer::from_iter(
    context.device.clone(),
    DEFAULT_BUFFER_USAGE,
    false,
    (0..buffer_size).map(|_| 0.0f32),
  )?;

  let k_x = 2.0f32 * std::f32::consts::TAU / size[0] as f32;
  let k_y = 1.0f32 * std::f32::consts::TAU / size[1] as f32;
  data.write()?.iter_mut().enumerate().for_each(|(i, val)| {
    let x = (i % size[0] as usize) as f32;
    let y = (i / size[0] as usize) as f32;
    *val = (k_x * x + k_y * y).cos()
  });

  println!("Data:");
  print_matrix_buffer(&data, &size);
  let config = Config::builder()
    .physical_device(context.physical)
    .device(context.device.clone())
    .fence(&context.fence)
    .queue(context.queue.clone())
    .input_buffer(data.clone())
    .buffer(data.clone())
    .input_formatted(true)
    .command_pool(context.pool.clone())
    .r2c()
    .dim(&size)
    .build()?;

  // Allocate a command buffer
  let primary_cmd_buffer = context.alloc_primary_cmd_buffer()?;

  // Create command buffer handle
  let builder =
    unsafe { UnsafeCommandBufferBuilder::new(&primary_cmd_buffer, Kind::primary(), Flags::None)? };

  // Configure FFT launch parameters
  let mut params = LaunchParams::builder().command_buffer(&builder).build()?;

  // Construct FFT "Application"
  let mut app = App::new(config)?;

  // Run forward FFT
  app.forward(&mut params)?;

  // Dispatch command buffer and wait for completion
  let command_buffer = builder.build()?;
  context.submit(command_buffer)?;

  println!("Transformed data:");
  print_complex_matrix_buffer(&data, &size_fft);
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
/// It is assumed that the even indicies are the real parts, and imaginary indicies are
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
