use vkfft::app::App;
use vkfft::app::LaunchParams;
use vkfft::config::Config;

use vkfft::context::FftType;
use vulkano::buffer::Subbuffer;
use vulkano::buffer::{BufferUsage, Buffer};
use vulkano::command_buffer::{
  sys::UnsafeCommandBufferBuilder,
};

use vulkano::instance::{Instance, InstanceExtensions};
use vkfft::context::Context;
use std::{error::Error, sync::Arc};

use util::SizeIterator;



/// Transform a kernel from spatial data to frequency data
pub fn transform_kernel(
  context: &mut Context,
  coordinate_features: u32,
  batch_count: u32,
  size: &[u32; 2],
  kernel: &Arc<Buffer>,
) -> Result<(), Box<dyn Error>> {
  // Configure kernel FFT
  let config = Config::builder()
    .buffer(kernel.clone())
    .command_pool(context.pool.clone())
    .kernel_convolution()
    .normalize()
    .coordinate_features(coordinate_features)
    .batch_count(1)
    .r2c()
    .disable_reorder_four_step()
    .dim(&size);
  context.single_fft(config, FftType::Forward)?;

  Ok(())
}

pub fn convolve(
  context: &mut Context,
  coordinate_features: u32,
  size: &[u32; 2],
  kernel: &Arc<Buffer>,
) -> Result<(), Box<dyn Error>> {
  let input_buffer_size = coordinate_features * 2 * (size[0] / 2 + 1) * size[1];
  let buffer_size = coordinate_features * 2 * (size[0] / 2 + 1) * size[1];

  let input_buffer = context.new_buffer_from_iter((0..input_buffer_size).map(|_| 0.0f32))?;
  let buffer = context.new_buffer_from_iter((0..buffer_size).map(|_| 0.0f32))?;

  {
    let mut buffer = input_buffer.write()?;

    for v in 0..coordinate_features {
      for [i, j] in SizeIterator::new(size) {
        let _0 = i + j * (size[0] / 2) + v * (size[0] / 2) * size[1];
        buffer[_0 as usize] = 1.0f32;
      }
    }
  }

  println!("Buffer:");
  print_matrix_buffer(&input_buffer, &size);
  //println!("{}", MatrixFormatter::new(size, &input_buffer));
  println!();

  // Configure kernel FFT
  let conv_config = Config::builder()
    .input_buffer(input_buffer.buffer().clone())
    .buffer(buffer.buffer().clone())
    .convolution()
    .kernel(kernel.clone())
    .normalize()
    .coordinate_features(coordinate_features)
    .batch_count(1)
    .r2c()
    .disable_reorder_four_step()
    .input_formatted(true)
    .dim(&size);


  context.single_fft(conv_config, FftType::Forward);


  println!("Result:");
  print_matrix_buffer(&buffer, &size);
  //println!("{}", MatrixFormatter::new(size, &buffer));
  println!();

  Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
  println!("VkFFT version: {}", vkfft::version());

  let library = vulkano::VulkanLibrary::new().expect("no local Vulkan library/DLL");

  let instance =
    Instance::new(library, 
      vulkano::instance::InstanceCreateInfo {
        flags: vulkano::instance::InstanceCreateFlags::ENUMERATE_PORTABILITY, 
        enabled_extensions: InstanceExtensions{khr_get_physical_device_properties2: true, khr_portability_enumeration: true, ..Default::default()},
        ..Default::default()}).expect("failed to create instance");

  let mut context = Context::new(&instance)?;

  let batch_count = 2;
  let coordinate_features = 2;
  let size = [32, 32];

  let kernel_size = batch_count * coordinate_features * 2 * (size[0] / 2 + 1) * size[1];

  let kernel = context.new_buffer_from_iter(
    (0..kernel_size).map(|_| 0.0f32))?;

  {
    let mut kernel_input = kernel.write()?;

    let mut range = size;
    range[0] = range[0] / 2 + 1;

    for f in 0..batch_count {
      for v in 0..coordinate_features {
        for [i, j] in SizeIterator::new(&range) {
          println!("{} {}", i, j);
          let _0 = 2 * i
            + j * (size[0] + 2)
            + v * (size[0] + 2) * size[1]
            + f * coordinate_features * (size[0] + 2) * size[1];
          let _1 = 2 * i
            + 1
            + j * (size[0] + 2)
            + v * (size[0] + 2) * size[1]
            + f * coordinate_features * (size[0] + 2) * size[1];
          kernel_input[_0 as usize] = (f * coordinate_features + v + 1) as f32;
          kernel_input[_1 as usize] = 0.0f32;
        }
      }
    }
  }

  println!("Kernel:");
  print_matrix_buffer(&kernel, &size);
  //println!("{}", &MatrixFormatter::new(&size, &kernel));
  println!();

  transform_kernel(
    &mut context,
    coordinate_features,
    batch_count,
    &size,
    &kernel.buffer().clone(),
  )?;

  println!("Transformed Kernel:");
  print_matrix_buffer(&kernel, &size);
  //println!("{}", &MatrixFormatter::new(&size, &kernel));
  println!();

  convolve(&mut context, coordinate_features, &size, &kernel.buffer().clone())?;

  Ok(())
}

fn print_matrix_buffer(buffer: &Subbuffer<[f32]>, shape: &[u32; 2]) {
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