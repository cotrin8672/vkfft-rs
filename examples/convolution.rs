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


use util::Context;

const DEFAULT_BUFFER_USAGE: BufferUsage = BufferUsage {
  storage_buffer: true,
  transfer_source: true,
  transfer_destination: true,
  ..BufferUsage::none()
};

/// Transform a kernel from spatial data to frequency data
pub fn transform_kernel(
  context: &mut Context,
  coordinate_features: u32,
  batch_count: u32,
  size: &[u32; 2],
  kernel: &Arc<CpuAccessibleBuffer<[f32]>>,
) -> Result<(), Box<dyn Error>> {
  // Configure kernel FFT
  let config = Config::builder()
    .physical_device(context.physical)
    .device(context.device.clone())
    .fence(&context.fence)
    .queue(context.queue.clone())
    .buffer(kernel.clone())
    .command_pool(context.pool.clone())
    .kernel_convolution()
    .normalize()
    .coordinate_features(coordinate_features)
    .batch_count(batch_count)
    .r2c()
    .disable_reorder_four_step()
    .dim(&size)
    .build()?;


  // Allocate a command buffer
  let primary_cmd_buffer = context.alloc_primary_cmd_buffer()?;

  // Create command buffer handle
  let builder =
    unsafe { UnsafeCommandBufferBuilder::new(&primary_cmd_buffer, Kind::primary(), Flags::None)? };

  // Configure FFT launch parameters
  let mut params = LaunchParams::builder()
    .command_buffer(&builder)
    .build()?;

  // Construct FFT "Application"
  let mut app = App::new(config)?;

  // Run forward FFT
  app.forward(&mut params)?;

  // Dispatch command buffer and wait for completion
  let command_buffer = builder.build()?;
  context.submit(command_buffer)?;

  Ok(())
}

pub fn convolve(
  context: &mut Context,
  coordinate_features: u32,
  size: &[u32; 2],
  kernel: &Arc<CpuAccessibleBuffer<[f32]>>,
) -> Result<(), Box<dyn Error>> {
  // Configure kernel FFT
  let config = Config::builder()
    .physical_device(context.physical)
    .device(context.device.clone())
    .fence(&context.fence)
    .queue(context.queue.clone())
    .buffer(kernel.clone())
    .command_pool(context.pool.clone())
    .convolution()
    .kernel(kernel.clone())
    .normalize()
    .coordinate_features(coordinate_features)
    .batch_count(1)
    .r2c()
    .disable_reorder_four_step()
    .dim(&size)
    .build()?;


  // Allocate a command buffer
  let primary_cmd_buffer = context.alloc_primary_cmd_buffer()?;

  // Create command buffer handle
  let builder =
    unsafe { UnsafeCommandBufferBuilder::new(&primary_cmd_buffer, Kind::primary(), Flags::None)? };

  // Configure FFT launch parameters
  let mut params = LaunchParams::builder()
    .command_buffer(&builder)
    .build()?;

  // Construct FFT "Application"
  let mut app = App::new(config)?;

  // Run forward FFT
  app.forward(&mut params)?;

  // Dispatch command buffer and wait for completion
  let command_buffer = builder.build()?;
  context.submit(command_buffer)?;

  Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
  println!("VkFFT version: {}", vkfft::version());

  let instance = Instance::new(
    None,
    &InstanceExtensions {
      ext_debug_utils: true,
      ..InstanceExtensions::none()
    },
    vec!["VK_LAYER_KHRONOS_validation"],
  )?;

  let mut context = Context::new(&instance)?;

  let batch_count = 2;
  let coordinate_features = 2;
  let size = [32, 32];

  let kernel_size = batch_count * coordinate_features * 2 * (size[0] / 2 + 1) * size[1];

  let kernel = CpuAccessibleBuffer::from_iter(
    context.device.clone(),
    DEFAULT_BUFFER_USAGE,
    false,
    (0..kernel_size).map(|_| 0.0f32),
  )?;

  {
    let mut kernel_input = kernel.write()?;
    for f in 0..batch_count {
      for v in 0..coordinate_features {
        for j in 0..size[1] {
          for i in 0..size[0] / 2 {
            let _0 = 2 * i + j * (size[0] + 2) + 0 * (size[0] + 2) * size[1] + v * (size[0] + 2) * size[1] + f * coordinate_features * (size[0] + 2) * size[1];
            let _1 = 2 * i + 1 + j * (size[0] + 2) + 0 * (size[0] + 2) * size[1] + v * (size[0] + 2) * size[1] + f * coordinate_features * (size[0] + 2) * size[1];
            kernel_input[_0 as usize] = (f * coordinate_features + v + 1) as f32;
            kernel_input[_1 as usize] = 0.0f32;
          }
        }
      }
    }
  }
  

  transform_kernel(
    &mut context,
    coordinate_features,
    batch_count,
    &size,
    &kernel,
  )?;

  Ok(())
}
