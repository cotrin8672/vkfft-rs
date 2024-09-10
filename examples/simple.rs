use std::{error::Error, sync::Arc};
use vkfft::config::Config;
use vkfft::context::{Context, FftType};
use vulkano::buffer::{subbuffer::Subbuffer, Buffer};
use vulkano::buffer::{BufferCreateInfo, BufferUsage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};

fn real_to_complex_2d(instance: &Arc<Instance>) -> Result<(), Box<dyn Error>> {
  let k_x_coord = 2;
  let k_y_coord = 3;
  println!("------------");
  println!("Performing 2D real-to-complex FFT. The plane wave should localize to position [{k_x_coord}, {k_y_coord}]:");

  let mut context = Context::new(instance)?;

  let size = [8, 8];
  let size_fft = [2 * (size[0] / 2 + 1), size[1]];
  let buffer_size = size_fft[0] * size_fft[1];
  let allocator = Arc::new(
    vulkano::memory::allocator::StandardMemoryAllocator::new_default(context.device.clone()),
  );
  let data = Buffer::from_iter(
    allocator,
    BufferCreateInfo {
      usage: BufferUsage::TRANSFER_DST,
      ..Default::default()
    },
    AllocationCreateInfo {
      memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
      ..Default::default()
    },
    (0..buffer_size as u32).map(|_| 0.0f32),
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
    .input_buffer(data.buffer().clone())
    .buffer(data.buffer().clone())
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

  let allocator = Arc::new(
    vulkano::memory::allocator::StandardMemoryAllocator::new_default(context.device.clone()),
  );
  let data = Buffer::from_iter(
    allocator,
    BufferCreateInfo {
      usage: BufferUsage::TRANSFER_DST,
      ..Default::default()
    },
    AllocationCreateInfo {
      memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
      ..Default::default()
    },
    (0..buffer_size as u32).map(|_| 0.0f32),
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
    .input_buffer(data.buffer().clone())
    .buffer(data.buffer().clone())
    .dim(&size);

  context.single_fft(config_builder, FftType::Forward)?;

  println!("Transformed data:");
  print_complex_matrix_buffer(&data, &printing_size);

  let config_builder = Config::builder()
    .input_buffer(data.buffer().clone())
    .buffer(data.buffer().clone())
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

  let library = vulkano::VulkanLibrary::new().expect("no local Vulkan library/DLL");
  let instance =
    Instance::new(library, InstanceCreateInfo::default()).expect("failed to create instance");
  complex_to_complex_1d(&instance)?;
  real_to_complex_2d(&instance)?;

  Ok(())
}

/// Prints a 2D matrix contained in a Vulkano buffer
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

/// Prints a 2D matrix contained in a Vulkano buffer representing complex numbers in the
/// format (re, im) (re, im) ...
/// It is assumed that the even indicies are the real parts, and odd indicies are
/// the imaginary parts.
fn print_complex_matrix_buffer(buffer: &Subbuffer<[f32]>, shape: &[u32; 2]) {
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
