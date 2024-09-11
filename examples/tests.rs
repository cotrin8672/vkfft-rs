use std::error::Error;
use vkfft::config::Config;
use vkfft::context::{Context, FftType};
use vulkano::buffer::subbuffer::Subbuffer;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions};

fn main() -> Result<(), Box<dyn Error>> {
  println!("VkFFT version: {}", vkfft::version());

  // These first steps will always take place in a Vulkan program; first the library has to be loaded, then the instance created
  let library = vulkano::VulkanLibrary::new().expect("no local Vulkan library/DLL");

  //The instance created here has the ENUMERATE_PORTABILITY flag enabled, as well as two Instance extensions. These
  //are required for the instance to be used on MacOS via the MoltenVK compatibility layer.
  let instance =
    Instance::new(library, 
      InstanceCreateInfo {
        flags: InstanceCreateFlags::ENUMERATE_PORTABILITY, 
        enabled_extensions: InstanceExtensions{khr_get_physical_device_properties2: true, khr_portability_enumeration: true, ..Default::default()},
        ..Default::default()}).expect("failed to create instance");
  
  //The Context struct is provided to contain a number of required elements for the Vulkan instance
  //to be used with VkFFT. The new() function creates one with reasonable defaults. However, it is not
  //required to use this struct: one can independently create the required elements, e.g. if integrating
  //VkFFT in a Vulkano toolchain.
  let context = Context::new(&instance)?;

  //Example ffts:
  complex_to_complex_1d(&context)?;
  real_to_complex_2d(&context)?;
  convolution(&context)?;
  Ok(())
}

fn complex_to_complex_1d(context: &Context) -> Result<(), Box<dyn Error>> {
  let k_coord = 2;
  println!("------------");
  println!(
    "Performing 1D complex-to-complex FFT. The plane wave should localize to position [{k_coord}]:"
  );

  let size = [12];
  let buffer_size = 2 * size[0];
  let printing_size = [buffer_size, 1];

  let data = context.new_buffer_from_iter((0..buffer_size as u32).map(|_| 0.0f32))?;

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

  //perform many FFTs in a row on the same command buffer
  let (mut app, mut params, builder) = context.start_fft_chain(config_builder, FftType::Inverse)?;
  for _ in 0..4095{
    (app, params) = context.chain_fft_with_app(app, params, FftType::Forward)?;
    (app, params) = context.chain_fft_with_app(app, params, FftType::Inverse)?;
  }
  context.submit(builder.build()?)?;
  println!("After 4096 forward and inverse transforms:");
  print_complex_matrix_buffer(&data, &printing_size);
  Ok(())
}

fn real_to_complex_2d(context: &Context) -> Result<(), Box<dyn Error>> {
  let k_x_coord = 2;
  let k_y_coord = 3;
  println!("------------");
  println!("Performing 2D real-to-complex FFT. The plane wave should localize to position [{k_x_coord}, {k_y_coord}]:");

  let size = [8, 8];
  let size_fft = [2 * (size[0] / 2 + 1), size[1]];
  let buffer_size = size_fft[0] * size_fft[1];

  let data = context.new_buffer_from_iter((0..buffer_size as u32).map(|_| 0.0f32))?;

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
    .output_buffer(data.buffer().clone())
    .buffer(data.buffer().clone())
    .input_formatted(true)
    .output_formatted(false)
    .r2c()
    .dim(&size);

  context.single_fft(config_builder, FftType::Forward)?;

  println!("Transformed data:");
  print_complex_matrix_buffer(&data, &size_fft);

  let config_builder2 = Config::builder()
  .input_buffer(data.buffer().clone())
  .output_buffer(data.buffer().clone())
  .buffer(data.buffer().clone())
  .r2c()
  .input_formatted(false)
  .output_formatted(true)
  .normalize()
  .dim(&size);

  context.single_fft(config_builder2, FftType::Inverse)?;
  println!("Transforming back:");
  print_matrix_buffer(&data, &size);
  Ok(())
}

fn convolution(context: &Context) -> Result<(), Box<dyn Error>> {

  let size = [12];
  let buffer_size = 2 * size[0];
  let printing_size = [buffer_size, 1];

  let data = context.new_buffer_from_iter((0..buffer_size as u32).map(|_| 0.0f32))?;
  let kernel = context.new_buffer_from_iter((0..buffer_size as u32).map(|_| 0.0f32))?;


  data.write()?.iter_mut().enumerate().for_each(|(i, val)| {
    if i == 2 { *val = 1.0f32;}
  });
  kernel.write()?.iter_mut().enumerate().for_each(|(i, val)| {
    if i == 6 { *val = 1.0f32;}
  });
  println!("Data:");
  print_complex_matrix_buffer(&data, &printing_size);

  let config_builder_kernel = Config::builder()
    .input_buffer(kernel.buffer().clone())
    .dim(&size)
    .kernel_convolution()
    .buffer(kernel.buffer().clone());
  context.single_fft(config_builder_kernel, FftType::Forward)?;

  println!("Transformed kernel:");
  print_complex_matrix_buffer(&kernel, &printing_size);

  let config_builder_convolution = Config::builder()
    .input_buffer(data.buffer().clone())
    .buffer(data.buffer().clone())
    .kernel(kernel.buffer().clone())
    .dim(&size)
    .convolution()
    .coordinate_features(1)
    .normalize();

  
  let (app, params, builder) = context.start_fft_chain(config_builder_convolution, FftType::Forward)?;
  context.chain_fft_with_app(app, params, FftType::Inverse)?;
  context.submit(builder.build()?)?;

  println!("Convolved data:");
  print_complex_matrix_buffer(&data, &printing_size);

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
