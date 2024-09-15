use std::error::Error;
use vkfft::config::Config;
use vkfft::context::{Context, FftType};
use vulkano::buffer::subbuffer::Subbuffer;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions};

fn main() -> Result<(), Box<dyn Error>> {
  println!("VkFFT version: {}", vkfft::version());

  // These first steps will always take place in a Vulkan program; first the library has to be loaded, then the instance created
  let library = vulkano::VulkanLibrary::new().expect("no local Vulkan library/DLL");

  // The instance created here has the ENUMERATE_PORTABILITY flag enabled, as well as two Instance extensions. These
  // are required for the instance to be used on MacOS via the MoltenVK compatibility layer.
  // This means that Vulkan will accept non-fully-conformant targets; if it is using the wrong hardware on your target
  // system, removing them might help.
  let instance = Instance::new(
    library,
    InstanceCreateInfo {
      flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
      enabled_extensions: InstanceExtensions {
        khr_get_physical_device_properties2: true,
        khr_portability_enumeration: true,
        ..Default::default()
      },
      ..Default::default()
    },
  )
  .expect("failed to create instance");

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

/// # Complex-to-complex FFT
///
/// Here we perform a simple test using a Fourier transform with a known result,
/// that the transform of the function exp(i k_x x) should be a delta function
/// in Fourier space.
///
/// We pass in an existing context, prepare the necessary buffer, configure
/// our FFT plan, and then apply it to the buffer.
///
/// For good measure, we transform back and forth thousands of times to make sure
/// that the Fourier inversion theorem holds
fn complex_to_complex_1d(context: &Context) -> Result<(), Box<dyn Error>> {
  let k_coord = 2;
  println!("================================================================================");
  println!(
    "Performing 1D complex-to-complex FFT.\nThe plane wave should localize to position [{k_coord}]\n:"
  );

  // The 1D grid will have 12 complex values, which requires a buffer with twice
  // as many points, since the buffer will have an [f32] type, with real and imaginary
  // parts given on the even and odd index values, respectively.
  let size = [12];
  let buffer_size = 2 * size[0];

  // The simple printing function defined below expects a matrix, so we put the buffer
  // size into one for convenience
  let printing_size = [buffer_size, 1];

  // The Context we passed in can be used to create a buffer from a simple iterator.
  let data = context.new_buffer_from_iter((0..buffer_size as u32).map(|_| 0.0f32))?;

  // k_x will be the floating point value of k that corresponds with the k-space grid coordinates
  // i.e. if we want the delta function to show up on the second point, we have to multiply 2 by
  // 2*pi/N
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

  // We have to specify a few things in the VkFFT config:
  // - the input buffer, which is provided by the Subbuffer object that the Context (via Vulkano)
  //   gave us using the .buffer() impl. We clone the Arc containing it.
  // - we tell VkFFT that the buffer where it should work is also the input buffer, i.e. we're
  //   doing an in-place FFT.
  // - The size of the system in an array with the dimensions of the FFT, just 1D here.
  let config_builder = Config::builder()
    .input_buffer(data.buffer().clone())
    .buffer(data.buffer().clone())
    .dim(&size);

  // In this simple case, we can just use the single_fft impl of the context to provide
  // the transform, and then look at the buffer.
  context.single_fft(config_builder, FftType::Forward)?;

  println!("Transformed data:");
  print_complex_matrix_buffer(&data, &printing_size);

  // Typically we are using a GPU library because there is a heavier workload that needs
  // to be done, often many times. In that case we don't want the initialization overhead
  // of creating an FFT plan each time. Instead we'll use the same VkFFT "App", and put
  // many transforms on the same Vulkan command buffer.
  let config_builder = Config::builder()
    .input_buffer(data.buffer().clone())
    .buffer(data.buffer().clone())
    .normalize()
    .dim(&size);
  let (mut app, mut params, builder) = context.start_fft_chain(config_builder, FftType::Inverse)?;
  for _ in 0..4095 {
    (app, params) = context.chain_fft_with_app(app, params, FftType::Forward)?;
    (app, params) = context.chain_fft_with_app(app, params, FftType::Inverse)?;
  }
  // we submit the command buffer to Vulkan to run on the GPU
  // Note that it is not necessary to do all of this through the Context we define here,
  // but this step especially is somewhat tricky in Vulkano since passing the command
  // buffer to a C library like VkFFT requires an UnsafeCommandBuffer struct, which
  // needs to be submitted to a Vulkan queue using low-level unsafe functions.
  context.submit(builder.build()?)?;

  println!("After 4096 forward and inverse transforms:");
  print_complex_matrix_buffer(&data, &printing_size);
  Ok(())
}

/// # 2D real-to-complex FFT
/// Here we start with real-valued data, and can save about half the required
/// memory since we can exploit the symmetry of the Fourier transform of real
/// data: the negative-k_x half-plane will just be the conjugate of the positive one,
/// so we don't need to store both. Accordingly, for an N x M 2D array, the dimensions
/// in Fourier space will be (N/2 + 1) x M. However, these numbers are complex, so the
/// memory occupied will be slightly larger than real space.
/// This function shows us where the various positions in k-space are in this reduced grid,
/// by again defining a plane wave, just a 2D one, which will also localize to a single
/// point in the Fourier grid.
fn real_to_complex_2d(context: &Context) -> Result<(), Box<dyn Error>> {
  //we define the position of the output delta function
  let k_x_coord = 2;
  let k_y_coord = 3;
  println!("================================================================================");
  println!("Performing 2D real-to-complex FFT.\nThe plane wave should localize to position [{k_x_coord}, {k_y_coord}]:\n");

  //The size array now has two elements
  let size = [8, 8];
  //The size of the FFT is as stated in the header; we store it for printing later
  //The factor 2 multiplying the leading dimension is because we are storing complex values
  let size_fft = [2 * (size[0] / 2 + 1), size[1]];
  let buffer_size = size_fft[0] * size_fft[1];

  //We obtain the data array as in the previous example, as one block of memory
  let data = context.new_buffer_from_iter((0..buffer_size as u32).map(|_| 0.0f32))?;

  //We have two spatial frequencies/coordinates now
  let k_x = k_x_coord as f32 * std::f32::consts::TAU / size[0] as f32;
  let k_y = k_y_coord as f32 * std::f32::consts::TAU / size[1] as f32;
  data.write()?.iter_mut().enumerate().for_each(|(i, val)| {
    let x = (i % size[0] as usize) as f32;
    let y = (i / size[0] as usize) as f32;
    *val = (k_x * x + k_y * y).cos()
  });
  println!("Data:");
  print_matrix_buffer(&data, &size);

  //The configuration step looks similar, we just have to call:
  // - r2c(), which tells VkFFT to perform the real-to-complex transform
  // - input_formatted(), which tells VkFFT that the array in the input buffer is
  //   contiguous, we aren't adding padding at the ends
  let config_builder = Config::builder()
    .input_buffer(data.buffer().clone())
    .buffer(data.buffer().clone())
    .input_formatted(true)
    .r2c()
    .dim(&size);

  context.single_fft(config_builder, FftType::Forward)?;

  println!("Transformed data:");
  print_complex_matrix_buffer(&data, &size_fft);

  // after performing the forward transform, we do an inverse one to check that everything
  // is in order. I'm only doing this so we can stop and  have a look at the data in the buffer,
  // otherwise it would be best to put these in the same command buffer. Note that we use a
  // similar set of commands to tell vkfft to put formatted data in the output buffer.
  let config_builder_inverse = Config::builder()
    .output_buffer(data.buffer().clone())
    .buffer(data.buffer().clone())
    .r2c()
    .output_formatted(true)
    .normalize()
    .dim(&size);

  context.single_fft(config_builder_inverse, FftType::Inverse)?;
  println!("Transforming back:");
  print_matrix_buffer(&data, &size);
  Ok(())
}

/// # Convolution
/// A common use of FFTs is to exploit the convolution theorem, which says that a convolution
/// in a given domain is a multiplication in its Fourier-conjugate domain. VkFFT has some optimizations
/// that allow this to take place more efficiently than if we were to do the multiplication step separately.
fn convolution(context: &Context) -> Result<(), Box<dyn Error>> {
  println!("================================================================================");
  println!("Perform a 2D convolution:\nCircular shift operation\n");
  // In doing convolutions, we may want to have a multi-element feature vector; in this case we don't
  // since we're just doing a simple scalar convolution
  let coordinate_features = 1;

  //We will again do a 2D real-to-complex transform, as is common in image filtering.
  let size = [8, 8];
  let size_fft = [2 * (size[0] / 2 + 1), size[1]];
  let buffer_size = coordinate_features * size_fft[0] * size_fft[1];

  //We get two buffers, one for the data, and the other for the kernel (the thing we're convolving the data with)
  let data = context.new_buffer_from_iter((0..buffer_size as u32).map(|_| 0.0f32))?;
  let kernel = context.new_buffer_from_iter((0..buffer_size as u32).map(|_| 0.0f32))?;

  //We'll just put a delta function in the data array
  data.write()?.iter_mut().enumerate().for_each(|(i, val)| {
    if i == 20 {
      *val = 100.0f32;
    }
  });

  //and another delta function in the kernel array. We can (circularly) shift the
  //data by the position of this; here we move it one pixel to the right
  kernel.write()?.iter_mut().enumerate().for_each(|(i, val)| {
    if i == 1 {
      *val = 1.0f32
    }
  });

  println!("Data:");
  print_matrix_buffer(&data, &size);

  // First we set up a plan for transforming the kernel, which has two additional function calls:
  // - coordinate_features(), which tells VkFFT the size of the feature vector
  // - kernel_convolution(), which tells it that we are preparing a kernel for convolution (not
  //   actually required right now, but you might need it for more complicated convolutions)
  let config_builder_kernel = Config::builder()
    .input_buffer(kernel.buffer().clone())
    .buffer(kernel.buffer().clone())
    .input_formatted(true)
    .r2c()
    .coordinate_features(coordinate_features)
    .kernel_convolution()
    .dim(&size);

  //we are going to run everything on one command buffer
  let (_app, _params, builder) =
    context.start_fft_chain(config_builder_kernel, FftType::Forward)?;

  // Next, we build the plan for the convolution. Here we have to call:
  // - input_buffer() with input_formatted() again
  // - inverse_return_to_input() so we get the output as it should look
  // - convolution() to tell VkFFT to do the convolution step
  // - kernel() to point VkFFT to the kernel buffer
  // - normalize() so that the output doesn't get multiplied by the size of the array
  let config_builder_convolution = Config::builder()
    .input_buffer(data.buffer().clone())
    .buffer(data.buffer().clone())
    .kernel(kernel.buffer().clone())
    .convolution()
    .coordinate_features(coordinate_features)
    .r2c()
    .input_formatted(true)
    .inverse_return_to_input()
    .normalize()
    .dim(&size);

  let (_app, _params, builder) =
    context.chain_fft_with_config(config_builder_convolution, builder, FftType::Forward)?;
  context.submit(builder.build()?)?;
  println!("Convolved:");
  print_matrix_buffer(&data, &size);
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
