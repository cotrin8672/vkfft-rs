use std::sync::Arc;

use derive_more::{Display, Error};
use std::pin::Pin;
use vulkano::{
  buffer::Buffer,
  command_buffer::pool::CommandPool,
  device::physical::PhysicalDevice,
  device::{Device, Queue},
  sync::fence::Fence,
  VulkanObject,
};

use std::ptr::addr_of_mut;

#[derive(Display, Debug, Error)]
pub enum BuildError {
  NoPhysicalDevice,
  NoDevice,
  NoQueue,
  NoFence,
  NoCommandPool,
  NoBuffer,
}

pub struct ConfigBuilder<'a> {
  fft_dim: u32,
  size: [u32; 4usize],

  physical_device: Option<Arc<PhysicalDevice>>,
  device: Option<Arc<Device>>,
  queue: Option<Arc<Queue>>,
  fence: Option<&'a Fence>,
  command_pool: Option<Arc<CommandPool>>,
  buffer: Option<Arc<Buffer>>,
  input_buffer: Option<Arc<Buffer>>,
  output_buffer: Option<Arc<Buffer>>,
  temp_buffer: Option<Arc<Buffer>>,
  kernel: Option<Arc<Buffer>>,
  normalize: bool,
  zero_padding: [bool; 3usize],
  zeropad_left: [u32; 4usize],
  zeropad_right: [u32; 4usize],
  kernel_convolution: bool,
  convolution: bool,
  r2c: bool,
  dct: Option<u64>,
  dst: Option<u64>,
  coordinate_features: u32,
  disable_reorder_four_step: bool,
  batch_count: Option<u32>,
  precision: Precision,
  use_lut: bool,
  symmetric_kernel: bool,
  input_formatted: Option<bool>,
  inverse_return_to_input: Option<bool>,
  output_formatted: Option<bool>,
  matrix_convolution: Option<u64>,
}
impl<'a> Default for ConfigBuilder<'a> {
  fn default() -> Self {
      Self::new()
  }
}
impl<'a> ConfigBuilder<'a> {
  pub fn new() -> Self {
    Self {
      fft_dim: 1,
      size: [1, 1, 1, 0],
      physical_device: None,
      device: None,
      queue: None,
      fence: None,
      command_pool: None,
      normalize: false,
      zero_padding: [false, false, false],
      zeropad_left: [0, 0, 0, 0],
      zeropad_right: [0, 0, 0, 0],
      kernel_convolution: false,
      r2c: false,
      dct: None,
      dst: None,
      coordinate_features: 1,
      disable_reorder_four_step: false,
      buffer: None,
      temp_buffer: None,
      input_buffer: None,
      output_buffer: None,
      batch_count: None,
      precision: Precision::Single,
      convolution: false,
      use_lut: false,
      symmetric_kernel: false,
      input_formatted: None,
      output_formatted: None,
      inverse_return_to_input: None,
      kernel: None,
      matrix_convolution: None,
    }
  }

  pub fn dim<const N: usize>(mut self, dim: &[u32; N]) -> Self {
    let len = dim.len();
    assert!(len <= 3);

    self.fft_dim = len as u32;
    if len > 0 {
      self.size[0] = dim[0];
    }
    if len > 1 {
      self.size[1] = dim[1];
    }
    if len > 2 {
      self.size[2] = dim[2];
    }
    self
  }

  pub fn physical_device(mut self, physical_device: Arc<PhysicalDevice>) -> Self {
    self.physical_device = Some(physical_device);
    self
  }

  pub fn device(mut self, device: Arc<Device>) -> Self {
    self.device = Some(device);
    self
  }

  pub fn queue(mut self, queue: Arc<Queue>) -> Self {
    self.queue = Some(queue);
    self
  }

  pub fn command_pool(mut self, command_pool: Arc<CommandPool>) -> Self {
    self.command_pool = Some(command_pool);
    self
  }

  pub fn fence(mut self, fence: &'a Fence) -> Self {
    self.fence = Some(fence);
    self
  }

  pub fn buffer(mut self, buffer: Arc<Buffer>) -> Self {
    self.buffer = Some(buffer);
    self
  }

  pub fn temp_buffer(mut self, temp_buffer: Arc<Buffer>) -> Self {
    self.temp_buffer = Some(temp_buffer);
    self
  }

  pub fn input_buffer(mut self, input_buffer: Arc<Buffer>) -> Self {
    self.input_buffer = Some(input_buffer);
    self
  }

  pub fn output_buffer(mut self, output_buffer: Arc<Buffer>) -> Self {
    self.output_buffer = Some(output_buffer);
    self
  }

  pub fn kernel(mut self, kernel: Arc<Buffer>) -> Self {
    self.kernel = Some(kernel);
    self
  }

  pub fn normalize(mut self) -> Self {
    self.normalize = true;
    self
  }

  pub fn kernel_convolution(mut self) -> Self {
    self.kernel_convolution = true;
    self
  }

  pub fn symmetric_kernel(mut self) -> Self {
    self.symmetric_kernel = true;
    self
  }

  pub fn convolution(mut self) -> Self {
    self.convolution = true;
    self
  }

  pub fn r2c(mut self) -> Self {
    self.r2c = true;
    self
  }

  pub fn dct(mut self, dct: u64) -> Self {
    self.dct = Some(dct);
    self
  }

  pub fn dst(mut self, dst: u64) -> Self {
    self.dct = Some(dst);
    self
  }

  pub fn use_lut(mut self) -> Self {
    self.use_lut = true;
    self
  }

  pub fn coordinate_features(mut self, coordinate_features: u32) -> Self {
    self.coordinate_features = coordinate_features;
    self
  }

  pub fn matrix_convolution(mut self, matrix_convolution: u64) -> Self {
    self.matrix_convolution = Some(matrix_convolution);
    self
  }

  pub fn disable_reorder_four_step(mut self) -> Self {
    self.disable_reorder_four_step = true;
    self
  }

  pub fn zero_padding<const N: usize>(mut self, zero_padding: &[bool; N]) -> Self {
    let len = zero_padding.len();
    assert!(len <= 3);

    if len > 0 {
      self.zero_padding[0] = zero_padding[0];
    }
    if len > 1 {
      self.zero_padding[1] = zero_padding[1];
    }
    if len > 2 {
      self.zero_padding[2] = zero_padding[2];
    }
    self
  }

  pub fn zeropad_left<const N: usize>(mut self, zeropad_left: &[u32; N]) -> Self {
    let len = zeropad_left.len();
    assert!(len <= 3);

    if len > 0 {
      self.zeropad_left[0] = zeropad_left[0];
    }
    if len > 1 {
      self.zeropad_left[1] = zeropad_left[1];
    }
    if len > 2 {
      self.zeropad_left[2] = zeropad_left[2];
    }
    self
  }

  pub fn zeropad_right<const N: usize>(mut self, zeropad_right: &[u32; N]) -> Self {
    let len = zeropad_right.len();
    assert!(len <= 3);

    if len > 0 {
      self.zeropad_right[0] = zeropad_right[0];
    }
    if len > 1 {
      self.zeropad_right[1] = zeropad_right[1];
    }
    if len > 2 {
      self.zeropad_right[2] = zeropad_right[2];
    }
    self
  }

  pub fn batch_count(mut self, batch_count: u32) -> Self {
    self.batch_count = Some(batch_count);
    self
  }

  pub fn input_formatted(mut self, input_formatted: bool) -> Self {
    self.input_formatted = Some(input_formatted);
    self
  }

  pub fn inverse_return_to_input(mut self) -> Self {
    self.inverse_return_to_input = Some(true);
    self
  }
  pub fn output_formatted(mut self, output_formatted: bool) -> Self {
    self.output_formatted = Some(output_formatted);
    self
  }

  pub fn build(self) -> Result<Config<'a>, BuildError> {
    let physical_device = match self.physical_device {
      Some(v) => v,
      None => return Err(BuildError::NoPhysicalDevice),
    };

    let device = match self.device {
      Some(v) => v,
      None => return Err(BuildError::NoDevice),
    };

    let queue = match self.queue {
      Some(v) => v,
      None => return Err(BuildError::NoQueue),
    };

    let fence = match self.fence {
      Some(v) => v,
      None => return Err(BuildError::NoFence),
    };

    let command_pool = match self.command_pool {
      Some(v) => v,
      None => return Err(BuildError::NoCommandPool),
    };

    Ok(Config {
      fft_dim: self.fft_dim,
      size: self.size,
      physical_device,
      device,
      queue,
      fence,
      command_pool,
      normalize: self.normalize,
      zero_padding: self.zero_padding,
      zeropad_left: self.zeropad_left,
      zeropad_right: self.zeropad_right,
      kernel_convolution: self.kernel_convolution,
      r2c: self.r2c,
      dct: self.dct,
      dst: self.dst,
      coordinate_features: self.coordinate_features,
      disable_reorder_four_step: self.disable_reorder_four_step,
      buffer: self.buffer,
      batch_count: self.batch_count,
      precision: self.precision,
      convolution: self.convolution,
      use_lut: self.use_lut,
      symmetric_kernel: self.symmetric_kernel,
      input_formatted: self.input_formatted,
      output_formatted: self.output_formatted,
      kernel: self.kernel,
      temp_buffer: self.temp_buffer,
      input_buffer: self.input_buffer,
      inverse_return_to_input: self.inverse_return_to_input,
      output_buffer: self.output_buffer,
      matrix_convolution: self.matrix_convolution,
    })
  }
}

pub enum Precision {
  /// Perform calculations in single precision (32-bit)
  Single,
  /// Perform calculations in double precision (64-bit)
  Double,
  /// Perform calculations in half precision (16-bit)
  Half,
  /// Use half precision only as input/output buffer. Input/Output have to be allocated as half,
  /// buffer/tempBuffer have to be allocated as float (out of place mode only).
  HalfMemory,
}

pub struct Config<'a> {
  pub fft_dim: u32,
  pub size: [u32; 4usize],

  pub physical_device: Arc<PhysicalDevice>,
  pub device: Arc<Device>,
  pub queue: Arc<Queue>,
  pub fence: &'a Fence,
  pub command_pool: Arc<CommandPool>,

  pub buffer: Option<Arc<Buffer>>,
  pub input_buffer: Option<Arc<Buffer>>,
  pub output_buffer: Option<Arc<Buffer>>,
  pub temp_buffer: Option<Arc<Buffer>>,
  pub kernel: Option<Arc<Buffer>>,

  /// Normalize inverse transform
  pub normalize: bool,

  /// Don't read some data/perform computations if some input sequences are zeropadded for each axis
  pub zero_padding: [bool; 3usize],

  /// Specify start boundary of zero block in the system for each axis
  pub zeropad_left: [u32; 4usize],

  /// Specify end boundary of zero block in the system for each axis
  pub zeropad_right: [u32; 4usize],

  /// Specify if this application is used to create kernel for convolution, so it has the same properties
  pub kernel_convolution: bool,

  /// Perform convolution in this application (0 - off, 1 - on). Disables reorderFourStep parameter
  pub convolution: bool,

  /// Perform R2C/C2R decomposition
  pub r2c: bool,

  /// Perform discrete cos transform (R2R) of type 1-4
  pub dct: Option<u64>,

  /// Perform discrete sin transform (R2R) of type 1-4
  pub dst: Option<u64>,

  /// C - coordinate, or dimension of features vector. In matrix convolution - size of vector
  pub coordinate_features: u32,

  /// Disables unshuffling of four step algorithm. Requires `temp_buffer` allocation.
  pub disable_reorder_four_step: bool,

  /// Used to perform multiple batches of initial data
  pub batch_count: Option<u32>,

  pub precision: Precision,

  /// Switches from calculating sincos to using precomputed LUT tables
  pub use_lut: bool,

  /// Specify if kernel in 2x2 or 3x3 matrix convolution is symmetric
  pub symmetric_kernel: bool,

  /// specify if input buffer is padded - false is padded, true is not padded.
  /// For example if it is not padded for R2C if out-of-place mode is selected
  /// (only if numberBatches==1 and numberKernels==1)
  pub input_formatted: Option<bool>,

  /// put the inverse transformed data into the input buffer, if input_formatted
  /// is set to true
  pub inverse_return_to_input: Option<bool>,

  /// specify if output buffer is padded - false is padded, true is not padded.
  /// For example if it is not padded for R2C if out-of-place mode is selected
  /// (only if numberBatches==1 and numberKernels==1)
  pub output_formatted: Option<bool>,

  /// If performing matrix convolution, leading dimension of the matrix, e.g. if
  /// convolving with a 3x3 matrix, matrix_convolution is 3, and coordinate_features
  /// should also be 3
  pub matrix_convolution: Option<u64>,
}

#[derive(Display, Debug, Error)]
pub enum ConfigError {
  InvalidConfig,
}

#[allow(dead_code)]
pub(crate) struct KeepAlive {
  pub device: Arc<Device>,
  pub queue: Arc<Queue>,
  pub command_pool: Arc<CommandPool>,
  pub buffer: Option<Arc<Buffer>>,
  pub input_buffer: Option<Arc<Buffer>>,
  pub output_buffer: Option<Arc<Buffer>>,
  pub temp_buffer: Option<Arc<Buffer>>,
  pub kernel: Option<Arc<Buffer>>,
}

#[repr(C)]
pub(crate) struct ConfigGuard {
  pub(crate) keep_alive: KeepAlive,
  pub(crate) config: vkfft_sys::VkFFTConfiguration,
  pub(crate) physical_device: ash::vk::PhysicalDevice,
  pub(crate) device: ash::vk::Device,
  pub(crate) queue: ash::vk::Queue,
  pub(crate) command_pool: ash::vk::CommandPool,
  pub(crate) fence: ash::vk::Fence,
  pub(crate) buffer_size: u64,
  pub(crate) buffer: Option<ash::vk::Buffer>,
  pub(crate) input_buffer_size: u64,
  pub(crate) input_buffer: Option<ash::vk::Buffer>,
  pub(crate) output_buffer_size: u64,
  pub(crate) output_buffer: Option<ash::vk::Buffer>,
  pub(crate) temp_buffer_size: u64,
  pub(crate) temp_buffer: Option<ash::vk::Buffer>,
  pub(crate) kernel_size: u64,
  pub(crate) kernel: Option<ash::vk::Buffer>,
}

impl<'a> Config<'a> {
  pub fn builder() -> ConfigBuilder<'a> {
    ConfigBuilder::new()
  }

  pub fn buffer_size(&self) -> usize {
    self.buffer.as_ref().map(|b| b.size() as usize).unwrap_or(0)
  }

  pub fn buffer(&self) -> Option<&Arc<Buffer>> {
    self.buffer.as_ref()
  }

  pub fn temp_buffer(&self) -> Option<&Arc<Buffer>> {
    self.temp_buffer.as_ref()
  }

  pub fn input_buffer(&self) -> Option<&Arc<Buffer>> {
    self.input_buffer.as_ref()
  }

  pub fn output_buffer(&self) -> Option<&Arc<Buffer>> {
    self.output_buffer.as_ref()
  }

  pub fn kernel_convolution(&self) -> bool {
    self.kernel_convolution
  }

  pub fn symmetric_kernel(&self) -> bool {
    self.symmetric_kernel
  }

  pub fn convolution(&self) -> bool {
    self.convolution
  }

  pub fn r2c(&self) -> bool {
    self.r2c
  }

  pub fn normalize(&self) -> bool {
    self.normalize
  }

  pub fn coordinate_features(&self) -> u32 {
    self.coordinate_features
  }

  pub fn batch_count(&self) -> Option<u32> {
    self.batch_count
  }

  pub fn use_lut(&self) -> bool {
    self.use_lut
  }

  pub(crate) fn as_sys(&self) -> Result<Pin<Box<ConfigGuard>>, ConfigError> {
    use std::mem::{transmute, zeroed};

    unsafe {
      let keep_alive = KeepAlive {
        device: self.device.clone(),
        buffer: self.buffer.as_ref().map(|b| b.clone()),
        input_buffer: self.input_buffer.as_ref().map(|b| b.clone()),
        output_buffer: self.output_buffer.as_ref().map(|b| b.clone()),
        kernel: self.kernel.as_ref().map(|b| b.clone()),
        command_pool: self.command_pool.clone(),
        queue: self.queue.clone(),
        temp_buffer: self.temp_buffer.as_ref().map(|b| b.clone()),
      };

      let mut res = Box::pin(ConfigGuard {
        keep_alive,
        config: zeroed(),
        physical_device: self.physical_device.handle(),
        device: self.device.handle(),
        queue: self.queue.handle(),
        command_pool: self.command_pool.handle(),
        fence: self.fence.handle(),
        buffer_size: self.buffer.as_ref().map(|b| b.size()).unwrap_or(0),
        temp_buffer_size: self.temp_buffer.as_ref().map(|b| b.size()).unwrap_or(0),
        input_buffer_size: self.input_buffer.as_ref().map(|b| b.size()).unwrap_or(0),
        output_buffer_size: self.output_buffer.as_ref().map(|b| b.size()).unwrap_or(0),
        kernel_size: self.kernel.as_ref().map(|b| b.size()).unwrap_or(0),
        buffer: self.buffer.as_ref().map(|b| b.handle()),
        temp_buffer: self.temp_buffer.as_ref().map(|b| b.handle()),
        input_buffer: self.input_buffer.as_ref().map(|b| b.handle()),
        output_buffer: self.output_buffer.as_ref().map(|b| b.handle()),
        kernel: self.kernel.as_ref().map(|b| b.handle()),
      });

      res.config.FFTdim = self.fft_dim as u64;
      res.config.size = self.size.map(u64::from);

      res.config.physicalDevice = transmute::<*mut ash::vk::PhysicalDevice, *mut *mut vkfft_sys::VkPhysicalDevice_T>(addr_of_mut!(res.physical_device));
      res.config.device = transmute::<*mut ash::vk::Device, *mut *mut vkfft_sys::VkDevice_T>(addr_of_mut!(res.device));
      res.config.queue = transmute::<*mut ash::vk::Queue, *mut *mut vkfft_sys::VkQueue_T>(addr_of_mut!(res.queue));
      res.config.commandPool = transmute::<*mut ash::vk::CommandPool, *mut *mut vkfft_sys::VkCommandPool_T>(addr_of_mut!(res.command_pool));
      res.config.fence = transmute::<*mut ash::vk::Fence, *mut *mut vkfft_sys::VkFence_T>(addr_of_mut!(res.fence));
      res.config.normalize = self.normalize.into();

      if res.kernel_size != 0 {
        res.config.kernelSize = addr_of_mut!(res.kernel_size);
      }

      if let Some(t) = &res.kernel {
        res.config.kernel = t as *const ash::vk::Buffer as *mut *mut vkfft_sys::VkBuffer_T;
      }

      if res.buffer_size != 0 {
        res.config.bufferSize = addr_of_mut!(res.buffer_size);
      }

      if let Some(t) = &res.buffer {
        res.config.buffer = t as *const ash::vk::Buffer as *mut *mut vkfft_sys::VkBuffer_T;
      }

      if res.temp_buffer_size != 0 {
        res.config.userTempBuffer = 1;
        res.config.tempBufferSize = addr_of_mut!(res.temp_buffer_size);
      }

      if let Some(t) = &res.temp_buffer {
        res.config.tempBuffer = t as *const ash::vk::Buffer as *mut *mut vkfft_sys::VkBuffer_T;
      }

      if res.input_buffer_size != 0 {
        res.config.inputBufferSize = addr_of_mut!(res.input_buffer_size);
      }

      if let Some(t) = &res.input_buffer {
        res.config.inputBuffer = t as *const ash::vk::Buffer as *mut *mut vkfft_sys::VkBuffer_T;
      }

      if res.output_buffer_size != 0 {
        res.config.outputBufferSize = addr_of_mut!(res.output_buffer_size);
      }

      if let Some(t) = &res.output_buffer {
        res.config.outputBuffer = t as *const ash::vk::Buffer as *mut *mut vkfft_sys::VkBuffer_T;
      }

      res.config.performZeropadding[0] = self.zero_padding[0].into();
      res.config.performZeropadding[1] = self.zero_padding[1].into();
      res.config.performZeropadding[2] = self.zero_padding[2].into();

      res.config.fft_zeropad_left = self.zeropad_left.map(u64::from);
      res.config.fft_zeropad_right = self.zeropad_right.map(u64::from);
      res.config.performConvolution = self.convolution.into();
      if self.convolution {
        res.config.numberKernels = 1;
      }
      res.config.kernelConvolution = self.kernel_convolution as u64;
      res.config.performR2C = self.r2c.into();
      res.config.performDCT = self.dct.unwrap_or(0);
      res.config.performDST = self.dst.unwrap_or(0);
      res.config.coordinateFeatures = self.coordinate_features as u64;
      res.config.disableReorderFourStep = self.disable_reorder_four_step.into();

      res.config.symmetricKernel = self.symmetric_kernel.into();

      if let Some(input_formatted) = self.input_formatted {
        res.config.isInputFormatted = input_formatted.into();
      }

      if let Some(inverse_return_to_input) = self.inverse_return_to_input {
        res.config.inverseReturnToInputBuffer = inverse_return_to_input.into();
      }

      if let Some(output_formatted) = self.output_formatted {
        res.config.isOutputFormatted = output_formatted.into();
      }

      match self.precision {
        Precision::Double => {
          res.config.doublePrecision = true.into();
        }
        Precision::Half => res.config.halfPrecision = true.into(),
        Precision::HalfMemory => {
          res.config.halfPrecisionMemoryOnly = true.into();

          if let Some(false) = self.input_formatted {
            return Err(ConfigError::InvalidConfig);
          }

          if let Some(false) = self.output_formatted {
            return Err(ConfigError::InvalidConfig);
          }

          res.config.isInputFormatted = true.into();
          res.config.isOutputFormatted = true.into();
        }
        _ => {}
      }

      if let Some(batch_count) = self.batch_count {
        res.config.numberBatches = batch_count as u64;
      }

      if let Some(matrix_convolution) = self.matrix_convolution {
        res.config.matrixConvolution = matrix_convolution;
      }

      Ok(res)
    }
  }
}
