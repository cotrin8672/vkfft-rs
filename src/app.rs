use std::sync::Arc;

use error::check_error;
use vulkano::{buffer::Buffer, Handle, VulkanObject};

use crate::{
  config::{Config, ConfigGuard},
  error,
};

use ash::vk;
use std::pin::Pin;

use std::ptr::addr_of_mut;

use derive_more::{Display, Error};

#[derive(Display, Debug, Error)]
pub enum BuildError {
  NoCommandBuffer,
  // NoBuffer,
  // NoTempBuffer,
  // NoInputBuffer,
  // NoOutputBuffer,
  // NoKernel,
}

#[derive(Display, Debug, Error)]
pub enum LaunchError {
  ConfigSpecifiesBuffer,
  ConfigSpecifiesTempBuffer,
  ConfigSpecifiesInputBuffer,
  ConfigSpecifiesOutputBuffer,
  ConfigSpecifiesKernel,
}

pub struct LaunchParamsBuilder {
  command_buffer: Option<vk::CommandBuffer>,
  buffer: Option<Arc<Buffer>>,
  temp_buffer: Option<Arc<Buffer>>,
  input_buffer: Option<Arc<Buffer>>,
  output_buffer: Option<Arc<Buffer>>,
  kernel: Option<Arc<Buffer>>,
}

impl Default for LaunchParamsBuilder {
  fn default() -> Self {
    Self::new()
  }
}

impl LaunchParamsBuilder {
  pub fn new() -> Self {
    Self {
      buffer: None,
      command_buffer: None,
      input_buffer: None,
      kernel: None,
      output_buffer: None,
      temp_buffer: None,
    }
  }

  pub fn command_buffer<C>(mut self, command_buffer: &C) -> Self
  where
    C: VulkanObject<Handle = vk::CommandBuffer>,
  {
    self.command_buffer = Some(command_buffer.handle());
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

  pub fn build(self) -> Result<LaunchParams, BuildError> {
    let command_buffer = match self.command_buffer {
      Some(command_buffer) => command_buffer,
      None => return Err(BuildError::NoCommandBuffer),
    };

    Ok(LaunchParams {
      buffer: self.buffer,
      command_buffer,
      input_buffer: self.input_buffer,
      output_buffer: self.output_buffer,
      temp_buffer: self.temp_buffer,
      kernel: self.kernel,
    })
  }
}

#[repr(C)]
pub(crate) struct LaunchParamsGuard {
  pub(crate) params: vkfft_sys::VkFFTLaunchParams,
  pub(crate) command_buffer: ash::vk::CommandBuffer,
  pub(crate) buffer: Option<u64>,
  pub(crate) temp_buffer: Option<u64>,
  pub(crate) input_buffer: Option<u64>,
  pub(crate) output_buffer: Option<u64>,
  pub(crate) kernel: Option<u64>,
}

#[derive(Clone)]
pub struct LaunchParams {
  pub command_buffer: vk::CommandBuffer,
  pub buffer: Option<Arc<Buffer>>,
  pub temp_buffer: Option<Arc<Buffer>>,
  pub input_buffer: Option<Arc<Buffer>>,
  pub output_buffer: Option<Arc<Buffer>>,
  pub kernel: Option<Arc<Buffer>>,
}

impl LaunchParams {
  fn buffer_object<B>(buffer: B) -> u64
  where
    B: AsRef<Buffer>,
  {
    buffer.as_ref().handle().as_raw()
  }
  // pub fn duplicate(&self) -> Self{
  //   LaunchParams{
  //     command_buffer: self.command_buffer.clone(),

  //   }
  // }
  pub(crate) fn as_sys(&self) -> Pin<Box<LaunchParamsGuard>> {
    use std::mem::{transmute, zeroed};

    unsafe {
      let mut res = Box::pin(LaunchParamsGuard {
        params: zeroed(),
        command_buffer: self.command_buffer,
        buffer: self.buffer.as_ref().map(Self::buffer_object),
        temp_buffer: self.temp_buffer.as_ref().map(Self::buffer_object),
        input_buffer: self.input_buffer.as_ref().map(Self::buffer_object),
        output_buffer: self.output_buffer.as_ref().map(Self::buffer_object),
        kernel: self.kernel.as_ref().map(Self::buffer_object),
      });

      res.params.commandBuffer = transmute::<*mut ash::vk::CommandBuffer, *mut *mut vkfft_sys::VkCommandBuffer_T>(addr_of_mut!(res.command_buffer));

      if let Some(b) = &res.buffer {
        res.params.buffer = b as *const u64 as *mut *mut vkfft_sys::VkBuffer_T;
      }

      if let Some(b) = &res.temp_buffer {
        res.params.tempBuffer = b as *const u64 as *mut *mut vkfft_sys::VkBuffer_T;
      }

      if let Some(b) = &res.input_buffer {
        res.params.inputBuffer = b as *const u64 as *mut *mut vkfft_sys::VkBuffer_T;
      }

      if let Some(b) = &res.output_buffer {
        res.params.outputBuffer = b as *const u64 as *mut *mut vkfft_sys::VkBuffer_T;
      }

      if let Some(k) = &res.kernel {
        res.params.kernel = k as *const u64 as *mut *mut vkfft_sys::VkBuffer_T;
      }

      res
    }
  }

  pub fn builder() -> LaunchParamsBuilder {
    LaunchParamsBuilder::new()
  }
}

pub struct App {
  app: vkfft_sys::VkFFTApplication,

  // Safety: We must keep a copy of the config to ensure our resources are kept alive
  config: Pin<Box<ConfigGuard>>,
}

impl App {
  pub fn new(config: Config) -> error::Result<Pin<Box<Self>>> {
    use vkfft_sys::*;

    let app: VkFFTApplication = unsafe { std::mem::zeroed() };

    let sys_config = config.as_sys()?;

    let mut res = Box::pin(Self {
      app,
      config: sys_config,
    });

    check_error(unsafe { initializeVkFFT(std::ptr::addr_of_mut!(res.app), res.config.config) })?;

    Ok(res)
  }

  pub fn launch(&mut self, params: &mut LaunchParams, inverse: bool) -> error::Result<()> {
    use vkfft_sys::VkFFTAppend;

    let mut params = params.as_sys();

    if self.config.buffer.is_some() && params.buffer.is_some() {
      return Err(LaunchError::ConfigSpecifiesBuffer.into());
    }

    if self.config.temp_buffer.is_some() && params.temp_buffer.is_some() {
      return Err(LaunchError::ConfigSpecifiesTempBuffer.into());
    }

    if self.config.input_buffer.is_some() && params.input_buffer.is_some() {
      return Err(LaunchError::ConfigSpecifiesInputBuffer.into());
    }

    if self.config.output_buffer.is_some() && params.output_buffer.is_some() {
      return Err(LaunchError::ConfigSpecifiesOutputBuffer.into());
    }

    check_error(unsafe {
      VkFFTAppend(
        std::ptr::addr_of_mut!(self.app),
        if inverse { 1 } else { -1 },
        std::ptr::addr_of_mut!(params.params),
      )
    })?;

    Ok(())
  }

  pub fn forward(&mut self, params: &mut LaunchParams) -> error::Result<()> {
    self.launch(params, false)
  }

  pub fn inverse(&mut self, params: &mut LaunchParams) -> error::Result<()> {
    self.launch(params, true)
  }
}

impl Drop for App {
  fn drop(&mut self) {
    use vkfft_sys::*;

    unsafe {
      deleteVkFFT(std::ptr::addr_of_mut!(self.app));
    }
  }
}
