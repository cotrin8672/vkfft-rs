use crate::{
  app::{App, LaunchParams},
  config::ConfigBuilder,
};
use std::{error::Error, sync::Arc};
use vulkano::command_buffer::{
  pool::{UnsafeCommandPool, UnsafeCommandPoolAlloc},
  sys::UnsafeCommandBufferBuilder,
  CommandBufferUsage,
};
use vulkano::command_buffer::{submit::SubmitCommandBufferBuilder, sys::UnsafeCommandBuffer};
use vulkano::device::{Device, DeviceExtensions, Features, Queue};
use vulkano::instance::debug::{DebugCallback, Message, MessageSeverity, MessageType};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::sync::Fence;

const MESSAGE_SEVERITIES: MessageSeverity = MessageSeverity {
  error: true,
  warning: true,
  information: true,
  verbose: true,
};
pub enum FftType {
  Forward,
  Inverse,
}

fn on_debug_message(msg: &Message) {
  if msg.ty.general && msg.severity.verbose {
    return;
  }

  let severity = if msg.severity.error {
    "error"
  } else if msg.severity.warning {
    "warning"
  } else if msg.severity.information {
    "information"
  } else if msg.severity.verbose {
    "verbose"
  } else {
    panic!("no-impl");
  };

  let ty = if msg.ty.general {
    "general"
  } else if msg.ty.validation {
    "validation"
  } else if msg.ty.performance {
    "performance"
  } else {
    panic!("no-impl");
  };

  eprintln!(
    "{} {} {}: {}",
    msg.layer_prefix.unwrap_or("unknown"),
    ty,
    severity,
    msg.description
  );
}

pub struct Context<'a> {
  pub instance: &'a Arc<Instance>,
  pub physical: PhysicalDevice<'a>,
  pub device: Arc<Device>,
  pub queue: Arc<Queue>,
  pub pool: Arc<UnsafeCommandPool>,
  pub fence: Fence,
  _debug_cb: Option<DebugCallback>,
}

impl<'a> Context<'a> {
  pub fn new(instance: &'a Arc<Instance>) -> Result<Self, Box<dyn std::error::Error>> {
    let debug_cb = DebugCallback::new(
      &instance,
      MESSAGE_SEVERITIES,
      MessageType::all(),
      on_debug_message,
    )
    .ok();

    let physical = PhysicalDevice::enumerate(&instance)
      .next()
      .ok_or("No device available")?;

    let queue_family = physical
      .queue_families()
      .find(|&q| q.supports_compute() && q.supports_graphics())
      .ok_or("Couldn't find a compute queue family")?;

    let (device, mut queues) = Device::new(
      physical,
      &Features::none(),
      &DeviceExtensions::none(),
      [(queue_family, 0.5)].iter().cloned(),
    )?;

    let queue = queues.next().unwrap();
    let pool = Arc::new(UnsafeCommandPool::new(
      device.clone(),
      queue_family,
      false,
      true,
    )?);

    let fence = Fence::alloc(device.clone())?;

    Ok(Self {
      instance,
      physical,
      queue,
      device,
      pool,
      fence,
      _debug_cb: debug_cb,
    })
  }

  pub fn submit(
    &mut self,
    command_buffer: UnsafeCommandBuffer,
  ) -> Result<(), Box<dyn std::error::Error>> {
    unsafe {
      let mut submit = SubmitCommandBufferBuilder::new();
      submit.add_command_buffer(&command_buffer);
      submit.set_fence_signal(&self.fence);

      submit.submit(&self.queue)?;

      self.fence.wait(None)?;

      self.fence.reset()?;
    }

    Ok(())
  }

  pub fn single_fft(
    &mut self,
    config_builder: ConfigBuilder,
    fft_type: FftType,
  ) -> Result<(), Box<dyn std::error::Error>> {
    // Allocate a command buffer
    let primary_cmd_buffer = self.alloc_primary_cmd_buffer()?;

    // Create command buffer handle
    let builder = unsafe {
      UnsafeCommandBufferBuilder::new(&primary_cmd_buffer, vulkano::command_buffer::CommandBufferLevel::primary(), CommandBufferUsage::MultipleSubmit)?
    };

    // Configure FFT launch parameters
    let mut params = LaunchParams::builder().command_buffer(&builder).build()?;

    // Add parameters from the present context to the config
    let config = config_builder
      .physical_device(self.physical)
      .device(self.device.clone())
      .fence(&self.fence)
      .queue(self.queue.clone())
      .command_pool(self.pool.clone())
      .build()?;

    // Construct FFT "Application"
    let mut app = App::new(config)?;

    // Run forward FFT
    match fft_type {
      FftType::Forward => app.forward(&mut params)?,
      FftType::Inverse => app.inverse(&mut params)?,
    }

    // Dispatch command buffer and wait for completion
    let command_buffer = builder.build()?;
    self.submit(command_buffer)?;
    Ok(())
  }
  pub fn alloc_cmd_buffer(
    &self,
    secondary: bool,
  ) -> Result<UnsafeCommandPoolAlloc, Box<dyn Error>> {
    Ok(
      self
        .pool
        .alloc_command_buffers(secondary, 1)?
        .next()
        .ok_or("Failed to allocate cmd buffer")?,
    )
  }

  pub fn alloc_primary_cmd_buffer(&self) -> Result<UnsafeCommandPoolAlloc, Box<dyn Error>> {
    self.alloc_cmd_buffer(false)
  }

  pub fn alloc_secondary_cmd_buffer(&self) -> Result<UnsafeCommandPoolAlloc, Box<dyn Error>> {
    self.alloc_cmd_buffer(true)
  }
}
