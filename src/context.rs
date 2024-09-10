use crate::{
  app::{App, LaunchParams},
  config::ConfigBuilder,
};

use std::sync::Arc;
use vulkano::command_buffer::sys::UnsafeCommandBuffer;
use vulkano::device::{physical::PhysicalDevice, Device, Queue};
use vulkano::{
  command_buffer::{
    allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
    pool::{CommandPool, CommandPoolCreateFlags, CommandPoolCreateInfo},
    sys::{CommandBufferBeginInfo, UnsafeCommandBufferBuilder},
    CommandBufferUsage,
  },
  device::{DeviceCreateInfo, QueueCreateInfo, QueueFlags},
  sync::fence::FenceCreateInfo,
  VulkanObject,
};

use vulkano::instance::Instance;
use vulkano::sync::fence::Fence;
use ash::vk::Result as ash_Result;
pub enum FftType {
  Forward,
  Inverse,
}

pub struct Context<'a> {
  pub instance: &'a Arc<Instance>,
  pub physical: Arc<PhysicalDevice>,
  pub device: Arc<Device>,
  pub queue: Arc<Queue>,
  pub pool: Arc<CommandPool>,
  pub fence: Fence,
}

impl<'a> Context<'a> {
  pub fn new(instance: &'a Arc<Instance>) -> Result<Self, Box<dyn std::error::Error>> {
    let physical = instance
      .enumerate_physical_devices()?
      .next()
      .ok_or("No device available")?;

    let queue_family_index = physical
      .queue_family_properties()
      .iter()
      .enumerate()
      .position(|(_queue_family_index, queue_family_properties)| {
        queue_family_properties
          .queue_flags
          .contains(QueueFlags::COMPUTE)
          && queue_family_properties
            .queue_flags
            .contains(QueueFlags::GRAPHICS)
      })
      .expect("couldn't find a compute+graphical queue family") as u32;
    let (device, mut queues) = Device::new(
      physical.clone(),
      DeviceCreateInfo {
        queue_create_infos: vec![QueueCreateInfo {
          queue_family_index: queue_family_index,
          ..Default::default()
        }],
        ..Default::default()
      },
    )?;
    let queue = queues.next().unwrap();
    let pool = Arc::new(CommandPool::new(
      device.clone(),
      CommandPoolCreateInfo {
        queue_family_index: queue_family_index,
        flags: CommandPoolCreateFlags::default(),
        ..Default::default()
      },
    )?);
    let fence = Fence::new(device.clone(), FenceCreateInfo::default())?;

    Ok(Self {
      instance,
      physical: physical.clone(),
      queue,
      device,
      pool,
      fence,
    })
  }

  pub fn submit(
    &mut self,
    command_buffer: UnsafeCommandBuffer,
  ) -> Result<(), Box<dyn std::error::Error>> {
    let fns = self.device.fns();
    let command_buffer_submit_info = ash::vk::CommandBufferSubmitInfo {
      command_buffer: command_buffer.handle(),
      device_mask: 0u32,
      ..Default::default()
    };
    let submit_info_vk = ash::vk::SubmitInfo2 {
      command_buffer_info_count: 1u32,
      p_command_buffer_infos: &command_buffer_submit_info,
      ..Default::default()
    };
    let submit_result = unsafe {
       (fns.v1_3.queue_submit2)(
        self.queue.handle(),
        1u32,
        &submit_info_vk,
        self.fence.handle(),
      )
    };
    if submit_result != ash_Result::SUCCESS {
      println!("Submission to Vulkan queue failed with result {:?}", submit_result);
      panic!("Vulkan in non-handled state, panicking.");
    }
    self.fence.wait(None)?;
    self.fence.reset()?;
    
    Ok(())
  }

  pub fn single_fft(
    &mut self,
    config_builder: ConfigBuilder,
    fft_type: FftType,
  ) -> Result<(), Box<dyn std::error::Error>> {
    let command_buffer_allocator = StandardCommandBufferAllocator::new(
      self.device.clone(),
      StandardCommandBufferAllocatorCreateInfo::default(),
    );
    let builder = unsafe {
      UnsafeCommandBufferBuilder::new(
        &command_buffer_allocator,
        self.queue.queue_family_index(),
        vulkano::command_buffer::CommandBufferLevel::Primary,
        CommandBufferBeginInfo {
          usage: CommandBufferUsage::OneTimeSubmit,
          ..Default::default()
        },
      )?
    };

    let mut params = LaunchParams::builder().command_buffer(&builder).build()?;
    let config = config_builder
      .physical_device(self.physical.clone())
      .device(self.device.clone())
      .fence(&self.fence)
      .queue(self.queue.clone())
      .command_pool(self.pool.clone())
      .build()?;
    let mut app = App::new(config)?;
    match fft_type {
      FftType::Forward => app.forward(&mut params)?,
      FftType::Inverse => app.inverse(&mut params)?,
    }
    let command_buffer = builder.build()?;
    self.submit(command_buffer)?;
    Ok(())
  }
}
