use crate::{
  app::{App, LaunchParams},
  config::ConfigBuilder,
};
use ash::vk::Result as ash_Result;
use std::{pin::Pin, sync::Arc};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::device::{physical::PhysicalDevice, Device, Queue};
use vulkano::instance::Instance;
use vulkano::sync::fence::Fence;
use vulkano::{
  buffer::{AllocateBufferError, Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
  memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
  Validated,
};
use vulkano::{
  command_buffer::{
    allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
    pool::{CommandPool, CommandPoolCreateFlags, CommandPoolCreateInfo},
    CommandBufferUsage,
  },
  device::{DeviceCreateInfo, QueueCreateInfo, QueueFlags},
  sync::fence::FenceCreateInfo,
  VulkanObject,
};

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
  pub allocator: Arc<dyn MemoryAllocator>,
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
          queue_family_index,
          ..Default::default()
        }],
        ..Default::default()
      },
    )?;
    let queue = queues.next().unwrap();
    let pool = Arc::new(CommandPool::new(
      device.clone(),
      CommandPoolCreateInfo {
        queue_family_index,
        flags: CommandPoolCreateFlags::default(),
        ..Default::default()
      },
    )?);
    let fence = Fence::new(device.clone(), FenceCreateInfo::default())?;
    let allocator =
      Arc::new(vulkano::memory::allocator::StandardMemoryAllocator::new_default(device.clone()));
    Ok(Self {
      instance,
      physical: physical.clone(),
      queue,
      device,
      pool,
      fence,
      allocator,
    })
  }
  pub fn new_buffer_from_iter<T, I>(
    &self,
    iter: I,
  ) -> Result<Subbuffer<[T]>, Validated<AllocateBufferError>>
  where
    T: BufferContents,
    I: IntoIterator<Item = T>,
    I::IntoIter: ExactSizeIterator,
  {
    Buffer::from_iter(
      self.allocator.clone(),
      BufferCreateInfo {
        usage: BufferUsage::TRANSFER_DST,
        ..Default::default()
      },
      AllocationCreateInfo {
        memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
        ..Default::default()
      },
      iter,
    )
  }

  pub fn submit(
    &self,
    command_buffer: Arc<PrimaryAutoCommandBuffer>,
  ) -> Result<(), Box<dyn std::error::Error>> {
    let fns = self.device.fns();
    let command_buffer_submit_info = ash::vk::CommandBufferSubmitInfo {
      command_buffer: command_buffer.handle(),
      device_mask: 0u32,
      ..Default::default()
    };
    if self.device.enabled_features().synchronization2 {
      let submit_info_vk = ash::vk::SubmitInfo2 {
        command_buffer_info_count: 1u32,
        p_command_buffer_infos: &command_buffer_submit_info,
        ..Default::default()
      };
      if self.device.api_version() >= vulkano::Version::V1_3 {
        self.queue.with(|_| unsafe {
          let submit_result = unsafe {
            (fns.v1_3.queue_submit2)(
              self.queue.handle(),
              1u32,
              &submit_info_vk,
              self.fence.handle(),
            )
          };
          if submit_result != ash_Result::SUCCESS {
            println!(
              "Submission to Vulkan queue failed with result {:?}",
              submit_result
            );
            panic!("Vulkan in non-handled state, panicking.");
          }
          self.fence.wait(None).unwrap();
          self.fence.reset().unwrap();
        });
      } else {
        self.queue.with(|_| unsafe {
          let submit_result = unsafe {
            (fns.khr_synchronization2.queue_submit2_khr)(
              self.queue.handle(),
              1u32,
              &submit_info_vk,
              self.fence.handle(),
            )
          };
          if submit_result != ash_Result::SUCCESS {
            println!(
              "Submission to Vulkan queue failed with result {:?}",
              submit_result
            );
            panic!("Vulkan in non-handled state, panicking.");
          }
          self.fence.wait(None).unwrap();
          self.fence.reset().unwrap();
        });
      }
    } else {
      let submit_info_vk = ash::vk::SubmitInfo {
        command_buffer_count: 1u32,
        p_command_buffers: &command_buffer_submit_info.command_buffer,
        ..Default::default()
      };
      self.queue.with(|_| unsafe {
        let submit_result = unsafe {
          (fns.v1_0.queue_submit)(
            self.queue.handle(),
            1u32,
            &submit_info_vk,
            self.fence.handle(),
          )
        };
        if submit_result != ash_Result::SUCCESS {
          println!(
            "Submission to Vulkan queue failed with result {:?}",
            submit_result
          );
          panic!("Vulkan in non-handled state, panicking.");
        }
        self.fence.wait(None).unwrap();
        self.fence.reset().unwrap();
      });
    }
    Ok(())
  }
  pub fn start_fft_chain(
    &self,
    config_builder: ConfigBuilder,
    fft_type: FftType,
  ) -> Result<(Pin<Box<App>>, LaunchParams, AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>), Box<dyn std::error::Error>>
  {
    let command_buffer_allocator = Arc::new(
      StandardCommandBufferAllocator::new(
        self.device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
      )
    );
    let builder = unsafe {
      AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        self.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
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
    Ok((app, params, builder))
  }
  pub fn chain_fft_with_app(
    &self,
    mut app: Pin<Box<App>>,
    mut params: LaunchParams,
    fft_type: FftType,
  ) -> Result<(Pin<Box<App>>, LaunchParams), Box<dyn std::error::Error>> {
    match fft_type {
      FftType::Forward => app.forward(&mut params)?,
      FftType::Inverse => app.inverse(&mut params)?,
    }
    Ok((app, params))
  }
  pub fn chain_fft_with_config(
    &self,
    config_builder: ConfigBuilder,
    builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    fft_type: FftType,
  ) -> Result<(Pin<Box<App>>, LaunchParams, AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>), Box<dyn std::error::Error>>
  {
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
    Ok((app, params, builder))
  }
  pub fn single_fft(
    &self,
    config_builder: ConfigBuilder,
    fft_type: FftType,
  ) -> Result<(), Box<dyn std::error::Error>> {
    let (_app, _params, builder) = self.start_fft_chain(config_builder, fft_type)?;
    self.submit(builder.build()?)?;
    Ok(())
  }
}
