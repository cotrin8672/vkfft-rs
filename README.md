# vkfft-rs

`vkfft-rs` allows high-performance execution of 1, 2, or 3D FFTs on the GPU using Vulkan in Rust, with built-in support for convolutions.

`vkfft-rs` is a binding for [VkFFT](https://github.com/DTolm/VkFFT) that assumes usage with [vulkano](https://vulkano.rs/). While VkFFT, despite the name, supports multiple backends, this wrapper requires usage with Vulkan.

While `vkfft-rs` attempts to maintain a safe API, it's very likely there are some safe functions in this codebase that can still cause unsafe behavior. VkFFT's API and associated data structures are unsafe and stateful, which presents difficulties in ensuring Rust's safety guarantees. Until its safety properties can be properly verified it is recommend to proceed with caution. PRs welcome!

## Usage

To see an example, which exhibits the data layout of the buffers for a 1D complex-to-complex transform, and a 2D real-to-complex one, run
```.sh
cargo run --example simple
```

The required libraries should now be built automatically by cargo, except for Vulkan itself. 

 - On Linux: You should install libvulkan using your system's package manager (on most systems, it will already be present).
 - On Mac and Windows: Install the [Vulkan SDK from LunarG](https://vulkan.lunarg.com). 
 