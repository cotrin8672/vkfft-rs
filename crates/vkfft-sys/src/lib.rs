#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

#[cfg(target_os = "windows")]
include!("bindings_win.rs");

#[cfg(target_os = "linux")]
include!("bindings_linux.rs");

#[cfg(target_os = "macos")]
include!("bindings_macos.rs");
