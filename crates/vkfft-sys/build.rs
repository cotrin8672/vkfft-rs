extern crate bindgen;
extern crate cc;

use bindgen::Bindings;
use regex::Regex;
use std::collections::HashMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::env;
use glob::glob;

const BINDGEN_FILENAME: &str = "src/bindings.rs";

//from https://github.com/SnowflakePowered/glslang-rs/blob/master/glslang-sys/build.rs
pub fn add_subdirectory(build: &mut cc::Build, directory: &str) {
  for entry in
      glob(&*format!("glslang/{directory}/**/*.cpp")).expect("failed to read glob")
  {
      if let Ok(path) = entry {
          build.file(path);
      }
  }

  for entry in glob(&*format!("glslang/{directory}/**/*.c")).expect("failed to read glob")
  {
      if let Ok(path) = entry {
          build.file(path);
      }
  }
}

fn build_glslang(){
  let mut glslang_build = cc::Build::new();
  glslang_build
      .cpp(true)
      .std("c++17")
      .define("ENABLE_HLSL", "ON")
      .define("ENABLE_OPT", "OFF")
      .define("ENABLE_GLSLANG_BINARIES", "OFF")
      .define("BUILD_EXTERNAL", "OFF")
      .includes(&["glslang", "glslang_build_info"]);

  add_subdirectory(&mut glslang_build, "glslang/CInterface");
  add_subdirectory(&mut glslang_build, "glslang/GenericCodeGen");
  add_subdirectory(&mut glslang_build, "glslang/HLSL");
  add_subdirectory(&mut glslang_build, "glslang/MachineIndependent");
  add_subdirectory(&mut glslang_build, "SPIRV");

  glslang_build.compile("glslang");
  println!("cargo:rustc-link-lib=static=glslang");
}

fn gen_wrapper<F, const N: usize>(
  file: F,
  defines: &[(&str, &str); N],
  include_dirs: &Vec<String>,
) -> Result<Bindings, Box<dyn Error>>
where
  F: AsRef<Path>,
{
  let base_args = ["".to_string()];

  let defines: Vec<String> = defines
    .iter()
    .map(|(k, v)| format!("-D{}={}", k, v))
    .collect();

  let include_dirs: Vec<String> = include_dirs.iter().map(|s| format!("-I{}", s)).collect();

  let clang_args = base_args
    .iter()
    .chain(defines.iter())
    .chain(include_dirs.iter());

  println!("{:?}", clang_args);

  let res = bindgen::Builder::default()
    .clang_args(clang_args)
    .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
    .header(file.as_ref().to_str().unwrap())
    .allowlist_recursively(true)
    .allowlist_type("VkFFTConfiguration")
    .allowlist_type("VkFFTLaunchParams")
    .allowlist_type("VkFFTResult")
    .allowlist_type("VkFFTSpecializationConstantsLayout")
    .allowlist_type("VkFFTPushConstantsLayout")
    .allowlist_type("VkFFTAxis")
    .allowlist_type("VkFFTPlan")
    .allowlist_type("VkFFTApplication")
    .allowlist_function("VkFFTSync")
    .allowlist_function("VkFFTAppend")
    .allowlist_function("VkFFTPlanAxis")
    .allowlist_function("initializeVkFFT")
    .allowlist_function("deleteVkFFT")
    .allowlist_function("VkFFTGetVersion")
    .generate();

  let bindings = match res {
    Ok(x) => x,
    Err(_) => {
      eprintln!("Failed to generate bindings.");
      std::process::exit(1);
    }
  };

  Ok(bindings)
}

/// Include the text of a file in place of the #include statement when making a wrapper.
/// The operation will only take place if the file name or path contains path_hint.
/// For example, if path_hint is "/VkFFT", #include "math.h" will be left as-is but the
/// first instance of #include "vkFFT/VkFFT_Structs/vkFFT_Structs.h" will be replaced by
/// that file's contents. Subsequent #include statements to that file will be removed.
/// This will be done recursively until no more matches are found.
fn process_includes(
  file: &str,
  root_path: &str,
  path_hint: &str,
) -> Result<String, Box<dyn Error>> {
  let mut included_files: HashMap<String, u32> = HashMap::new();
  let mut result: String = std::fs::read_to_string(&format!("{}/{}", root_path, file))?;
  let re = Regex::new(&format!("#include \"{}(.*)\"", path_hint)).unwrap();

  while let Some(capture) = re.captures(&result.clone()) {
    let current_file = &capture[1];
    if included_files.contains_key(current_file) {
      result = result.replace(&format!("#include \"{}{}\"", path_hint, current_file), "");
      continue;
    } else {
      included_files.insert(current_file.to_string(), 1);
    }

    let current_path = format!("{}/{}", root_path, current_file);
    let file_content = std::fs::read_to_string(&current_path)?;
    result = result.replace(
      &format!("#include \"{}{}\"", path_hint, current_file),
      &file_content,
    );
  }

  Ok(result)
}

fn build_vkfft() -> Result<(), Box<dyn Error>>{
  let out_dir = std::env::var("OUT_DIR")?;
  let out_dir = PathBuf::from(out_dir);

  println!("cargo:rerun-if-changed=wrapper.c");
  println!("cargo:rerun-if-changed=build.rs");

  let mut include_dirs = vec!["VkFFT/vkFFT/vkFFt".to_string()];

  if let Ok(var) = env::var("VULKAN_SDK") {
    include_dirs.push(var.to_string()+"/Include");
  }

  let defines = [("VKFFT_BACKEND", "0"), ("VK_API_VERSION", "11")];

  let wrapper = process_includes(
    &format!("../vkFFT.h"),
    "VkFFT/vkFFT/vkFFT",
    "vkFFT"
  )?
  .replace("static inline VkFFTResult VkFFTSync", "VkFFTResult VkFFTSync")
  .replace("static inline VkFFTResult VkFFTAppend", "VkFFTResult VkFFTAppend")
  .replace("static inline VkFFTResult VkFFTPlanAxis", "VkFFTResult VkFFTPlanAxis")
  .replace("static inline VkFFTResult initializeVkFFT", "VkFFTResult initializeVkFFT")
  .replace("static inline void deleteVkFFT", "void deleteVkFFT")
  .replace("static inline int VkFFTGetVersion", "int VkFFTGetVersion")
  .replace("#include \"glslang_c_interface.h\"", "#include \"glslang/Include/glslang_c_interface.h\"")
  .replace("pfLD double_PI;", "double double_PI;")
  .replace("pfLD d; // long double", "double d; uint64_t alignment[2];// long double replaced with double");

  let rw = out_dir.join("vkfft_rw.h");
  
  std::fs::write(&rw, wrapper.as_str())?;

  let mut build = cc::Build::default();

  build
    .file("wrapper.c")
    .warnings(false)
    .include(out_dir.clone());

  build.cargo_metadata(true).static_flag(true);

  for (key, value) in defines.iter() {
    build.define(*key, Some(*value));
  }

  for include_dir in include_dirs.iter() {
    build.include(include_dir);
  }

  build.compile("vkfft");


  let bindings_path = Path::new(BINDGEN_FILENAME);

  if !bindings_path.exists() {
    let bindings = gen_wrapper(&rw, &defines, &include_dirs)?;
    bindings.write_to_file(bindings_path)?;
  }

  Ok(())
}

fn link_vulkan(){
  //logic copied from ash-rs
  let target_family = env::var("CARGO_CFG_TARGET_FAMILY").unwrap();
  let target_pointer_width = env::var("CARGO_CFG_TARGET_POINTER_WIDTH").unwrap();

  println!("cargo:rerun-if-env-changed=VULKAN_SDK");
  if let Ok(var) = env::var("VULKAN_SDK") {
      let suffix = match (&*target_family, &*target_pointer_width) {
          ("windows", "32") => "Lib32",
          ("windows", "64") => "Lib",
          _ => "lib",
      };
      println!("cargo:rustc-link-search={var}/{suffix}");
  }
  let lib = match &*target_family {
      "windows" => "vulkan-1",
      _ => "vulkan",
  };
  println!("cargo:rustc-link-lib={lib}");
}

fn main() -> Result<(), Box<dyn Error>> {
  if env::var("DOCS_RS").is_ok() {
      println!("cargo:warning=Skipping glslang native build for docs.rs.");
      return Ok(());
  }
  build_glslang();
  build_vkfft()?;
  link_vulkan();
  Ok(())
}
