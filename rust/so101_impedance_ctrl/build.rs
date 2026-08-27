//! Compiles the cerebellum's compute shaders to SPIR-V at build time.
//!
//! `glslc` (from shaderc, packaged as `glslc` on Debian/Ubuntu) has to be on `PATH`. Shipping the
//! `.spv` blobs pre-built would remove that requirement, but it would also mean a shader and its
//! binary could drift apart in a commit -- and a stale compute shader fails as *wrong numbers*,
//! not as an error. Compiling from source on every build makes that impossible.

use std::path::{Path, PathBuf};
use std::process::Command;

const SHADERS: [&str; 4] = ["granule", "normalise", "purkinje", "learn"];

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let shader_dir = Path::new("shaders");

    println!("cargo:rerun-if-changed=shaders");
    for name in SHADERS {
        println!("cargo:rerun-if-changed=shaders/{name}.comp");
    }
    println!("cargo:rerun-if-changed=shaders/common.glsl");

    for name in SHADERS {
        let src = shader_dir.join(format!("{name}.comp"));
        let dst = out_dir.join(format!("{name}.spv"));
        let output = Command::new("glslc")
            .arg("-O")
            .arg("-I")
            .arg(shader_dir)
            .arg(&src)
            .arg("-o")
            .arg(&dst)
            .output();

        match output {
            Ok(out) if out.status.success() => {}
            Ok(out) => panic!(
                "glslc failed on {}:\n{}",
                src.display(),
                String::from_utf8_lossy(&out.stderr)
            ),
            Err(e) => panic!(
                "could not run glslc to compile {}: {e}\n\
                 Install it with `sudo apt install glslc` (or the shaderc package for your \
                 distribution). It is a build-time requirement only -- the daemon itself needs \
                 just a Vulkan driver at runtime.",
                src.display()
            ),
        }
    }
}
