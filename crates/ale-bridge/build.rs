use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=native");
    let dest: PathBuf = cmake::Config::new("native")
        .define("BUILD_SHARED_LIBS", "OFF")
        .build();
    println!("cargo:rustc-link-search=native={}/lib", dest.display());
    println!("cargo:rustc-link-lib=static=rl_ale");
    println!("cargo:rustc-link-lib=static=z");

    let target_os: String = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=c++");
    } else if target_os != "windows" {
        println!("cargo:rustc-link-lib=stdc++");
    }
}
