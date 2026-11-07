use anyhow::{Context, Result, bail};
use serde::{de::DeserializeOwned, Serialize};
use std::path::Path;

#[derive(Clone, Copy)]
enum ConfigFormat {
    Json,
    Toml,
}

impl ConfigFormat {
    fn from_path(path: &Path) -> Result<Self> {
        let ext: Option<String> = path
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .map(str::to_ascii_lowercase);
        match ext.as_deref() {
            Some("json") => Ok(Self::Json),
            Some("toml") => Ok(Self::Toml),
            _ => bail!(
                "Unsupported configuration format: {}; use .json or .toml",
                path.display()
            ),
        }
    }
}

pub(super) fn read<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let fmt: ConfigFormat = ConfigFormat::from_path(path)?;
    let txt: String = std::fs::read_to_string(path)
        .with_context(|| format!("Could not read configuration {}", path.display()))?;
    match fmt {
        ConfigFormat::Json => serde_json::from_str::<T>(&txt)
            .with_context(|| format!("Invalid JSON configuration {}", path.display())),
        ConfigFormat::Toml => toml::from_str::<T>(&txt)
            .with_context(|| format!("Invalid TOML configuration {}", path.display())),
    }
}

pub(super) fn write<T: Serialize>(path: &Path, cfg: &T) -> Result<()> {
    let fmt: ConfigFormat = ConfigFormat::from_path(path)?;
    let mut txt: String = match fmt {
        ConfigFormat::Json => serde_json::to_string_pretty(cfg)?,
        ConfigFormat::Toml => toml::to_string_pretty(cfg)?,
    };
    if !txt.ends_with('\n') {
        txt.push('\n');
    }
    if let Some(parent) = path
        .parent()
        .filter(|parent: &&Path| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Could not create directory {}", parent.display()))?;
    }
    std::fs::write(path, txt)
        .with_context(|| format!("Could not write configuration {}", path.display()))
}
