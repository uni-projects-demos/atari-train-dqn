use std::{
    ffi::{CStr, CString, c_char, c_void},
    path::Path,
    ptr::NonNull,
    sync::Once,
};

const MAX_ACTIONS: usize = 18;
const RAM_BYTES: usize = 128;
const ERROR_BYTES: usize = 512;
const MAX_SCREEN_DIMENSION: i32 = 1024;

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct Info {
    pub reward: f32,
    pub done: i32,
    pub truncated: i32,
    pub lives: i32,
    pub width: i32,
    pub height: i32,
}

unsafe extern "C" {
    fn rl_ale_quiet();
    fn rl_ale_create(
        rom: *const c_char,
        seed: u32,
        sticky: f32,
        acts: *mut i32,
        n: *mut i32,
        w: *mut i32,
        h: *mut i32,
        err: *mut c_char,
        len: usize,
    ) -> *mut c_void;
    fn rl_ale_destroy(handle: *mut c_void);
    fn rl_ale_reset(
        handle: *mut c_void,
        rgb: *mut u8,
        cap: usize,
        ram: *mut u8,
        info: *mut Info,
        err: *mut c_char,
        len: usize,
    ) -> i32;
    fn rl_ale_step(
        handle: *mut c_void,
        action: i32,
        repeat: u32,
        pool: i32,
        rgb: *mut u8,
        cap: usize,
        ram: *mut u8,
        info: *mut Info,
        err: *mut c_char,
        len: usize,
    ) -> i32;
}

pub struct Ale {
    handle: NonNull<c_void>,
    rgb: Vec<u8>,
    ram: [u8; RAM_BYTES],
    acts: Vec<i32>,
    w: usize,
    h: usize,
}

unsafe impl Send for Ale {}

impl Ale {
    pub fn new(path: &Path, seed: u32, sticky: f32) -> Result<Self, String> {
        static INITIALIZE: Once = Once::new();
        INITIALIZE.call_once(|| unsafe {
            rl_ale_quiet();
        });

        if !path.is_file() {
            return Err(format!("ROM not found: {}", path.display()));
        }
        if !sticky.is_finite() || !(0.0..=1.0).contains(&sticky) {
            return Err("Invalid sticky probability".to_owned());
        }

        let path_str: &str = path.to_str().ok_or("ROM path is not UTF-8")?;
        let path_c: CString = CString::new(path_str).map_err(|err| err.to_string())?;
        let mut acts: [i32; MAX_ACTIONS] = [0; MAX_ACTIONS];
        let mut n: i32 = 0;
        let mut w: i32 = 0;
        let mut h: i32 = 0;
        let mut err: [c_char; ERROR_BYTES] = [0; ERROR_BYTES];
        let raw_handle: *mut c_void = unsafe {
            rl_ale_create(
                path_c.as_ptr(),
                seed,
                sticky,
                acts.as_mut_ptr(),
                &mut n,
                &mut w,
                &mut h,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        let handle: NonNull<c_void> = NonNull::new(raw_handle).ok_or_else(|| msg(&err))?;

        if !(1..=MAX_ACTIONS as i32).contains(&n)
            || !(1..=MAX_SCREEN_DIMENSION).contains(&w)
            || !(1..=MAX_SCREEN_DIMENSION).contains(&h)
        {
            unsafe {
                rl_ale_destroy(handle.as_ptr());
            }
            return Err("Invalid ALE dimensions/actions".to_owned());
        }

        let n: usize = usize::try_from(n).expect("validated action count fits usize");
        let w: usize = usize::try_from(w).expect("validated width fits usize");
        let h: usize = usize::try_from(h).expect("validated height fits usize");
        let rgb_len: usize = w
            .checked_mul(h)
            .and_then(|pixels: usize| pixels.checked_mul(3))
            .ok_or_else(|| "ALE RGB buffer size overflow".to_owned())?;

        Ok(Self {
            handle,
            rgb: vec![0; rgb_len],
            ram: [0; RAM_BYTES],
            acts: acts[..n].to_vec(),
            w,
            h,
        })
    }

    pub fn reset(&mut self) -> Result<Info, String> {
        self.call(None)
    }

    pub fn step(&mut self, action: usize, repeat: u32, pool: bool) -> Result<Info, String> {
        if action >= self.acts.len() || !(1..=60).contains(&repeat) {
            return Err("Invalid minimal action or frame repeat".to_owned());
        }
        self.call(Some((action, repeat, pool)))
    }

    fn call(&mut self, action: Option<(usize, u32, bool)>) -> Result<Info, String> {
        let mut info: Info = Info::default();
        let mut err: [c_char; ERROR_BYTES] = [0; ERROR_BYTES];
        let res: i32 = unsafe {
            match action {
                Some((action, repeat, pool)) => {
                    let action: i32 = i32::try_from(action).expect("ALE action index fits i32");
                    rl_ale_step(
                        self.handle.as_ptr(),
                        action,
                        repeat,
                        i32::from(pool),
                        self.rgb.as_mut_ptr(),
                        self.rgb.len(),
                        self.ram.as_mut_ptr(),
                        &mut info,
                        err.as_mut_ptr(),
                        err.len(),
                    )
                }
                None => rl_ale_reset(
                    self.handle.as_ptr(),
                    self.rgb.as_mut_ptr(),
                    self.rgb.len(),
                    self.ram.as_mut_ptr(),
                    &mut info,
                    err.as_mut_ptr(),
                    err.len(),
                ),
            }
        };
        if res != 0 {
            return Err(msg(&err));
        }

        let width: usize = usize::try_from(info.width)
            .map_err(|_| "ALE returned a negative screen width".to_owned())?;
        let height: usize = usize::try_from(info.height)
            .map_err(|_| "ALE returned a negative screen height".to_owned())?;
        if width != self.w || height != self.h {
            return Err("ALE changed screen dimensions".to_owned());
        }
        Ok(info)
    }

    pub fn rgb(&self) -> &[u8] {
        &self.rgb
    }

    pub const fn ram(&self) -> &[u8; RAM_BYTES] {
        &self.ram
    }

    pub const fn dimensions(&self) -> [usize; 2] {
        [self.w, self.h]
    }

    pub fn actions(&self) -> &[i32] {
        &self.acts
    }
}

fn msg(buf: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

impl Drop for Ale {
    fn drop(&mut self) {
        unsafe {
            rl_ale_destroy(self.handle.as_ptr());
        }
    }
}
