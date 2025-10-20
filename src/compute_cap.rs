use std::ffi::c_int;
use cudarc::driver::sys::{cuDeviceComputeCapability, CUresult};
use cudarc::driver::CudaContext;
use crate::error::{Result, Error};

/// CUDA compute capability (major.minor) for a device.
///
/// Implements `Display` as a two-digit code (e.g. 90 for 9.0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputeCapability {
    major: c_int,
    minor: c_int,
}

impl std::fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let final_cap = self.major * 10 + self.minor;
        write!(f, "{final_cap}")
    }
}

/// Queries the primary device and returns its CUDA compute capability.
///
/// On success, returns a [`ComputeCapability`]; on failure returns [`crate::error::Error`].
pub fn get() -> Result<ComputeCapability> {
    let mut major = 0;
    let mut minor = 0;
    let cuda_context = CudaContext::new(0)?;
    let device = cuda_context.cu_device();
    unsafe {
        let result = cuDeviceComputeCapability(&mut major, &mut minor, device);
        if result != CUresult::CUDA_SUCCESS {
            return Err(Error::CudaRunTimeSys(result));
        }
    }
    Ok(ComputeCapability { major, minor })
}
