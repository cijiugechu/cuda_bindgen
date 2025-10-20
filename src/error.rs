use cudarc::driver::sys::cudaError_enum;
use cudarc::driver::result::DriverError;

/// Error messages
#[derive(Debug, Clone)]
pub enum Error {
    CudaRunTimeSys(cudaError_enum),
    CudaRunTimeDriver(DriverError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self)
    }
}

impl std::error::Error for Error {}

impl From<cudaError_enum> for Error {
    fn from(error: cudaError_enum) -> Self {
        Self::CudaRunTimeSys(error)
    }
}

impl From<DriverError> for Error {
    fn from(error: DriverError) -> Self {
        Self::CudaRunTimeDriver(error)
    }
}

pub type Result<T> = std::result::Result<T, Error>;