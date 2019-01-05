use std::error;
use std::fmt;

#[derive(Debug, Clone)]
pub struct TensorError {
    message: String,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "tensor error: {}", self.message)
    }
}

impl error::Error for TensorError {
    fn cause(&self) -> Option<&error::Error> {
        None
    }
}

impl TensorError {
    pub fn new(message: &str) -> TensorError {
        TensorError {
            message: String::from(message),
        }
    }
}
