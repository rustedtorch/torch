pub mod debug;
pub mod error;
pub mod ops;
pub mod scalar;

use self::scalar::*;
use std::rc::Rc;

pub type Result<T> = std::result::Result<T, error::TensorError>;

pub struct Storage<T: Scalar> {
    elements: Vec<T>,
}

pub trait Function {
    fn clone(&self) -> Box<dyn Function>;
}

pub struct Tensor<T: Scalar> {
    storage: Rc<Storage<T>>,
    dimensions: Vec<usize>,
    src_fn: Option<Box<Function>>,
}

impl<T: Scalar> Clone for Tensor<T> {
    fn clone(&self) -> Tensor<T> {
        Tensor {
            storage: self.storage.clone(),
            dimensions: self.dimensions.iter().cloned().collect(),
            src_fn: match &self.src_fn {
                Some(src_fn_ref) => Some((*src_fn_ref).clone()),
                None => None,
            },
        }
    }
}

impl<T: Scalar> Tensor<T> {
    pub fn new(flattened_data: Vec<T>, dimensions: Vec<usize>) -> Tensor<T> {
        Tensor {
            storage: Rc::new(Storage {
                elements: flattened_data,
            }),
            dimensions,
            src_fn: None,
        }
    }
}
