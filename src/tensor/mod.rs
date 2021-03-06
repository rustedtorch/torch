pub mod debug;
pub mod error;
pub mod ops;
pub mod scalar;
pub mod tensor_macro;

use self::scalar::*;
use im::Vector;
use std::rc::Rc;

pub type Result<T> = std::result::Result<T, error::TensorError>;

pub struct Storage<T: Scalar<T>> {
    elements: Vector<T>,
}

pub trait Function {
    fn clone(&self) -> Box<dyn Function>;
}

pub struct Tensor<T: Scalar<T>> {
    storage: Rc<Storage<T>>,
    dimensions: Vector<usize>,
    src_fn: Option<Box<Function>>,
}

impl<T: Scalar<T>> Clone for Tensor<T> {
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

impl<T: Scalar<T>> Tensor<T> {
    pub fn new(flattened_data: Vector<T>, dimensions: Vector<usize>) -> Tensor<T> {
        Tensor {
            storage: Rc::new(Storage {
                elements: flattened_data,
            }),
            dimensions,
            src_fn: None,
        }
    }
}
