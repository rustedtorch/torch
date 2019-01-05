pub mod ops;

use num_traits::Num;
use std::rc::Rc;

pub trait Scalar: 'static + Clone + fmt::Debug + Num {}

impl Scalar for f64 {}
impl Scalar for f32 {}
impl Scalar for i64 {}
impl Scalar for i32 {}
impl Scalar for i16 {}

// <CUSTOM-ERROR>

use std::error;
use std::fmt;

pub type Result<T> = std::result::Result<T, TensorError>;

#[derive(Debug, Clone)]
pub struct TensorError {
    message: &'static str,
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

// </CUSTOM-ERROR>

pub struct Storage<T: Num> {
    elements: Vec<T>,
}

pub trait Function {
    fn clone(&self) -> Box<dyn Function>;
}

pub struct Tensor<T: Num> {
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

impl<T: Scalar> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut rendered = vec![];
        let header = format!("tensor(");
        let header_len = header.len();
        rendered.push(header);
        self.fmt_recursive(&mut rendered, 0, 0, header_len);
        rendered.push(format!(")"));
        write!(f, "{}", rendered.join(""))
    }
}

impl<T: Scalar> Tensor<T> {
    fn fmt_recursive(
        &self,
        rendered: &mut Vec<String>,
        dim_index: usize,
        index: usize,
        indent: usize,
    ) -> usize {
        if self.dimensions.len() == 0 {
            rendered.push(format!("{:?}", self.storage.elements[0]));
            return 0;
        }
        rendered.push(format!("["));
        let last_dim = dim_index + 1 == self.dimensions.len();
        let mut local_index = index;
        let length = self.dimensions[dim_index];
        for i in 0..length {
            if last_dim {
                rendered.push(format!("{:?}", self.storage.elements[local_index]));
                if i + 1 < length {
                    rendered.push(format!(", "));
                }
                local_index += 1;
            } else {
                rendered.push(format!("\n{}", " ".repeat(indent + 1)));
                local_index = self.fmt_recursive(rendered, dim_index + 1, local_index, indent + 1);
                if i + 1 < length {
                    rendered.push(format!(","));
                } else {
                    rendered.push(format!("\n{}", " ".repeat(indent)));
                }
            }
        }
        rendered.push(format!("]"));
        local_index
    }
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
