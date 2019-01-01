use super::function;

use num_traits::Num;
use std::fmt;

pub struct Storage<T: Num> {
    elements: Vec<T>,
}

pub struct Tensor<T: Num> {
    storage: Box<Storage<T>>,
    dimensions: Vec<usize>,
    src_fn: Option<Box<function::Function>>,
}

impl<T: Num + std::fmt::Debug> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "tensor(");
        match self.dimensions.len() {
            0 => {
                write!(f, "{:?}", self.storage.elements[0]);
            }
            1 => {
                let mut index = 0usize;
                write!(f, "[");
                for i in 0..self.dimensions[0] {
                    write!(f, "{:?}, ", self.storage.elements[index]);
                    index += 1usize;
                }
                write!(f, "]");
            }
            2 => {
                let mut index = 0usize;
                write!(f, "[");
                for i in 0..self.dimensions[0] {
                    write!(f, "[");
                    for i in 0..self.dimensions[1] {
                        write!(f, "{:?}, ", self.storage.elements[index]);
                        index += 1usize;
                    }
                    write!(f, "]");
                }
                write!(f, "]");
            }
            3 => {
                let mut index = 0usize;
                write!(f, "[");
                for i in 0..self.dimensions[0] {
                    write!(f, "[");
                    for i in 0..self.dimensions[1] {
                        write!(f, "[");
                        for i in 0..self.dimensions[2] {
                            write!(f, "{:?}, ", self.storage.elements[index]);
                            index += 1usize;
                        }
                        write!(f, "]");
                    }
                    write!(f, "]");
                }
                write!(f, "]");
            }
            _ => panic!("Can't display debug format for tensors with more than 3 dimensions"),
        };
        write!(f, ")")
    }
}

impl<T: 'static + Clone + Num> Tensor<T> {
    pub fn new_from_cube(cube: Vec<Vec<Vec<T>>>) -> Tensor<T> {
        let mut storage = Storage { elements: vec![] };
        let mut dimensions = vec![cube.len(), cube[0].len(), cube[0][0].len()];
        for matrix in cube {
            for vector in matrix {
                for scalar in vector {
                    storage.elements.push(scalar);
                }
            }
        }
        Tensor {
            storage: Box::new(storage),
            dimensions,
            src_fn: None,
        }
    }
    pub fn new_from_matrix(matrix: Vec<Vec<T>>) -> Tensor<T> {
        let mut storage = Storage { elements: vec![] };
        let mut dimensions = vec![matrix.len(), matrix[0].len()];
        for vector in matrix {
            for scalar in vector {
                storage.elements.push(scalar);
            }
        }
        Tensor {
            storage: Box::new(storage),
            dimensions,
            src_fn: None,
        }
    }
    pub fn new_from_vector(vector: Vec<T>) -> Tensor<T> {
        let mut storage = Storage { elements: vec![] };
        let mut dimensions = vec![vector.len()];
        for scalar in vector {
            storage.elements.push(scalar);
        }
        Tensor {
            storage: Box::new(storage),
            dimensions,
            src_fn: None,
        }
    }
    pub fn new_from_scalar(scalar: T) -> Tensor<T> {
        let mut storage = Storage { elements: vec![] };
        let mut dimensions = vec![];
        storage.elements.push(scalar);
        Tensor {
            storage: Box::new(storage),
            dimensions,
            src_fn: None,
        }
    }
    pub fn add(&self, other: Tensor<T>) -> Result<Tensor<T>, &'static str> {
        if self.dimensions != other.dimensions {
            return Err("Can't add tensors of different dimensions");
        }
        let mut result = vec![];
        for i in 0..self.storage.elements.len() {
            result.push(
                self.storage.elements[i]
                    .clone()
                    .add(other.storage.elements[i].clone()),
            );
        }
        Ok(Tensor {
            storage: Box::new(Storage { elements: result }),
            dimensions: self.dimensions.iter().cloned().collect(),
            src_fn: Some(Box::new(function::AddFunction { factor: other })),
        })
    }
}
