use super::tensor;
use num_traits::Num;

pub trait Function {}

pub struct AddFunction<T: Num> {
    pub factor: tensor::Tensor<T>,
}

impl<T: Num> Function for AddFunction<T> {}
