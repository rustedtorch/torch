use super::super::*;

impl<T: Scalar> Tensor<T> {
    pub fn backward(&self, _gradient: Option<&Tensor<T>>) {}
}
