use super::super::*;

impl<T: Scalar<T>> Tensor<T> {
    pub fn backward(&self, _gradient: Option<&Tensor<T>>) {}
}
