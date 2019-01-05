use super::super::*;

impl<T: Scalar> Tensor<T>
{
    pub fn backward(&self, gradient: Option<&Tensor<T>>) {}
}
