use super::tensor;
use num_traits::Num;

pub trait Function {
    fn clone(&self) -> Box<dyn Function>;
}

pub struct AddFunction<T>
where
    T: 'static + Clone + Num + std::fmt::Debug,
{
    pub factor: tensor::Tensor<T>,
}

impl<T> Function for AddFunction<T>
where
    T: 'static + Clone + Num + std::fmt::Debug,
{
    fn clone(&self) -> Box<dyn Function> {
        Box::new(AddFunction {
            factor: self.factor.clone(),
        })
    }
}
