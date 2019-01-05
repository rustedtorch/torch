use super::super::*;

pub struct AddFunction<T: Scalar> {
    pub factor: Tensor<T>,
}

impl<T: Scalar> Function for AddFunction<T> {
    fn clone(&self) -> Box<dyn Function> {
        Box::new(AddFunction {
            factor: self.factor.clone(),
        })
    }
}

impl<T: Scalar> Tensor<T> {
    pub fn add(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        if self.dimensions != other.dimensions {
            return Err(error::TensorError::new(
                "Can't add tensors of different dimensions",
            ));
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
            storage: Rc::new(Storage { elements: result }),
            dimensions: self.dimensions.iter().cloned().collect(),
            src_fn: Some(Box::new(AddFunction {
                factor: other.clone(),
            })),
        })
    }
}
