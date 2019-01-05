use super::super::*;

pub struct MulFunction<T: Scalar> {
    pub factor: Tensor<T>,
}

impl<T: Scalar> Function for MulFunction<T> {
    fn clone(&self) -> Box<dyn Function> {
        Box::new(MulFunction {
            factor: self.factor.clone(),
        })
    }
}

impl<T: Scalar> Tensor<T> {
    pub fn mul(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        if self.dimensions != other.dimensions {
            return Err(TensorError {
                message: "Can't multiply tensors of unmatching dimensions",
            });
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
            src_fn: Some(Box::new(MulFunction {
                factor: other.clone(),
            })),
        })
    }
}
