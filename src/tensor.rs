use super::function::{AddFunction, Function};

use num_traits::Num;

pub struct Storage<T: Num> {
    elements: Vec<T>,
}

pub struct Tensor<T: Num> {
    storage: Box<Storage<T>>,
    dimensions: Vec<usize>,
    src_fn: Option<Box<Function>>,
}

impl<T> std::fmt::Debug for Tensor<T>
where
    T: 'static + Clone + Num + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut rendered = vec![];
        let header = format!("tensor(");
        let header_len = header.len();
        rendered.push(header);
        self.fmt_recursive(&mut rendered, 0, 0, header_len);
        rendered.push(format!(")"));
        write!(f, "{}", rendered.join(""))
    }
}

impl<T> Tensor<T>
where
    T: 'static + Clone + Num + std::fmt::Debug,
{
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
            storage: Box::new(Storage {
                elements: flattened_data,
            }),
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
            src_fn: Some(Box::new(AddFunction { factor: other })),
        })
    }
}
