use super::*;

use std::fmt;

impl<T: Scalar<T>> fmt::Debug for Tensor<T> {
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

impl<T: Scalar<T>> Tensor<T> {
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
}
