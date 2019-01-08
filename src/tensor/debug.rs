use super::*;

use std::fmt;

impl<T: Scalar<T>> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut rendered = vector![];
        let header = format!("tensor(");
        let header_len = header.len();
        rendered.push_back(header);
        self.fmt_recursive(&mut rendered, 0, 0, header_len);
        rendered.push_back(format!(")"));
        let mut joined = String::new();
        for rendered_part in rendered {
            joined = joined + &rendered_part[..];
        }
        write!(f, "{}", joined)
    }
}

impl<T: Scalar<T>> Tensor<T> {
    fn fmt_recursive(
        &self,
        rendered: &mut Vector<String>,
        dim_index: usize,
        index: usize,
        indent: usize,
    ) -> usize {
        if self.dimensions.len() == 0 {
            rendered.push_back(format!("{:?}", self.storage.elements[0]));
            return 0;
        }
        rendered.push_back(format!("["));
        let last_dim = dim_index + 1 == self.dimensions.len();
        let mut local_index = index;
        let length = self.dimensions[dim_index];
        for i in 0..length {
            if last_dim {
                rendered.push_back(format!("{:?}", self.storage.elements[local_index]));
                if i + 1 < length {
                    rendered.push_back(format!(", "));
                }
                local_index += 1;
            } else {
                rendered.push_back(format!("\n{}", " ".repeat(indent + 1)));
                local_index = self.fmt_recursive(rendered, dim_index + 1, local_index, indent + 1);
                if i + 1 < length {
                    rendered.push_back(format!(","));
                } else {
                    rendered.push_back(format!("\n{}", " ".repeat(indent)));
                }
            }
        }
        rendered.push_back(format!("]"));
        local_index
    }
}
