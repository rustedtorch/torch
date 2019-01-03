extern crate num_traits;

mod function;
mod tensor;

use tensor::Tensor;

pub fn test() -> Result<(), &'static str> {
    let x = Tensor::new_from_vector(vec![1.0, 2.01, 3.02, 4.5]);
    let y = x.add(Tensor::new_from_vector(vec![0.1, 0.2, 0.3, 0.4]))?;
    let z = Tensor::new_from_cube(vec![
        vec![
            vec![1.0, 2.01, 3.02, 4.5],
            vec![1.0, 2.01, 3.02, 4.5],
            vec![1.0, 2.01, 3.02, 4.5],
            vec![1.0, 2.01, 3.02, 4.5],
        ],
        vec![
            vec![1.0, 2.01, 3.02, 4.5],
            vec![1.0, 2.01, 3.02, 4.5],
            vec![1.0, 2.01, 3.02, 4.5],
            vec![1.0, 2.01, 3.02, 4.5],
        ],
    ]);
    println!("{:?}", z);
    Ok(())
}
