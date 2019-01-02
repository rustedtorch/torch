extern crate num_traits;

mod function;
mod tensor;

pub fn test() -> Result<(), &'static str> {
    let x = tensor::Tensor::new_from_vector(vec![1.0, 2.01, 3.02, 4.5]);
    let y = x.add(tensor::Tensor::new_from_vector(vec![0.1, 0.2, 0.3, 0.4]))?;
    let z = tensor::Tensor::new_from_cube(vec![
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
