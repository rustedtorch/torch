use num_traits;
use std::fmt;

pub trait Scalar: 'static + Clone + fmt::Debug + num_traits::Num {}

impl Scalar for f64 {}
impl Scalar for f32 {}
impl Scalar for i64 {}
impl Scalar for i32 {}
impl Scalar for i16 {}
