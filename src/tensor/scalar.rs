use std::fmt;
use std::ops;

pub trait Scalar<T>:
    'static
    + Clone
    + fmt::Debug
    + ops::Add<Output = T>
    + ops::Mul<Output = T>
    + ops::Sub<Output = T>
    + ops::Div<Output = T>
{
}

impl Scalar<f64> for f64 {}
impl Scalar<f32> for f32 {}
impl Scalar<i64> for i64 {}
impl Scalar<i32> for i32 {}
impl Scalar<i16> for i16 {}
