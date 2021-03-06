use super::*;
use im::Vector;

#[derive(Clone)]
pub enum Data<T> {
    Item(T),
    Vector(Vector<Data<T>>),
}

pub fn flatten<T: Scalar<T>>(data: Data<T>) -> Tensor<T> {
    let (flattened_data, dimensions) = flatten_recursive(data);
    Tensor::new(flattened_data, dimensions)
}

pub fn flatten_recursive<T: Scalar<T>>(data: Data<T>) -> (Vector<T>, Vector<usize>) {
    match data {
        Data::Item(item) => (vector![item], vector![]),
        Data::Vector(vector) => {
            let mut flattened_data = vector![];
            let mut current_inner_dimension = 0;
            let mut dimensions = vector![vector.len()];
            let mut dimension_recorded = false;
            for item in vector {
                let (inner_flattened_data, inner_dimensions) = flatten_recursive(item);
                if !dimension_recorded {
                    if inner_dimensions.len() > 0 {
                        current_inner_dimension = inner_dimensions[0];
                    }
                    dimensions.extend(inner_dimensions);
                    dimension_recorded = true;
                } else {
                    if inner_dimensions.len() > 0 {
                        if current_inner_dimension != inner_dimensions[0] {
                            if current_inner_dimension == 0 {
                                panic!(
                                    "Expected scalar got vector of length {}",
                                    inner_dimensions[0]
                                );
                            } else {
                                panic!(
                                    "Expected vector of length {} got vector of length {}",
                                    current_inner_dimension, inner_dimensions[0]
                                );
                            }
                        }
                    } else {
                        if current_inner_dimension != 0 {
                            panic!(
                                "Expected vector of length {} got scalar",
                                current_inner_dimension
                            );
                        }
                    }
                }
                flattened_data.extend(inner_flattened_data);
            }
            (flattened_data, dimensions)
        }
    }
}

#[macro_export(local_inner_macros)]
macro_rules! tensor {
    ($($element:tt)+) => {
        tensor_f32!($($element)+)
    };
}

#[macro_export(local_inner_macros)]
macro_rules! tensor_f64 {
    ($($element:tt)+) => {
        $crate::tensor::tensor_macro::flatten(tensor_internal!(f64, $($element)+))
    };
}

#[macro_export(local_inner_macros)]
macro_rules! tensor_f32 {
    ($($element:tt)+) => {
        $crate::tensor::tensor_macro::flatten(tensor_internal!(f32, $($element)+))
    };
}

#[macro_export(local_inner_macros)]
macro_rules! tensor_i64 {
    ($($element:tt)+) => {
        $crate::tensor::tensor_macro::flatten(tensor_internal!(i64, $($element)+))
    };
}

#[macro_export(local_inner_macros)]
macro_rules! tensor_i32 {
    ($($element:tt)+) => {
        $crate::tensor::tensor_macro::flatten(tensor_internal!(i32, $($element)+))
    };
}

#[macro_export(local_inner_macros)]
macro_rules! tensor_i16 {
    ($($element:tt)+) => {
        $crate::tensor::tensor_macro::flatten(tensor_internal!(i16, $($element)+))
    };
}

#[macro_export(local_inner_macros)]
#[doc(hidden)]
macro_rules! tensor_internal {
    // Done with trailing comma.
    ($type:ty, @array [$($elems:expr,)*]) => {
       tensor_internal_vector![$($elems,)*]
    };

    // Done without trailing comma.
    ($type:ty, @array [$($elems:expr),*]) => {
        tensor_internal_vector![$($elems),*]
    };

    // Next element is an array.
    ($type:ty, @array [$($elems:expr,)*] [$($array:tt)*] $($rest:tt)*) => {
        tensor_internal!($type, @array [$($elems,)* tensor_internal!($type, [$($array)*])] $($rest)*)
    };

    // Next element is an expression followed by comma.
    ($type:ty, @array [$($elems:expr,)*] $next:expr, $($rest:tt)*) => {
        tensor_internal!($type, @array [$($elems,)* tensor_internal!($type, $next),] $($rest)*)
    };

    // Last element is an expression with no trailing comma.
    ($type:ty, @array [$($elems:expr,)*] $last:expr) => {
        tensor_internal!($type, @array [$($elems,)* tensor_internal!($type, $last)])
    };

    // Comma after the most recent element.
    ($type:ty, @array [$($elems:expr),*] , $($rest:tt)*) => {
        tensor_internal!($type, @array [$($elems,)*] $($rest)*)
    };

    // Unexpected token after most recent element.
    ($type:ty, @array [$($elems:expr),*] $unexpected:tt $($rest:tt)*) => {
        tensor_unexpected!($unexpected)
    };

    ($type:ty, [ $($tt:tt)+ ]) => {
        tensor_internal!($type, @array [] $($tt)+)
    };

    // Any Serialize type: numbers, strings, struct literals, variables etc.
    // Must be below every other rule.
    ($type:ty, $other:expr) => {
        // And we force it to always be $type
        (|val: $type| -> $crate::tensor::tensor_macro::Data<$type> { $crate::tensor::tensor_macro::Data::Item(val) }) ($other)
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! tensor_internal_vector {
    ($($content:tt)*) => {
        $crate::tensor::tensor_macro::Data::Vector(vector![$($content)*])
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! tensorunexpected {
    () => {};
}
