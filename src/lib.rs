extern crate num_traits;

pub mod function;
pub mod tensor;

// let flattened_data = vec![];
// let dimensions = vec![];
// tensor::Tensor::new(flattened_data, dimensions);

#[macro_export(local_inner_macros)]
macro_rules! tensor {
    ($($element:tt)+) => {
        tensor_f32!($($element)+)
    };
}

#[macro_export(local_inner_macros)]
macro_rules! tensor_f64 {
    ($($element:tt)+) => {
        tensor_internal!(f64, $($element)+)
    };
}

#[macro_export(local_inner_macros)]
macro_rules! tensor_f32 {
    ($($element:tt)+) => {
        tensor_internal!(f32, $($element)+)
    };
}

#[macro_export(local_inner_macros)]
macro_rules! tensor_i64 {
    ($($element:tt)+) => {
        tensor_internal!(i64, $($element)+)
    };
}

#[macro_export(local_inner_macros)]
macro_rules! tensor_i32 {
    ($($element:tt)+) => {
        tensor_internal!(i32, $($element)+)
    };
}

#[macro_export(local_inner_macros)]
macro_rules! tensor_i16 {
    ($($element:tt)+) => {
        tensor_internal!(i16, $($element)+)
    };
}

#[macro_export(local_inner_macros)]
macro_rules! tensor_internal {
    // Done with trailing comma.
    ($type:ty, @array [$($elems:expr,)*]) => {
       tensor_internal_vec![$($elems,)*]
    };

    // Done without trailing comma.
    ($type:ty, @array [$($elems:expr),*]) => {
        tensor_internal_vec![$($elems),*]
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
        (|val: $type| -> $type { val }) ($other)
    };
}

#[macro_export]
macro_rules! tensor_internal_vec {
    ($($content:tt)*) => {
        vec![$($content)*]
    };
}

#[macro_export]
macro_rules! tensorunexpected {
    () => {};
}
