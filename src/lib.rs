extern crate num_traits;

pub mod function;
pub mod tensor;

pub fn to_value(x: &f64) -> Result<&f64, &'static str> {
    Ok(x)
}

#[macro_export(local_inner_macros)]
macro_rules! tensor {
    ($($element:tt)+) => {
        tensor_internal!($($element)+)
    };
}

#[macro_export(local_inner_macros)]
macro_rules! tensor_internal {
    // Done with trailing comma.
    (@array [$($elems:expr,)*]) => {
        tensor_internal_vec![$($elems,)*]
    };

    // Done without trailing comma.
    (@array [$($elems:expr),*]) => {
        tensor_internal_vec![$($elems),*]
    };

    // Next element is an array.
    (@array [$($elems:expr,)*] [$($array:tt)*] $($rest:tt)*) => {
        tensor_internal!(@array [$($elems,)* tensor_internal!([$($array)*])] $($rest)*)
    };

    // Next element is an expression followed by comma.
    (@array [$($elems:expr,)*] $next:expr, $($rest:tt)*) => {
        tensor_internal!(@array [$($elems,)* tensor_internal!($next),] $($rest)*)
    };

    // Last element is an expression with no trailing comma.
    (@array [$($elems:expr,)*] $last:expr) => {
        tensor_internal!(@array [$($elems,)* tensor_internal!($last)])
    };

    // Comma after the most recent element.
    (@array [$($elems:expr),*] , $($rest:tt)*) => {
        tensor_internal!(@array [$($elems,)*] $($rest)*)
    };

    // Unexpected token after most recent element.
    (@array [$($elems:expr),*] $unexpected:tt $($rest:tt)*) => {
        tensor_unexpected!($unexpected)
    };

    ([]) => {
        tensor_internal_vec![]
    };

    ([ $($tt:tt)+ ]) => {
        tensor_internal!(@array [] $($tt)+)
    };

    // Any Serialize type: numbers, strings, struct literals, variables etc.
    // Must be below every other rule.
    ($other:expr) => {
        $crate::to_value(&$other).unwrap()
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
