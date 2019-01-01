extern crate torch;

use torch::*;

fn main() {
    match test() {
        Err(message) => panic!(message),
        _ => (),
    };
}
