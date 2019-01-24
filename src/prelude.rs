pub use nalgebra as na;

pub type Scalar = f32;

pub type Vec3 = na::Vector3<Scalar>;
pub type Point3 = na::Point3<Scalar>;

pub use alga::{
    linear::{EuclideanSpace, NormedSpace},
};

pub use crate::{
    System,
    body::{Body},
};