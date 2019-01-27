pub use nalgebra as na;
pub use nalgebra_glm as nag;

pub type Scalar = f32;

pub type Vec3 = na::Vector3<Scalar>;
pub type Vec4 = na::Vector4<Scalar>;
pub type Point3 = na::Point3<Scalar>;
pub type Mat4 = na::Matrix4<Scalar>;

pub use alga::linear::{EuclideanSpace, NormedSpace};

pub use crate::{body::Body, System};
