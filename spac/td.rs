use spa::prelude::*;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
	pub pos: Point3,
	pub col: Vec4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelViewProjection {
	pub model: Mat4,
	pub view: Mat4,
	pub proj: Mat4,
}

impl ModelViewProjection {
	pub fn new(proj: Mat4) -> Self {
		ModelViewProjection {
			model: Mat4::new_rotation(Vec3::zeros()),
			view: Mat4::look_at_rh(&Point3::from([0.0, 0.0, -1.0]), &Point3::from([0.0, 0.0, 0.0]), &Vec3::from([0.0, 1.0, 0.0])),
			proj,
		}
	}
}