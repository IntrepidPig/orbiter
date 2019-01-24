#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
	pos: [f32; 3],
	col: [f32; 4],
}

pub const TRI: [Vertex; 3] = [
	Vertex {
		pos: [0.0, -1.0, 0.0],
		col: [1.0, 0.0, 0.0, 1.0],
	},
	Vertex {
		pos: [-1.0, 0.0, 0.0],
		col: [0.0, 0.0, 1.0, 1.0],
	},
	Vertex {
		pos: [0.0, 1.0, 0.0],
		col: [0.0, 1.0, 0.0, 1.0],
	},
];
