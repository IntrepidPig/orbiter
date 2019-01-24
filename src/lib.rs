use crate::prelude::*;

pub mod body;
pub mod prelude;

pub struct System {
	pub current: Vec<Body>,
	pub next: Vec<Body>,
}

impl System {
	pub fn new() -> Self {
		System {
			current: Vec::new(),
			next: Vec::new(),
		}
	}

	pub fn step(&mut self, g: Scalar, dt: Scalar) {
		println!("dt: {}", dt);
		let len = self.current.len();

		for i in 0..len {
			self.next.push(self.current[i].clone());
		}

		for i in 0..len {
			#[allow(mutable_transmutes)]
			let body: &mut Body = unsafe { std::mem::transmute(&self.current[i]) };

			for j in 0..len {
				if i == j {
					continue;
				}

				#[allow(mutable_transmutes)]
				let bodyb = &self.current[j];

				// F = m * a
				// F = G * (m1 * m2) / r^2
				let f = g * body.mass * bodyb.mass / body.dist2(&bodyb);
				let a = f / body.mass;

				let dv = (bodyb.pos - body.pos).normalize() * a * dt;
				self.next[i].vel += dv;
			}

			let vel = self.next[i].vel;
			self.next[i].pos += vel;
		}

		std::mem::swap(&mut self.current, &mut self.next);
		self.next.clear();
	}
}
