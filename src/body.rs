use crate::prelude::*;

// Do not put anything with a drop impl
#[derive(Debug, Clone, PartialEq)]
pub struct Body {
    pub mass: Scalar,
    pub pos: Vec3,
    pub vel: Vec3,
    pub stat: bool,
}

impl Body {
    pub fn new(mass: Scalar, pos: Vec3, vel: Vec3, stat: bool) -> Self {
        Body {
            mass,
            pos,
            vel,
            stat,
        }
    }

    pub fn dist(&self, other: &Self) -> Scalar {
        Point3::from(self.pos).distance(&Point3::from(other.pos))
    }

    pub fn dist2(&self, other: &Self) -> Scalar {
        Point3::from(self.pos).distance_squared(&Point3::from(other.pos))
    }
}