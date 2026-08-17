//! Port of `src/sundials/stl/sunstl_vector.h` (TTYPE template →
//! Rust generic). `destroyValue` maps to Rust `Drop`; the C `nullish`
//! zero value maps to `T::default()`.

use crate::sundials_types::*;

/* Error codes used by callers (SUN_SUCCESS / SUN_ERR_* come from
sundials_errors). */
use crate::sundials_errors::{SUN_ERR_MALLOC_FAIL, SUN_ERR_OUTOFRANGE, SUN_SUCCESS};

/// Growth factor applied on resize (C: `SUNSTLVECTOR_GROWTH_FACTOR 1.5L`).
pub const SUNSTLVECTOR_GROWTH_FACTOR: f64 = 1.5;

pub struct SUNStlVector<T: Default> {
    size: i64,
    capacity: i64,
    values: Vec<T>,
}

impl<T: Default> SUNStlVector<T> {
    /// C `SUNStlVector_TTYPE_New(init_capacity, destroyValue)`.
    pub fn New(init_capacity: i64) -> Option<Self> {
        if init_capacity < 0 {
            return None;
        }
        let mut values = Vec::new();
        values.reserve_exact(init_capacity as usize);
        Some(SUNStlVector {
            size: 0,
            capacity: init_capacity,
            values,
        })
    }

    pub fn IsEmpty(&self) -> sunbooleantype {
        self.size == 0
    }

    pub fn Reserve(&mut self, new_capacity: i64) -> SUNErrCode {
        if new_capacity <= self.capacity {
            return SUN_SUCCESS;
        }
        self.values.reserve_exact((new_capacity - self.size) as usize);
        self.capacity = new_capacity;
        SUN_SUCCESS
    }

    fn Grow(&mut self) -> SUNErrCode {
        if self.size == self.capacity {
            let new_capacity = if self.capacity == 0 {
                2
            } else {
                ((self.capacity as f64) * SUNSTLVECTOR_GROWTH_FACTOR).ceil() as i64
            };
            return self.Reserve(new_capacity);
        }
        SUN_SUCCESS
    }

    pub fn PushBack(&mut self, element: T) -> SUNErrCode {
        if self.size == self.capacity {
            let err = self.Grow();
            if err != SUN_SUCCESS {
                return err;
            }
        }
        self.values.push(element);
        self.size += 1;
        SUN_SUCCESS
    }

    /// C returns `TTYPE*` or NULL when out of bounds.
    pub fn At(&self, index: i64) -> Option<&T> {
        if index >= self.size || index < 0 {
            return None;
        }
        Some(&self.values[index as usize])
    }

    pub fn AtMut(&mut self, index: i64) -> Option<&mut T> {
        if index >= self.size || index < 0 {
            return None;
        }
        Some(&mut self.values[index as usize])
    }

    pub fn Set(&mut self, index: i64, element: T) -> SUNErrCode {
        if index >= self.size || index < 0 {
            return SUN_ERR_OUTOFRANGE;
        }
        self.values[index as usize] = element;
        SUN_SUCCESS
    }

    pub fn PopBack(&mut self) -> SUNErrCode {
        if self.size == 0 {
            return SUN_SUCCESS;
        }
        self.values.pop();
        self.size -= 1;
        SUN_SUCCESS
    }

    pub fn Erase(&mut self, index: i64) -> SUNErrCode {
        if self.size == 0 {
            return SUN_SUCCESS;
        }
        if index >= self.size || index < 0 {
            return SUN_ERR_OUTOFRANGE;
        }
        self.values.remove(index as usize);
        self.size -= 1;
        SUN_SUCCESS
    }

    pub fn Size(&self) -> i64 {
        self.size
    }

    pub fn Capacity(&self) -> i64 {
        self.capacity
    }
}

/* Silence an unused warning: MALLOC_FAIL exists in the C error surface of
this container but safe Rust allocation cannot observe the failure. */
const _: SUNErrCode = SUN_ERR_MALLOC_FAIL;
