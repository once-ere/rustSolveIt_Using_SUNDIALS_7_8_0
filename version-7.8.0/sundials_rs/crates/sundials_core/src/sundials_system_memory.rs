//! Port of `src/sunmemory/system/sundials_system_memory.c` +
//! `include/sunmemory/sunmemory_system.h` (host-heap SUNMemoryHelper).
//!
//! Mapping decisions: C `malloc(mem_size)` becomes a zero-filled
//! `Vec<u8>` and `free(mem->ptr)` clears the vector; the C
//! `mem->ptr != NULL` test maps to a non-empty data buffer. The GPU-only
//! `queue` parameters are ignored exactly as in the serial C implementation.

use std::any::Any;
use std::cell::RefMut;
use std::rc::Rc;

use crate::sundials_context::SUNContext;
use crate::sundials_errors::SUN_SUCCESS;
use crate::sundials_math::SUNMAX;
use crate::sundials_memory::*;
use crate::sundials_types::*;

pub struct SUNMemoryHelper_Content_Sys_ {
    pub num_allocations: u64,
    pub num_deallocations: u64,
    pub bytes_allocated: usize,
    pub bytes_high_watermark: usize,
}

pub type SUNMemoryHelper_Content_Sys = SUNMemoryHelper_Content_Sys_;

/// C `SUNHELPER_CONTENT(h)`.
fn content_mut(helper: &SUNMemoryHelper) -> RefMut<'_, SUNMemoryHelper_Content_Sys_> {
    RefMut::map(helper.content.borrow_mut(), |c| {
        c.downcast_mut::<SUNMemoryHelper_Content_Sys_>()
            .expect("Sys SUNMemoryHelper content")
    })
}

pub fn SUNMemoryHelper_Sys(sunctx: &SUNContext) -> Option<SUNMemoryHelper> {
    /* Allocate the helper */
    let helper = SUNMemoryHelper_NewEmpty(sunctx)?;

    /* Set the ops */
    {
        let mut ops = helper.ops.borrow_mut();
        ops.alloc = Some(SUNMemoryHelper_Alloc_Sys);
        ops.allocstrided = Some(SUNMemoryHelper_AllocStrided_Sys);
        ops.dealloc = Some(SUNMemoryHelper_Dealloc_Sys);
        ops.copy = Some(SUNMemoryHelper_Copy_Sys);
        ops.getallocstats = Some(SUNMemoryHelper_GetAllocStats_Sys);
        ops.clone = Some(SUNMemoryHelper_Clone_Sys);
        ops.destroy = Some(SUNMemoryHelper_Destroy_Sys);
    }

    /* Attach content and ops */
    *helper.content.borrow_mut() = Box::new(SUNMemoryHelper_Content_Sys_ {
        num_allocations: 0,
        num_deallocations: 0,
        bytes_allocated: 0,
        bytes_high_watermark: 0,
    });

    Some(helper)
}

pub fn SUNMemoryHelper_Alloc_Sys(
    helper: &SUNMemoryHelper,
    memptr: &mut Option<SUNMemory>,
    mem_size: usize,
    mem_type: SUNMemoryType,
    _queue: Option<&mut dyn Any>,
) -> SUNErrCode {
    let mem = SUNMemoryNewEmpty(&helper.sunctx.borrow()).expect("SUNMemoryNewEmpty");

    {
        let mut m = mem.borrow_mut();
        m.data = Vec::new();
        m.own = SUNTRUE;
        m.type_ = mem_type;
        m.bytes = mem_size;

        if mem_type == SUNMEMTYPE_HOST || mem_type == SUNMEMTYPE_UVM {
            m.data = vec![0u8; mem_size];
            let mut content = content_mut(helper);
            content.bytes_allocated += mem_size;
            content.num_allocations += 1;
            content.bytes_high_watermark =
                SUNMAX(content.bytes_allocated, content.bytes_high_watermark);
        }
    }

    *memptr = Some(mem);
    SUN_SUCCESS
}

pub fn SUNMemoryHelper_AllocStrided_Sys(
    helper: &SUNMemoryHelper,
    memptr: &mut Option<SUNMemory>,
    mem_size: usize,
    stride: usize,
    mem_type: SUNMemoryType,
    queue: Option<&mut dyn Any>,
) -> SUNErrCode {
    /* C SUNCheckCall: evaluates the call but never early-returns in this
     * build configuration */
    let _ = SUNMemoryHelper_Alloc_Sys(helper, memptr, mem_size, mem_type, queue);

    memptr.as_ref().expect("memptr").borrow_mut().stride = stride;

    SUN_SUCCESS
}

pub fn SUNMemoryHelper_Dealloc_Sys(
    helper: &SUNMemoryHelper,
    mem: Option<SUNMemory>,
    _queue: Option<&mut dyn Any>,
) -> SUNErrCode {
    let mem = match mem {
        None => return SUN_SUCCESS,
        Some(mem) => mem,
    };

    {
        let mut m = mem.borrow_mut();
        if !m.data.is_empty() && m.own {
            if m.type_ == SUNMEMTYPE_HOST || m.type_ == SUNMEMTYPE_UVM {
                {
                    let mut content = content_mut(helper);
                    content.num_deallocations += 1;
                    /* C size_t arithmetic wraps on underflow */
            content.bytes_allocated = content.bytes_allocated.wrapping_sub(m.bytes);
                }
                /* free(mem->ptr); mem->ptr = NULL; */
                m.data = Vec::new();
            }
        }
    }

    /* free(mem) */
    drop(mem);
    SUN_SUCCESS
}

pub fn SUNMemoryHelper_Copy_Sys(
    _helper: &SUNMemoryHelper,
    dst: &SUNMemory,
    src: &SUNMemory,
    memory_size: usize,
    _queue: Option<&mut dyn Any>,
) -> SUNErrCode {
    /* memcpy(dst->ptr, src->ptr, memory_size); a self-copy through aliased
     * handles is the identity */
    if !Rc::ptr_eq(dst, src) {
        let s = src.borrow();
        let mut d = dst.borrow_mut();
        d.data[..memory_size].copy_from_slice(&s.data[..memory_size]);
    }
    SUN_SUCCESS
}

pub fn SUNMemoryHelper_GetAllocStats_Sys(
    helper: &SUNMemoryHelper,
    _mem_type: SUNMemoryType,
    num_allocations: &mut u64,
    num_deallocations: &mut u64,
    bytes_allocated: &mut usize,
    bytes_high_watermark: &mut usize,
) -> SUNErrCode {
    let content = content_mut(helper);
    *num_allocations = content.num_allocations;
    *num_deallocations = content.num_deallocations;
    *bytes_allocated = content.bytes_allocated;
    *bytes_high_watermark = content.bytes_high_watermark;
    SUN_SUCCESS
}

pub fn SUNMemoryHelper_Clone_Sys(helper: &SUNMemoryHelper) -> Option<SUNMemoryHelper> {
    SUNMemoryHelper_Sys(&helper.sunctx.borrow())
}

pub fn SUNMemoryHelper_Destroy_Sys(helper: SUNMemoryHelper) -> SUNErrCode {
    /* content, ops, and the helper itself are all dropped with the handle */
    drop(helper);
    SUN_SUCCESS
}
