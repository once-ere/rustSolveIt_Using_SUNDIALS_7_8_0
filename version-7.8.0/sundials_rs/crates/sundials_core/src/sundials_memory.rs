//! Port of `src/sundials/sundials_memory.c` +
//! `include/sundials/sundials_memory.h` (SUNMemoryHelper abstraction).
//!
//! Mapping decisions: the C `void* ptr` payload of `SUNMemory_` is an owned
//! byte buffer `data: Vec<u8>` (empty vector = C NULL pointer); the C field
//! `type` (Rust keyword) is `type_`. GPU-only `void* queue` op parameters are
//! ignored placeholders `Option<&mut dyn Any>`; the helper's stored default
//! queue is `Option<Box<dyn Any>>`.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::sundials_context::SUNContext;
use crate::sundials_errors::SUN_SUCCESS;
use crate::sundials_types::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SUNMemoryType {
    /// pageable memory accessible on the host
    SUNMEMTYPE_HOST,
    /// page-locked memory accessible on the host
    SUNMEMTYPE_PINNED,
    /// memory accessible from the device
    SUNMEMTYPE_DEVICE,
    /// memory accessible from the host or device
    SUNMEMTYPE_UVM,
}
pub use SUNMemoryType::*;

/// C `struct SUNMemory_` — a simple abstraction of a pointer to some
/// contiguous memory, so that we can keep track of its type and ownership.
pub struct SUNMemory_ {
    /// C `void* ptr` — owned byte buffer (empty ⇔ C NULL pointer).
    pub data: Vec<u8>,
    /// C `type` (renamed: Rust keyword).
    pub type_: SUNMemoryType,
    pub own: sunbooleantype,
    pub bytes: usize,
    pub stride: usize,
}

pub type SUNMemory = Rc<RefCell<SUNMemory_>>;

#[derive(Default, Clone)]
pub struct SUNMemoryHelper_Ops_ {
    /* operations that implementations are required to provide */
    pub alloc: Option<
        fn(
            &SUNMemoryHelper,
            &mut Option<SUNMemory>,
            usize,
            SUNMemoryType,
            Option<&mut dyn Any>,
        ) -> SUNErrCode,
    >,
    pub dealloc: Option<fn(&SUNMemoryHelper, Option<SUNMemory>, Option<&mut dyn Any>) -> SUNErrCode>,
    pub copy:
        Option<fn(&SUNMemoryHelper, &SUNMemory, &SUNMemory, usize, Option<&mut dyn Any>) -> SUNErrCode>,

    /* operations that provide default implementations */
    pub allocstrided: Option<
        fn(
            &SUNMemoryHelper,
            &mut Option<SUNMemory>,
            usize,
            usize,
            SUNMemoryType,
            Option<&mut dyn Any>,
        ) -> SUNErrCode,
    >,
    pub copyasync:
        Option<fn(&SUNMemoryHelper, &SUNMemory, &SUNMemory, usize, Option<&mut dyn Any>) -> SUNErrCode>,
    pub getallocstats: Option<
        fn(&SUNMemoryHelper, SUNMemoryType, &mut u64, &mut u64, &mut usize, &mut usize) -> SUNErrCode,
    >,
    pub clone: Option<fn(&SUNMemoryHelper) -> Option<SUNMemoryHelper>>,
    pub destroy: Option<fn(SUNMemoryHelper) -> SUNErrCode>,
}

pub type SUNMemoryHelper_Ops = SUNMemoryHelper_Ops_;

pub struct SUNMemoryHelper_ {
    pub content: RefCell<Box<dyn Any>>,
    /// C `void* queue` — default queue for memory helper operations.
    pub queue: RefCell<Option<Box<dyn Any>>>,
    pub ops: RefCell<SUNMemoryHelper_Ops_>,
    pub sunctx: RefCell<SUNContext>,
}

pub type SUNMemoryHelper = Rc<SUNMemoryHelper_>;

/// Creates a new SUNMemory object with a NULL (empty) data buffer.
///
/// C only sets `bytes = 0` and `stride = 1`; the remaining fields are
/// uninitialized in C and get inert defaults here.
pub fn SUNMemoryNewEmpty(_sunctx: &SUNContext) -> Option<SUNMemory> {
    let mem = Rc::new(RefCell::new(SUNMemory_ {
        data: Vec::new(),
        type_: SUNMEMTYPE_HOST,
        own: SUNFALSE,
        bytes: 0,
        stride: 1,
    }));

    Some(mem)
}

/// Creates an empty SUNMemoryHelper object.
///
/// The C `sunctx == NULL` early return cannot occur with `&SUNContext`.
pub fn SUNMemoryHelper_NewEmpty(sunctx: &SUNContext) -> Option<SUNMemoryHelper> {
    let helper = Rc::new(SUNMemoryHelper_ {
        content: RefCell::new(Box::new(())),
        queue: RefCell::new(None),
        ops: RefCell::new(SUNMemoryHelper_Ops_::default()),
        sunctx: RefCell::new(sunctx.clone()),
    });

    Some(helper)
}

/// Copies the SUNMemoryHelper ops structure from `src.ops` to `dst.ops`.
pub fn SUNMemoryHelper_CopyOps(src: &SUNMemoryHelper, dst: &SUNMemoryHelper) -> SUNErrCode {
    *dst.ops.borrow_mut() = src.ops.borrow().clone();
    SUN_SUCCESS
}

/// Checks that all required SUNMemoryHelper ops are provided.
pub fn SUNMemoryHelper_ImplementsRequiredOps(helper: &SUNMemoryHelper) -> sunbooleantype {
    let ops = helper.ops.borrow();
    if ops.alloc.is_none() || ops.dealloc.is_none() || ops.copy.is_none() {
        return SUNFALSE;
    }
    SUNTRUE
}

/// Creates a new SUNMemory object which points to the same data as another
/// SUNMemory object; the result does not own the data.
///
/// C copies the raw `ptr` (shared storage); with owned buffers the alias
/// receives a snapshot copy of the bytes (`own = SUNFALSE` as in C, and
/// `bytes`/`stride` keep their `SUNMemoryNewEmpty` values as in C).
pub fn SUNMemoryHelper_Alias(helper: &SUNMemoryHelper, mem: &SUNMemory) -> Option<SUNMemory> {
    let alias = SUNMemoryNewEmpty(&helper.sunctx.borrow())?;

    {
        let m = mem.borrow();
        let mut a = alias.borrow_mut();
        a.data = m.data.clone();
        a.type_ = m.type_;
        a.own = SUNFALSE;
    }

    Some(alias)
}

/// Creates a new SUNMemory object wrapping user-provided data; the result
/// does not own the data for deallocation-statistics purposes.
///
/// C `void* ptr` becomes an owned `Vec<u8>` handed over by the caller. The C
/// `mem_type` range check is expressed by the `SUNMemoryType` enum itself.
pub fn SUNMemoryHelper_Wrap(
    helper: &SUNMemoryHelper,
    ptr: Vec<u8>,
    mem_type: SUNMemoryType,
) -> Option<SUNMemory> {
    let mem = SUNMemoryNewEmpty(&helper.sunctx.borrow())?;

    {
        let mut m = mem.borrow_mut();
        m.data = ptr;
        m.own = SUNFALSE;
        m.type_ = mem_type;
    }

    Some(mem)
}

pub fn SUNMemoryHelper_GetAllocStats(
    helper: &SUNMemoryHelper,
    mem_type: SUNMemoryType,
    num_allocations: &mut u64,
    num_deallocations: &mut u64,
    bytes_allocated: &mut usize,
    bytes_high_watermark: &mut usize,
) -> SUNErrCode {
    let f = helper.ops.borrow().getallocstats.expect("getallocstats");
    f(
        helper,
        mem_type,
        num_allocations,
        num_deallocations,
        bytes_allocated,
        bytes_high_watermark,
    )
}

pub fn SUNMemoryHelper_Alloc(
    helper: &SUNMemoryHelper,
    memptr: &mut Option<SUNMemory>,
    mem_size: usize,
    mem_type: SUNMemoryType,
    queue: Option<&mut dyn Any>,
) -> SUNErrCode {
    let f = helper.ops.borrow().alloc.expect("alloc");
    f(helper, memptr, mem_size, mem_type, queue)
}

pub fn SUNMemoryHelper_AllocStrided(
    helper: &SUNMemoryHelper,
    memptr: &mut Option<SUNMemory>,
    mem_size: usize,
    stride: usize,
    mem_type: SUNMemoryType,
    queue: Option<&mut dyn Any>,
) -> SUNErrCode {
    let f = helper.ops.borrow().allocstrided.expect("allocstrided");
    f(helper, memptr, mem_size, stride, mem_type, queue)
}

pub fn SUNMemoryHelper_Dealloc(
    helper: &SUNMemoryHelper,
    mem: Option<SUNMemory>,
    queue: Option<&mut dyn Any>,
) -> SUNErrCode {
    let ier;
    if mem.is_none() {
        ier = SUN_SUCCESS;
    } else {
        let f = helper.ops.borrow().dealloc.expect("dealloc");
        ier = f(helper, mem, queue);
    }
    ier
}

pub fn SUNMemoryHelper_Copy(
    helper: &SUNMemoryHelper,
    dst: &SUNMemory,
    src: &SUNMemory,
    memory_size: usize,
    queue: Option<&mut dyn Any>,
) -> SUNErrCode {
    let f = helper.ops.borrow().copy.expect("copy");
    f(helper, dst, src, memory_size, queue)
}

pub fn SUNMemoryHelper_CopyAsync(
    helper: &SUNMemoryHelper,
    dst: &SUNMemory,
    src: &SUNMemory,
    memory_size: usize,
    queue: Option<&mut dyn Any>,
) -> SUNErrCode {
    let copyasync = helper.ops.borrow().copyasync;
    match copyasync {
        None => SUNMemoryHelper_Copy(helper, dst, src, memory_size, queue),
        Some(f) => f(helper, dst, src, memory_size, queue),
    }
}

/// Frees the SUNMemoryHelper.
///
/// The C `helper == NULL` early return cannot occur with a by-value handle.
pub fn SUNMemoryHelper_Destroy(helper: SUNMemoryHelper) -> SUNErrCode {
    let mut err = SUN_SUCCESS;

    let destroy = helper.ops.borrow().destroy;
    if let Some(f) = destroy {
        /* user helper defined destroy */
        err = f(helper);
    } else {
        /* default destroy */
        drop(helper);
    }

    err
}

/// Clones the SUNMemoryHelper.
///
/// C `helper->content != NULL` maps to the content box holding something
/// other than the empty `()` placeholder set by `SUNMemoryHelper_NewEmpty`.
pub fn SUNMemoryHelper_Clone(helper: &SUNMemoryHelper) -> Option<SUNMemoryHelper> {
    let clone_op = helper.ops.borrow().clone;
    match clone_op {
        None => {
            if !helper.content.borrow().is::<()>() {
                None
            } else {
                let hclone = SUNMemoryHelper_NewEmpty(&helper.sunctx.borrow());
                if let Some(hclone) = &hclone {
                    SUNMemoryHelper_CopyOps(helper, hclone);
                }
                hclone
            }
        }
        Some(f) => f(helper),
    }
}

/// Sets the default queue to use for memory helper operations.
///
/// Stored as `Option<Box<dyn Any>>` (the op-parameter placeholders are
/// `Option<&mut dyn Any>`); unused by the serial system implementation.
pub fn SUNMemoryHelper_SetDefaultQueue(
    helper: &SUNMemoryHelper,
    queue: Option<Box<dyn Any>>,
) -> SUNErrCode {
    *helper.queue.borrow_mut() = queue;

    SUN_SUCCESS
}
