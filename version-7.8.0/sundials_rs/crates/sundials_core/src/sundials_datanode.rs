//! Port of `src/sundials/sundials_datanode.c` +
//! `src/sundials/sundials_datanode.h` (the generic SUNDataNode class).
//!
//! The SUNDataNode class is a hierarchical object which could be used to
//! build something like a JSON tree. Nodes can be lists, objects, or leaves.
//! Using the JSON analogy:
//!
//! ```text
//!   "i_am_object": {
//!     "i_am_list": [
//!       "i_am_object": {...
//!       },
//!       "i_am_leaf"
//!     ],
//!     "i_am_leaf": "value"
//!   }
//! ```
//!
//! Object nodes hold named nodes (children), while list nodes hold anonymous
//! nodes (children). Leaf nodes do not have children, they have values. The
//! SUNDataNode can be used to build all sorts of useful things, but we
//! primarily use it as the backbone for checkpointing states in adjoint
//! sensitivity analysis.
//!
//! Mapping decisions (handle model, ARCHITECTURE.md):
//!
//! * `SUNDataNode` is `Rc<SUNDataNode_>`; cloning the `Rc` is the C pointer
//!   copy, `Rc::ptr_eq` is C pointer equality, dropping frees. The C
//!   `void* content` is `content: RefCell<Box<dyn Any>>` (the empty `()` box is
//!   C `NULL`), and the ops table is embedded as `RefCell<SUNDataNode_Ops_>`
//!   of plain `Option<fn>` pointers taking `&SUNDataNode` — the identical call
//!   shape to C.
//! * `dtype` is the one plain scalar field of the C base struct and must stay
//!   writable through the shared handle: implementations set it right after
//!   `SUNDataNode_CreateEmpty` (C `BASE_MEMBER(node, dtype) = ...`) and read it
//!   back inside their ops. It is therefore `RefCell<SUNDataNodeType>`, matching
//!   the house pattern for every other mutable-through-the-handle base field;
//!   both accesses are transient (`SUNDataNodeType` is `Copy`), so no borrow is
//!   ever held across an op call.
//! * C `void** data` (the `getdata` output) hands back the live pointer into
//!   the leaf's buffer. Safe Rust cannot alias a `Vec`'s interior, so the
//!   output is the leaf's `SUNMemory` handle instead — one level of
//!   indirection more than C, but it keeps the essential property: the caller
//!   observes the node's live bytes (`mem.borrow().data`) rather than a copy.
//! * C `void* data` (the `setdata` input) is handed straight to
//!   `SUNMemoryHelper_Wrap`, whose ported signature takes an owned `Vec<u8>`
//!   (accepted deviation class 2, ownership snapshots). `setdata` therefore
//!   takes `data: Vec<u8>` by value: the caller transfers the source buffer
//!   instead of lending a pointer. `SUNDataNode_SetData` copies it into the
//!   node's own memory exactly as C does, so no caller observes the source
//!   buffer after the call in either language.
//! * `SUNDataNode_CreateEmpty` leaves `addnamedchild`, `getnamedchild`,
//!   `removenamedchild`, `getdatanvector` and `setdatanvector` uninitialized
//!   in C (only ten of the fifteen ops are NULLed).
//!   `SUNDataNode_Ops_::default()` sets every op to `None` — C UB replaced by
//!   the deterministic value C intended (accepted deviation class 5). No path
//!   observes the difference: every implementation assigns all fifteen ops
//!   immediately after the call.
//! * Reference-build configuration: profiling is off, so every
//!   `SUNDIALS_MARK_FUNCTION_BEGIN/END` is omitted; `SUNFunctionBegin` is a
//!   no-op local assignment; `SUNAssert`/`SUNCheck` compile to nothing.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::sundatanode_inmem::{
    SUNDataNode_CreateLeaf_InMem, SUNDataNode_CreateList_InMem, SUNDataNode_CreateObject_InMem,
};
use crate::sundials_context::SUNContext;
use crate::sundials_errors::{SUN_ERR_ARG_OUTOFRANGE, SUN_ERR_NOT_IMPLEMENTED, SUN_SUCCESS};
use crate::sundials_memory::{SUNMemory, SUNMemoryHelper, SUNMemoryType};
use crate::sundials_nvector::N_Vector;
use crate::sundials_types::*;

pub type sundataindex = i64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SUNDataNodeType {
    SUNDATANODE_LEAF,
    SUNDATANODE_LIST,
    SUNDATANODE_OBJECT,
}
pub use SUNDataNodeType::*;

#[derive(Default, Clone)]
pub struct SUNDataNode_Ops_ {
    pub haschildren: Option<fn(&SUNDataNode, &mut sunbooleantype) -> SUNErrCode>,
    pub isleaf: Option<fn(&SUNDataNode, &mut sunbooleantype) -> SUNErrCode>,
    pub islist: Option<fn(&SUNDataNode, &mut sunbooleantype) -> SUNErrCode>,
    pub isobject: Option<fn(&SUNDataNode, &mut sunbooleantype) -> SUNErrCode>,
    pub addchild: Option<fn(&SUNDataNode, &SUNDataNode) -> SUNErrCode>,
    pub addnamedchild: Option<fn(&SUNDataNode, &str, &SUNDataNode) -> SUNErrCode>,
    pub getchild: Option<fn(&SUNDataNode, sundataindex, &mut Option<SUNDataNode>) -> SUNErrCode>,
    pub getnamedchild: Option<fn(&SUNDataNode, &str, &mut Option<SUNDataNode>) -> SUNErrCode>,
    pub removechild: Option<fn(&SUNDataNode, sundataindex, &mut Option<SUNDataNode>) -> SUNErrCode>,
    pub removenamedchild: Option<fn(&SUNDataNode, &str, &mut Option<SUNDataNode>) -> SUNErrCode>,
    pub getdata:
        Option<fn(&SUNDataNode, &mut Option<SUNMemory>, &mut usize, &mut usize) -> SUNErrCode>,
    pub getdatanvector: Option<fn(&SUNDataNode, &N_Vector, &mut sunrealtype) -> SUNErrCode>,
    pub setdata:
        Option<fn(&SUNDataNode, SUNMemoryType, SUNMemoryType, Vec<u8>, usize, usize) -> SUNErrCode>,
    pub setdatanvector: Option<fn(&SUNDataNode, &N_Vector, sunrealtype) -> SUNErrCode>,
    pub destroy: Option<fn(&mut Option<SUNDataNode>) -> SUNErrCode>,
}

pub type SUNDataNode_Ops = SUNDataNode_Ops_;

pub struct SUNDataNode_ {
    pub ops: RefCell<SUNDataNode_Ops_>,
    pub dtype: RefCell<SUNDataNodeType>,
    pub content: RefCell<Box<dyn Any>>,
    pub sunctx: RefCell<SUNContext>,
}

pub type SUNDataNode = Rc<SUNDataNode_>;

/// :param sunctx: The SUNContext.
/// :param node_out: Pointer to the output SUNDataNode.
/// :return: SUNErrCode indicating success or failure.
pub fn SUNDataNode_CreateEmpty(
    sunctx: &SUNContext,
    node_out: &mut Option<SUNDataNode>,
) -> SUNErrCode {
    /* C NULLs ten of the fifteen ops one by one and leaves the remaining five
    uninitialized; `default()` gives every op the intended `None`. */
    let ops = SUNDataNode_Ops_::default();

    let self_ = Rc::new(SUNDataNode_ {
        ops: RefCell::new(ops),
        /* C: `self->dtype = 0`, i.e. the first enumerator. */
        dtype: RefCell::new(SUNDATANODE_LEAF),
        content: RefCell::new(Box::new(())),
        sunctx: RefCell::new(sunctx.clone()),
    });

    *node_out = Some(self_);
    SUN_SUCCESS
}

/// :param io_mode: The I/O mode used for storing the data.
/// :param mem_helper: The memory helper.
/// :param sunctx: The SUNContext.
/// :param node_out: Pointer to the output SUNDataNode.
/// :return: SUNErrCode indicating success or failure.
pub fn SUNDataNode_CreateLeaf(
    io_mode: SUNDataIOMode,
    mem_helper: &SUNMemoryHelper,
    sunctx: &SUNContext,
    node_out: &mut Option<SUNDataNode>,
) -> SUNErrCode {
    let err = match io_mode {
        SUNDataIOMode::SUNDATAIOMODE_INMEM => {
            SUNDataNode_CreateLeaf_InMem(mem_helper, sunctx, node_out)
        }
    };

    /* C reads `err` again in `SUNCheck(err == SUN_SUCCESS, err)`, which is a
    no-op in the reference build. */
    err
}

/// :param io_mode: The I/O mode used for storing the data.
/// :param num_elements: The number of elements in the list.
/// :param sunctx: The SUNContext.
/// :param node_out: Pointer to the output SUNDataNode.
/// :return: SUNErrCode indicating success or failure.
pub fn SUNDataNode_CreateList(
    io_mode: SUNDataIOMode,
    num_elements: sundataindex,
    sunctx: &SUNContext,
    node_out: &mut Option<SUNDataNode>,
) -> SUNErrCode {
    let err = match io_mode {
        SUNDataIOMode::SUNDATAIOMODE_INMEM => {
            SUNDataNode_CreateList_InMem(num_elements, sunctx, node_out)
        }
    };

    /* C reads `err` again in `SUNCheck(err == SUN_SUCCESS, err)`, which is a
    no-op in the reference build. */
    err
}

/// :param io_mode: The I/O mode used for storing the data.
/// :param num_elements: The number of elements in the object.
/// :param sunctx: The SUNContext.
/// :param node_out: Pointer to the output SUNDataNode.
/// :return: SUNErrCode indicating success or failure.
pub fn SUNDataNode_CreateObject(
    io_mode: SUNDataIOMode,
    num_elements: sundataindex,
    sunctx: &SUNContext,
    node_out: &mut Option<SUNDataNode>,
) -> SUNErrCode {
    let err = match io_mode {
        SUNDataIOMode::SUNDATAIOMODE_INMEM => {
            SUNDataNode_CreateObject_InMem(num_elements, sunctx, node_out)
        }
    };

    /* C reads `err` again in `SUNCheck(err == SUN_SUCCESS, err)`, which is a
    no-op in the reference build. */
    err
}

/// :param self_: The SUNDataNode.
/// :param yes_or_no: Pointer to the output boolean result.
/// :return: SUNErrCode indicating success or failure.
pub fn SUNDataNode_IsLeaf(self_: &SUNDataNode, yes_or_no: &mut sunbooleantype) -> SUNErrCode {
    let f = self_.ops.borrow().isleaf;
    if let Some(f) = f {
        let err = f(self_, yes_or_no);
        return err;
    }

    SUN_ERR_NOT_IMPLEMENTED
}

/// :param self_: The SUNDataNode.
/// :param yes_or_no: Pointer to the output boolean result.
/// :return: SUNErrCode indicating success or failure.
pub fn SUNDataNode_IsList(self_: &SUNDataNode, yes_or_no: &mut sunbooleantype) -> SUNErrCode {
    let f = self_.ops.borrow().islist;
    if let Some(f) = f {
        let err = f(self_, yes_or_no);
        return err;
    }

    SUN_ERR_NOT_IMPLEMENTED
}

/// :param self_: The SUNDataNode.
/// :param yes_or_no: Pointer to the output boolean result.
/// :return: SUNErrCode indicating success or failure.
pub fn SUNDataNode_HasChildren(self_: &SUNDataNode, yes_or_no: &mut sunbooleantype) -> SUNErrCode {
    let f = self_.ops.borrow().haschildren;
    if let Some(f) = f {
        let err = f(self_, yes_or_no);
        return err;
    }

    SUN_ERR_NOT_IMPLEMENTED
}

/// :param self_: The SUNDataNode.
/// :param child_node: The child SUNDataNode to add.
/// :return: SUNErrCode indicating success or failure.
pub fn SUNDataNode_AddChild(self_: &SUNDataNode, child_node: &SUNDataNode) -> SUNErrCode {
    let f = self_.ops.borrow().addchild;
    if let Some(f) = f {
        let err = f(self_, child_node);
        return err;
    }

    SUN_ERR_NOT_IMPLEMENTED
}

/// :param self_: The SUNDataNode.
/// :param name: The name of the child.
/// :param child_node: The child SUNDataNode to add.
/// :return: SUNErrCode indicating success or failure.
pub fn SUNDataNode_AddNamedChild(
    self_: &SUNDataNode,
    name: &str,
    child_node: &SUNDataNode,
) -> SUNErrCode {
    let f = self_.ops.borrow().addnamedchild;
    if let Some(f) = f {
        let err = f(self_, name, child_node);
        return err;
    }

    SUN_ERR_NOT_IMPLEMENTED
}

/// :param self_: The SUNDataNode.
/// :param index: The index of the child.
/// :param child_node: Pointer to the output child SUNDataNode.
/// :return: SUNErrCode indicating success or failure.
pub fn SUNDataNode_GetChild(
    self_: &SUNDataNode,
    index: sundataindex,
    child_node: &mut Option<SUNDataNode>,
) -> SUNErrCode {
    let f = self_.ops.borrow().getchild;
    if let Some(f) = f {
        let err = f(self_, index, child_node);
        return err;
    }

    SUN_ERR_NOT_IMPLEMENTED
}

/// :param self_: The SUNDataNode.
/// :param name: The name of the child.
/// :param child_node: Pointer to the output child SUNDataNode.
/// :return: SUNErrCode indicating success or failure.
pub fn SUNDataNode_GetNamedChild(
    self_: &SUNDataNode,
    name: &str,
    child_node: &mut Option<SUNDataNode>,
) -> SUNErrCode {
    let f = self_.ops.borrow().getnamedchild;
    if let Some(f) = f {
        let err = f(self_, name, child_node);
        return err;
    }

    SUN_ERR_NOT_IMPLEMENTED
}

/// :param self_: The SUNDataNode.
/// :param name: The name of the child.
/// :param child_node: Pointer to the output child SUNDataNode.
/// :return: SUNErrCode indicating success or failure.
pub fn SUNDataNode_RemoveNamedChild(
    self_: &SUNDataNode,
    name: &str,
    child_node: &mut Option<SUNDataNode>,
) -> SUNErrCode {
    let f = self_.ops.borrow().removenamedchild;
    if let Some(f) = f {
        let err = f(self_, name, child_node);
        return err;
    }

    SUN_ERR_NOT_IMPLEMENTED
}

/// :param self_: The SUNDataNode.
/// :param index: The index of the child.
/// :param child_node: Pointer to the output child SUNDataNode.
/// :return: SUNErrCode indicating success or failure.
pub fn SUNDataNode_RemoveChild(
    self_: &SUNDataNode,
    index: sundataindex,
    child_node: &mut Option<SUNDataNode>,
) -> SUNErrCode {
    let f = self_.ops.borrow().removechild;
    if let Some(f) = f {
        let err = f(self_, index, child_node);
        return err;
    }

    SUN_ERR_NOT_IMPLEMENTED
}

/// :param self_: The SUNDataNode.
/// :param data: Pointer to the output data.
/// :param data_stride: Pointer to the output data stride.
/// :param data_bytes: Pointer to the output data bytes.
/// :return: SUNErrCode indicating success or failure.
pub fn SUNDataNode_GetData(
    self_: &SUNDataNode,
    data: &mut Option<SUNMemory>,
    data_stride: &mut usize,
    data_bytes: &mut usize,
) -> SUNErrCode {
    let f = self_.ops.borrow().getdata;
    if let Some(f) = f {
        let err = f(self_, data, data_stride, data_bytes);
        return err;
    }

    SUN_ERR_NOT_IMPLEMENTED
}

/// :param self_: The SUNDataNode.
/// :param v: The output state N_Vector.
/// :param t: On output, the time associated with the output state vector.
/// :return: SUNErrCode indicating success or failure.
pub fn SUNDataNode_GetDataNvector(
    self_: &SUNDataNode,
    v: &N_Vector,
    t: &mut sunrealtype,
) -> SUNErrCode {
    let f = self_.ops.borrow().getdatanvector;
    if let Some(f) = f {
        let err = f(self_, v, t);
        return err;
    }

    SUN_ERR_NOT_IMPLEMENTED
}

/// :param self_: The SUNDataNode.
/// :param src_mem_type: The source memory type.
/// :param node_mem_type: The node memory type.
/// :param data: The data to set.
/// :param data_stride: The data stride.
/// :param data_bytes: The data bytes.
/// :return: SUNErrCode indicating success or failure.
pub fn SUNDataNode_SetData(
    self_: &SUNDataNode,
    src_mem_type: SUNMemoryType,
    node_mem_type: SUNMemoryType,
    data: Vec<u8>,
    data_stride: usize,
    data_bytes: usize,
) -> SUNErrCode {
    let f = self_.ops.borrow().setdata;
    if let Some(f) = f {
        let err = f(
            self_,
            src_mem_type,
            node_mem_type,
            data,
            data_stride,
            data_bytes,
        );
        return err;
    }

    SUN_ERR_NOT_IMPLEMENTED
}

/// :param self_: The SUNDataNode.
/// :param v: The state N_Vector.
/// :param t: The time associated with the state vector.
/// :return: SUNErrCode indicating success or failure.
pub fn SUNDataNode_SetDataNvector(self_: &SUNDataNode, v: &N_Vector, t: sunrealtype) -> SUNErrCode {
    let f = self_.ops.borrow().setdatanvector;
    if let Some(f) = f {
        let err = f(self_, v, t);
        return err;
    }

    SUN_ERR_NOT_IMPLEMENTED
}

/// :param node: Pointer to the SUNDataNode to destroy.
/// :return: SUNErrCode indicating success or failure.
///
/// C dereferences `*node` unconditionally (`SUNFunctionBegin((*node)->sunctx)`);
/// a NULL `*node` is UB there and a deterministic panic here (accepted
/// deviation class 5).
pub fn SUNDataNode_Destroy(node: &mut Option<SUNDataNode>) -> SUNErrCode {
    let f = node
        .as_ref()
        .expect("SUNDataNode_Destroy: *node is NULL")
        .ops
        .borrow()
        .destroy;
    if let Some(f) = f {
        let err = f(node);
        return err;
    }

    /* Default destroy: dropping the handle releases the ops table and the
    content, and clears the caller's pointer (C `*node = NULL`). */
    *node = None;

    SUN_SUCCESS
}

/* The `default:` arm of the C `SUNDataIOMode` switch in the three Create
functions returns this code. `SUNDataIOMode` has exactly one enumerator, so
the arm is unreachable in Rust and writing it would be a warning; the constant
is pinned here to keep the C error surface documented. */
const _: SUNErrCode = SUN_ERR_ARG_OUTOFRANGE;
