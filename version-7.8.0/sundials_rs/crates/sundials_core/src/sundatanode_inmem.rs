//! Port of `src/sundials/sundatanode/sundatanode_inmem.c` +
//! `src/sundials/sundatanode/sundatanode_inmem.h` (the in-memory
//! `SUNDataNode` implementation).
//!
//! Mapping decisions:
//!
//! * The C accessor macros `BASE_MEMBER(node, prop)` / `GET_CONTENT(node)` /
//!   `IMPL_MEMBER(node, prop)` become the private `base_dtype`/`set_base_dtype`
//!   accessors and the house-standard `content_mut` downcast guard.
//! * `SUNStlVector_SUNDataNode` (C X-macro template over `TTYPE SUNDataNode`)
//!   becomes `SUNStlVector<Option<SUNDataNode>>`: the element `Option` is the
//!   C `SUNDataNode` pointer, whose NULL case `SUNDataNode_GetChild_InMem` and
//!   `SUNDataNode_RemoveChild_InMem` explicitly test.
//! * `SUNHashMap` with `void*` values becomes `SUNHashMap<SUNDataNode>`.
//! * The two C container destroy callbacks — `sunDataNode_FreeValue_InMem` and
//!   `sunDataNode_FreeKeyValue_InMem` — map to Rust `Drop` (the mapping fixed
//!   by the ported `sunstl_vector`/`sundials_hashmap` containers). The part of
//!   `SUNDataNode_Destroy_InMem` that `Drop` cannot express by itself, the
//!   `SUNMemoryHelper_Dealloc` of a leaf's data, is implemented as
//!   `impl Drop for SUNDataNode_InMemContent_` so that container-driven
//!   destruction updates the helper's allocation statistics exactly as C does.
//! * `content->parent` is a non-owning back-pointer in C; it maps to
//!   `Option<Weak<SUNDataNode_>>` so a parent/child pair does not form an
//!   uncollectable `Rc` cycle. It is written but never read (as in C).
//! * `content->name` aliases the caller's `const char*` in C; it is stored as
//!   an owned `Option<String>` here (write-only field, as in C).
//! * `SUNMemory::ptr` is an owned `Vec<u8>` in this workspace, so the C
//!   `sunrealtype* data_ptr = leaf_data->ptr` reinterpretation is done with
//!   `f64::from_ne_bytes`/`to_ne_bytes` (bit-exact, same as C's memcpy view).
//! * `SUNAssert`/`SUNCheckCall`/`SUNCheckCallNull`/`SUNCheckLastErr` are
//!   no-ops in this build configuration: call sites evaluate the call and
//!   continue.

use std::cell::RefMut;
use std::rc::{Rc, Weak};

use crate::sundials_context::SUNContext;
use crate::sundials_datanode::{
    sundataindex, SUNDataNode, SUNDataNodeType, SUNDataNode_, SUNDataNode_CreateEmpty,
    SUNDATANODE_LEAF, SUNDATANODE_LIST, SUNDATANODE_OBJECT,
};
use crate::sundials_errors::{
    SUN_ERR_DATANODE_NODENOTFOUND, SUN_ERR_MEM_FAIL, SUN_ERR_OP_FAIL, SUN_SUCCESS,
};
use crate::sundials_hashmap::{
    SUNHashMap, SUNHashMap_GetValue, SUNHashMap_Insert, SUNHashMap_New, SUNHashMap_Remove,
};
use crate::sundials_memory::{
    SUNMemory, SUNMemoryHelper, SUNMemoryHelper_Alloc, SUNMemoryHelper_AllocStrided,
    SUNMemoryHelper_Copy, SUNMemoryHelper_Dealloc, SUNMemoryHelper_Wrap, SUNMemoryType,
    SUNMEMTYPE_HOST,
};
use crate::sundials_nvector::{N_VBufPack, N_VBufSize, N_VBufUnpack, N_Vector};
use crate::sundials_types::*;
use crate::sunstl_vector::SUNStlVector;

/* C: #define BASE_MEMBER(node, prop) ((node)->prop) */

fn base_dtype(node: &SUNDataNode) -> SUNDataNodeType {
    *node.dtype.borrow()
}

fn set_base_dtype(node: &SUNDataNode, dtype: SUNDataNodeType) {
    *node.dtype.borrow_mut() = dtype;
}

/* -----------------------------------------------------------------
 * Implementation content
 * ----------------------------------------------------------------- */

pub struct SUNDataNode_InMemContent_ {
    /// Reference to the parent node of this node (non-owning).
    pub parent: Option<Weak<SUNDataNode_>>,

    // Node can only be an object, leaf, or list. It cannot be more than one of these at a time.

    /* Properties for Leaf nodes (nodes that store data) */
    pub mem_helper: Option<SUNMemoryHelper>,
    pub leaf_data: Option<SUNMemory>,

    /* Properties for Object nodes (nodes that are a collection of named nodes) */
    pub name: Option<String>,
    pub named_children: Option<SUNHashMap<SUNDataNode>>,
    pub num_named_children: sundataindex,

    /* Properties for a List node (nodes that are a collection of anonymous nodes) */
    pub anon_children: Option<SUNStlVector<Option<SUNDataNode>>>,
}

pub type SUNDataNode_InMemContent = SUNDataNode_InMemContent_;

impl Drop for SUNDataNode_InMemContent_ {
    /// C `sunDataNode_FreeValue_InMem`/`sunDataNode_FreeKeyValue_InMem` route
    /// container-owned children through `SUNDataNode_Destroy_InMem`, whose
    /// LEAF branch returns the leaf data to the memory helper. Nodes destroyed
    /// through the explicit `SUNDataNode_Destroy_InMem` path take `leaf_data`
    /// out first, so this never deallocates twice.
    fn drop(&mut self) {
        if let Some(leaf_data) = self.leaf_data.take() {
            if let Some(mem_helper) = self.mem_helper.clone() {
                /* Use the default queue for the memory helper */
                let _ = SUNMemoryHelper_Dealloc(&mem_helper, Some(leaf_data), None);
            }
        }
    }
}

/// C `GET_CONTENT(node)` / `IMPL_MEMBER(node, prop)`.
fn content_mut(node: &SUNDataNode) -> RefMut<'_, SUNDataNode_InMemContent_> {
    RefMut::map(node.content.borrow_mut(), |c| {
        c.downcast_mut::<SUNDataNode_InMemContent_>()
            .expect("InMem SUNDataNode content")
    })
}

/* -----------------------------------------------------------------
 * `SUNMemory` byte buffer viewed as a `sunrealtype` array
 * ----------------------------------------------------------------- */

/// C `sunrealtype* data_ptr = leaf_data->ptr` (read side).
fn sunDataNodeBytesAsReal(bytes: &[u8]) -> Vec<sunrealtype> {
    bytes
        .chunks_exact(std::mem::size_of::<sunrealtype>())
        .map(|c| sunrealtype::from_ne_bytes(c.try_into().expect("sunrealtype chunk")))
        .collect()
}

/// C `sunrealtype* data_ptr = leaf_data->ptr` (write side).
fn sunDataNodeRealAsBytes(vals: &[sunrealtype], bytes: &mut [u8]) {
    let sz = std::mem::size_of::<sunrealtype>();
    for (i, val) in vals.iter().enumerate() {
        let off = i * sz;
        if off + sz > bytes.len() {
            break;
        }
        bytes[off..off + sz].copy_from_slice(&val.to_ne_bytes());
    }
}

/* -----------------------------------------------------------------
 * Private constructor/destructor common to all node types
 * ----------------------------------------------------------------- */

fn sunDataNode_CreateCommon_InMem(sunctx: &SUNContext) -> Option<SUNDataNode> {
    let mut node: Option<SUNDataNode> = None;
    let _ = SUNDataNode_CreateEmpty(sunctx, &mut node);
    let node = node?;

    {
        let mut ops = node.ops.borrow_mut();
        ops.haschildren = Some(SUNDataNode_HasChildren_InMem);
        ops.isleaf = Some(SUNDataNode_IsLeaf_InMem);
        ops.islist = Some(SUNDataNode_IsList_InMem);
        ops.isobject = Some(SUNDataNode_IsObject_InMem);
        ops.addchild = Some(SUNDataNode_AddChild_InMem);
        ops.addnamedchild = Some(SUNDataNode_AddNamedChild_InMem);
        ops.getchild = Some(SUNDataNode_GetChild_InMem);
        ops.getnamedchild = Some(SUNDataNode_GetNamedChild_InMem);
        ops.removechild = Some(SUNDataNode_RemoveChild_InMem);
        ops.removenamedchild = Some(SUNDataNode_RemoveNamedChild_InMem);
        ops.getdata = Some(SUNDataNode_GetData_InMem);
        ops.getdatanvector = Some(SUNDataNode_GetDataNvector_InMem);
        ops.setdata = Some(SUNDataNode_SetData_InMem);
        ops.setdatanvector = Some(SUNDataNode_SetDataNvector_InMem);
        ops.destroy = Some(SUNDataNode_Destroy_InMem);
    }

    *node.content.borrow_mut() = Box::new(SUNDataNode_InMemContent_ {
        parent: None,
        mem_helper: None,
        leaf_data: None,
        name: None,
        named_children: None,
        num_named_children: 0,
        anon_children: None,
    });

    Some(node)
}

fn sunDataNode_DestroyCommon_InMem(node: &mut Option<SUNDataNode>) {
    let n = match node.as_ref() {
        None => return,
        Some(n) => n,
    };
    /* C: free(BASE_MEMBER(*node, content)); BASE_MEMBER(*node, content) = NULL; */
    let content = std::mem::replace(&mut *n.content.borrow_mut(), Box::new(()));
    drop(content);
    /* C: free(BASE_MEMBER(*node, ops)); free(*node); *node = NULL;
    — both live inside the handle and are released with the last Rc. */
    *node = None;
}

/* -----------------------------------------------------------------
 * Exported functions
 * ----------------------------------------------------------------- */

pub fn SUNDataNode_CreateList_InMem(
    init_size: sundataindex,
    sunctx: &SUNContext,
    node_out: &mut Option<SUNDataNode>,
) -> SUNErrCode {
    let node = sunDataNode_CreateCommon_InMem(sunctx).expect("sunDataNode_CreateCommon_InMem");

    set_base_dtype(&node, SUNDATANODE_LIST);
    let anon_children = SUNStlVector::<Option<SUNDataNode>>::New(init_size);
    if anon_children.is_none() {
        let mut node = Some(node);
        sunDataNode_DestroyCommon_InMem(&mut node);
        return SUN_ERR_MEM_FAIL;
    }
    content_mut(&node).anon_children = anon_children;

    *node_out = Some(node);
    SUN_SUCCESS
}

pub fn SUNDataNode_CreateObject_InMem(
    init_size: sundataindex,
    sunctx: &SUNContext,
    node_out: &mut Option<SUNDataNode>,
) -> SUNErrCode {
    let node = sunDataNode_CreateCommon_InMem(sunctx).expect("sunDataNode_CreateCommon_InMem");

    set_base_dtype(&node, SUNDATANODE_OBJECT);

    /* C SUNCheckCall does not early-return in this build; a failed
    SUNHashMap_New leaves `map` unset (uninitialized in C, None here). */
    let map = SUNHashMap_New::<SUNDataNode>(init_size).ok();

    content_mut(&node).named_children = map;

    *node_out = Some(node);
    SUN_SUCCESS
}

pub fn SUNDataNode_CreateLeaf_InMem(
    mem_helper: &SUNMemoryHelper,
    sunctx: &SUNContext,
    node_out: &mut Option<SUNDataNode>,
) -> SUNErrCode {
    let node = sunDataNode_CreateCommon_InMem(sunctx).expect("sunDataNode_CreateCommon_InMem");

    set_base_dtype(&node, SUNDATANODE_LEAF);
    {
        let mut content = content_mut(&node);
        content.mem_helper = Some(mem_helper.clone());
        content.leaf_data = None;
    }

    *node_out = Some(node);
    SUN_SUCCESS
}

pub fn SUNDataNode_IsLeaf_InMem(self_: &SUNDataNode, yes_or_no: &mut sunbooleantype) -> SUNErrCode {
    *yes_or_no = base_dtype(self_) == SUNDATANODE_LEAF;
    SUN_SUCCESS
}

pub fn SUNDataNode_IsList_InMem(self_: &SUNDataNode, yes_or_no: &mut sunbooleantype) -> SUNErrCode {
    *yes_or_no = base_dtype(self_) == SUNDATANODE_LIST;
    SUN_SUCCESS
}

pub fn SUNDataNode_IsObject_InMem(
    self_: &SUNDataNode,
    yes_or_no: &mut sunbooleantype,
) -> SUNErrCode {
    *yes_or_no = base_dtype(self_) == SUNDATANODE_OBJECT;
    SUN_SUCCESS
}

pub fn SUNDataNode_HasChildren_InMem(
    self_: &SUNDataNode,
    yes_or_no: &mut sunbooleantype,
) -> SUNErrCode {
    let content = content_mut(self_);
    *yes_or_no = (match content.anon_children.as_ref() {
        Some(anon_children) => anon_children.Size() != 0,
        None => SUNFALSE,
    }) || content.num_named_children != 0;
    SUN_SUCCESS
}

pub fn SUNDataNode_AddChild_InMem(self_: &SUNDataNode, child_node: &SUNDataNode) -> SUNErrCode {
    /* SUNAssert(BASE_MEMBER(self, dtype) == SUNDATANODE_LIST, SUN_ERR_ARG_WRONGTYPE) */

    {
        let mut content = content_mut(self_);
        let anon_children = content.anon_children.as_mut().expect("anon_children");
        let _ = anon_children.PushBack(Some(child_node.clone()));
    }
    content_mut(child_node).parent = Some(Rc::downgrade(self_));
    SUN_SUCCESS
}

pub fn SUNDataNode_AddNamedChild_InMem(
    self_: &SUNDataNode,
    name: &str,
    child_node: &SUNDataNode,
) -> SUNErrCode {
    /* SUNAssert(BASE_MEMBER(self, dtype) == SUNDATANODE_OBJECT,
    SUN_ERR_ARG_WRONGTYPE) */

    content_mut(child_node).name = Some(name.to_string());
    {
        let mut content = content_mut(self_);
        let named_children = content.named_children.as_mut().expect("named_children");
        if SUNHashMap_Insert(named_children, name, child_node.clone()) != 0 {
            return SUN_ERR_OP_FAIL;
        }
    }

    content_mut(child_node).parent = Some(Rc::downgrade(self_));
    content_mut(self_).num_named_children += 1;

    SUN_SUCCESS
}

pub fn SUNDataNode_GetChild_InMem(
    self_: &SUNDataNode,
    index: sundataindex,
    child_node: &mut Option<SUNDataNode>,
) -> SUNErrCode {
    let mut has_children = SUNFALSE;
    let _ = SUNDataNode_HasChildren_InMem(self_, &mut has_children);

    if !has_children {
        return SUN_ERR_DATANODE_NODENOTFOUND;
    }

    let child_node_ptr = {
        let content = content_mut(self_);
        let anon_children = content.anon_children.as_ref().expect("anon_children");
        anon_children.At(index).cloned()
    };

    if let Some(child_node_ptr) = child_node_ptr {
        *child_node = child_node_ptr;
        SUN_SUCCESS
    } else {
        SUN_ERR_DATANODE_NODENOTFOUND
    }
}

pub fn SUNDataNode_GetNamedChild_InMem(
    self_: &SUNDataNode,
    name: &str,
    child_node: &mut Option<SUNDataNode>,
) -> SUNErrCode {
    *child_node = None;

    let mut has_children = SUNFALSE;
    let _ = SUNDataNode_HasChildren_InMem(self_, &mut has_children);

    if has_children {
        let found = {
            let content = content_mut(self_);
            let named_children = content.named_children.as_ref().expect("named_children");
            let (retval, value) = SUNHashMap_GetValue(named_children, name);
            if retval != 0 {
                None
            } else {
                Some(value.expect("hash map value").clone())
            }
        };
        match found {
            None => SUN_ERR_DATANODE_NODENOTFOUND,
            Some(found) => {
                *child_node = Some(found);
                SUN_SUCCESS
            }
        }
    } else {
        SUN_ERR_DATANODE_NODENOTFOUND
    }
}

pub fn SUNDataNode_RemoveChild_InMem(
    self_: &SUNDataNode,
    index: sundataindex,
    child_node: &mut Option<SUNDataNode>,
) -> SUNErrCode {
    let mut has_children = SUNFALSE;
    let _ = SUNDataNode_HasChildren_InMem(self_, &mut has_children);

    if !has_children {
        *child_node = None;
        return SUN_SUCCESS;
    }

    let child_node_ptr = {
        let content = content_mut(self_);
        let anon_children = content.anon_children.as_ref().expect("anon_children");
        anon_children.At(index).cloned()
    };

    if let Some(child_node_ptr) = child_node_ptr {
        *child_node = child_node_ptr;
        if let Some(child) = child_node.as_ref() {
            content_mut(child).parent = None;
            let mut content = content_mut(self_);
            let anon_children = content.anon_children.as_mut().expect("anon_children");
            let _ = anon_children.Erase(index);
        } else {
            return SUN_ERR_DATANODE_NODENOTFOUND;
        }
    } else {
        return SUN_ERR_DATANODE_NODENOTFOUND;
    }

    SUN_SUCCESS
}

pub fn SUNDataNode_RemoveNamedChild_InMem(
    self_: &SUNDataNode,
    name: &str,
    child_node: &mut Option<SUNDataNode>,
) -> SUNErrCode {
    *child_node = None;

    let mut has_children = SUNFALSE;
    let _ = SUNDataNode_HasChildren_InMem(self_, &mut has_children);

    if has_children {
        let (retval, value) = {
            let mut content = content_mut(self_);
            let named_children = content.named_children.as_mut().expect("named_children");
            SUNHashMap_Remove(named_children, name)
        };
        if retval != 0 {
            *child_node = None;
            return SUN_ERR_DATANODE_NODENOTFOUND;
        }
        *child_node = value;
        content_mut(child_node.as_ref().expect("child_node")).parent = None;
        content_mut(self_).num_named_children -= 1;
    }

    SUN_SUCCESS
}

pub fn SUNDataNode_GetData_InMem(
    self_: &SUNDataNode,
    data: &mut Option<SUNMemory>,
    data_stride: &mut usize,
    data_bytes: &mut usize,
) -> SUNErrCode {
    let leaf_data = content_mut(self_).leaf_data.clone().expect("leaf_data");

    {
        let leaf_data = leaf_data.borrow();
        *data_stride = leaf_data.stride;
        *data_bytes = leaf_data.bytes;
    }
    *data = Some(leaf_data);

    SUN_SUCCESS
}

pub fn SUNDataNode_GetDataNvector_InMem(
    self_: &SUNDataNode,
    v: &N_Vector,
    t: &mut sunrealtype,
) -> SUNErrCode {
    /* Use the default queue for the memory helper: `queue = NULL` */

    let (mem_helper, leaf_data) = {
        let content = content_mut(self_);
        (
            content.mem_helper.clone(),
            content.leaf_data.clone().expect("leaf_data"),
        )
    };

    let leaf_mem_type = leaf_data.borrow().type_;

    let mut buffer_size: sunindextype = 0;
    let _ = N_VBufSize(v, &mut buffer_size);
    /* SUNAssert((buffer_size + sizeof(sunrealtype)) == leaf_data->bytes,
    SUN_ERR_ARG_INCOMPATIBLE) */

    if leaf_mem_type != SUNMEMTYPE_HOST {
        /* BufUnpack assumes the data is on the host. So if the leaf has it elsewhere,
        we need to move it to the host first. */
        let mem_helper = mem_helper.expect("mem_helper");
        let leaf_bytes = leaf_data.borrow().bytes;

        let mut leaf_host_data: Option<SUNMemory> = None;
        let _ = SUNMemoryHelper_Alloc(
            &mem_helper,
            &mut leaf_host_data,
            leaf_bytes,
            SUNMEMTYPE_HOST,
            None,
        );

        let _ = SUNMemoryHelper_Copy(
            &mem_helper,
            leaf_host_data.as_ref().expect("leaf_host_data"),
            &leaf_data,
            buffer_size as usize,
            None,
        );

        let data_ptr = {
            let leaf_host_data = leaf_host_data.as_ref().expect("leaf_host_data").borrow();
            sunDataNodeBytesAsReal(&leaf_host_data.data)
        };
        *t = data_ptr[0];
        let _ = N_VBufUnpack(v, &data_ptr[1..]);

        let _ = SUNMemoryHelper_Dealloc(&mem_helper, leaf_host_data, None);
    } else {
        let data_ptr = {
            let leaf_data = leaf_data.borrow();
            sunDataNodeBytesAsReal(&leaf_data.data)
        };
        *t = data_ptr[0];
        let _ = N_VBufUnpack(v, &data_ptr[1..]);
    }

    SUN_SUCCESS
}

pub fn SUNDataNode_SetData_InMem(
    self_: &SUNDataNode,
    src_mem_type: SUNMemoryType,
    node_mem_type: SUNMemoryType,
    data: Vec<u8>,
    data_stride: usize,
    data_bytes: usize,
) -> SUNErrCode {
    /* Use the default queue for the memory helper: `queue = NULL` */

    /* SUNAssert(BASE_MEMBER(self, dtype) == SUNDATANODE_LEAF,
    SUN_ERR_ARG_WRONGTYPE) */

    let mem_helper = content_mut(self_).mem_helper.clone().expect("mem_helper");

    let data_mem_src =
        SUNMemoryHelper_Wrap(&mem_helper, data, src_mem_type).expect("SUNMemoryHelper_Wrap");

    let mut data_mem_dst: Option<SUNMemory> = None;
    let _ = SUNMemoryHelper_AllocStrided(
        &mem_helper,
        &mut data_mem_dst,
        data_bytes,
        data_stride,
        node_mem_type,
        None,
    );

    let _ = SUNMemoryHelper_Copy(
        &mem_helper,
        data_mem_dst.as_ref().expect("data_mem_dst"),
        &data_mem_src,
        data_bytes,
        None,
    );

    let _ = SUNMemoryHelper_Dealloc(&mem_helper, Some(data_mem_src), None);

    content_mut(self_).leaf_data = data_mem_dst;

    SUN_SUCCESS
}

pub fn SUNDataNode_SetDataNvector_InMem(
    self_: &SUNDataNode,
    v: &N_Vector,
    t: sunrealtype,
) -> SUNErrCode {
    /* Use the default queue for the memory helper: `queue = NULL` */

    let leaf_mem_type = SUNMEMTYPE_HOST;

    let mut buffer_size: sunindextype = 0;
    let _ = N_VBufSize(v, &mut buffer_size);

    let mem_helper = content_mut(self_).mem_helper.clone().expect("mem_helper");

    /* We allocate 1 extra sunrealtype for storing t */
    let mut leaf_data: Option<SUNMemory> = None;
    let _ = SUNMemoryHelper_AllocStrided(
        &mem_helper,
        &mut leaf_data,
        buffer_size as usize + std::mem::size_of::<sunrealtype>(),
        std::mem::size_of::<sunrealtype>(),
        leaf_mem_type,
        None,
    );
    let leaf_data = leaf_data.expect("leaf_data");

    /* BufPack will handle any necessary copies from the device and will fill data_ptr on the host */
    let mut data_ptr: Vec<sunrealtype> =
        vec![0.0; buffer_size as usize / std::mem::size_of::<sunrealtype>() + 1];
    data_ptr[0] = t;
    let _ = N_VBufPack(v, &mut data_ptr[1..]);
    sunDataNodeRealAsBytes(&data_ptr, &mut leaf_data.borrow_mut().data);

    content_mut(self_).leaf_data = Some(leaf_data);

    SUN_SUCCESS
}

pub fn SUNDataNode_Destroy_InMem(node: &mut Option<SUNDataNode>) -> SUNErrCode {
    /* Use the default queue for the memory helper: `queue = NULL` */

    {
        let self_ = node.as_ref().expect("node");

        if base_dtype(self_) == SUNDATANODE_OBJECT {
            let map = content_mut(self_).named_children.take();
            drop(map);
        } else if base_dtype(self_) == SUNDATANODE_LIST {
            let anon_children = content_mut(self_).anon_children.take();
            drop(anon_children);
        } else if base_dtype(self_) == SUNDATANODE_LEAF {
            let (mem_helper, leaf_data) = {
                let mut content = content_mut(self_);
                (content.mem_helper.clone(), content.leaf_data.take())
            };
            if leaf_data.is_some() {
                let _ = SUNMemoryHelper_Dealloc(&mem_helper.expect("mem_helper"), leaf_data, None);
            }
        }
    }

    sunDataNode_DestroyCommon_InMem(node);
    *node = None;

    SUN_SUCCESS
}
