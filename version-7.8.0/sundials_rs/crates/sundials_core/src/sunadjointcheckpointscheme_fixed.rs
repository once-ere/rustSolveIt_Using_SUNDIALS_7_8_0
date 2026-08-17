//! Port of `src/sunadjointcheckpointscheme/fixed/sunadjointcheckpointscheme_fixed.c` +
//! `include/sunadjointcheckpointscheme/sunadjointcheckpointscheme_fixed.h`
//! (SUNAdjointCheckpointScheme_Fixed class definition).
//!
//! The C `GET_CONTENT(S)`/`IMPL_MEMBER(S, prop)` accessor macros become
//! `content_mut(S).prop` field reads/writes through the handle's `RefCell`.
//! Every content borrow is scoped tightly around a single field access so no
//! borrow is ever held across a `SUNDataNode` call.
//!
//! C `char* key = sunSignedToString(...)` / `free(key)` map to an owned
//! `String` and an explicit `drop`. The `SUNLogExtraDebug` sites compile away
//! entirely at `SUNDIALS_LOGGING_LEVEL = 2` and are omitted here;
//! `SUNCheckCall`/`SUNAssert` are release no-ops, so their call sites just
//! evaluate the call and continue.

use std::cell::RefMut;

use crate::sundials_adjointcheckpointscheme::{
    SUNAdjointCheckpointScheme, SUNAdjointCheckpointScheme_NewEmpty,
};
use crate::sundials_context::SUNContext;
use crate::sundials_datanode::{
    SUNDataNode, SUNDataNode_AddChild, SUNDataNode_AddNamedChild, SUNDataNode_CreateLeaf,
    SUNDataNode_CreateList, SUNDataNode_CreateObject, SUNDataNode_Destroy, SUNDataNode_GetChild,
    SUNDataNode_GetDataNvector, SUNDataNode_GetNamedChild, SUNDataNode_HasChildren,
    SUNDataNode_RemoveChild, SUNDataNode_RemoveNamedChild, SUNDataNode_SetDataNvector,
};
use crate::sundials_errors::{
    SUN_ERR_CHECKPOINT_NOT_FOUND, SUN_ERR_DATANODE_NODENOTFOUND, SUN_SUCCESS,
};
use crate::sundials_memory::SUNMemoryHelper;
use crate::sundials_nvector::N_Vector;
use crate::sundials_types::*;
use crate::sundials_utils::sunSignedToString;

/* ----------------------------------------------------------------
 * Fixed-interval implementation of SUNAdjointCheckpointScheme
 * ---------------------------------------------------------------- */

/// C `struct SUNAdjointCheckpointScheme_Fixed_Content_`.
///
/// The three `SUNDataNode` members are `Option` (C NULL = `None`); the node
/// handles kept in `current_insert_step_node`/`current_load_step_node` are
/// `Rc` clones of nodes owned by `root_node`, mirroring C's raw aliases.
pub struct SUNAdjointCheckpointScheme_Fixed_Content_ {
    pub backup_interval: suncountertype,
    pub interval: suncountertype,
    pub step_num_of_current_insert: suncountertype,
    pub step_num_of_current_load: suncountertype,
    pub mem_helper: SUNMemoryHelper,
    pub root_node: Option<SUNDataNode>,
    pub current_insert_step_node: Option<SUNDataNode>,
    pub current_load_step_node: Option<SUNDataNode>,
    pub io_mode: SUNDataIOMode,
    pub keep: sunbooleantype,
}

pub type SUNAdjointCheckpointScheme_Fixed_Content = SUNAdjointCheckpointScheme_Fixed_Content_;

/// C `GET_CONTENT(S)` — downcast of the handle's `void* content`.
fn content_mut(
    S: &SUNAdjointCheckpointScheme,
) -> RefMut<'_, SUNAdjointCheckpointScheme_Fixed_Content_> {
    RefMut::map(S.content.borrow_mut(), |c| {
        c.downcast_mut::<SUNAdjointCheckpointScheme_Fixed_Content_>()
            .expect("Fixed SUNAdjointCheckpointScheme content")
    })
}

pub fn SUNAdjointCheckpointScheme_Create_Fixed(
    io_mode: SUNDataIOMode,
    mem_helper: &SUNMemoryHelper,
    interval: suncountertype,
    estimate: suncountertype,
    keep: sunbooleantype,
    sunctx: &SUNContext,
    check_scheme_ptr: &mut Option<SUNAdjointCheckpointScheme>,
) -> SUNErrCode {
    let mut check_scheme: Option<SUNAdjointCheckpointScheme> = None;
    let _ = SUNAdjointCheckpointScheme_NewEmpty(sunctx, &mut check_scheme);
    /* C dereferences the (possibly NULL) handle immediately below */
    let check_scheme = check_scheme.expect("SUNAdjointCheckpointScheme_NewEmpty returned NULL");

    {
        let mut ops = check_scheme.ops.borrow_mut();
        ops.needssaving = Some(SUNAdjointCheckpointScheme_NeedsSaving_Fixed);
        ops.insertvector = Some(SUNAdjointCheckpointScheme_InsertVector_Fixed);
        ops.loadvector = Some(SUNAdjointCheckpointScheme_LoadVector_Fixed);
        ops.enableDense = Some(SUNAdjointCheckpointScheme_EnableDense_Fixed);
        ops.destroy = Some(SUNAdjointCheckpointScheme_Destroy_Fixed);
    }

    let mut content = SUNAdjointCheckpointScheme_Fixed_Content_ {
        mem_helper: mem_helper.clone(),
        interval,
        keep,
        root_node: None,
        current_insert_step_node: None,
        step_num_of_current_insert: -2,
        current_load_step_node: None,
        step_num_of_current_load: -2,
        io_mode,
        /* C leaves `backup_interval` uninitialized here; it is written by
        every SUNAdjointCheckpointScheme_EnableDense(scheme, SUNTRUE) call
        before it can be read back by an EnableDense(scheme, SUNFALSE).
        Rust has no indeterminate value, so seed it with `interval`, which
        makes an unpaired EnableDense(SUNFALSE) a no-op. */
        backup_interval: interval,
    };

    let _ = SUNDataNode_CreateObject(io_mode, estimate, sunctx, &mut content.root_node);

    *check_scheme.content.borrow_mut() = Box::new(content);
    *check_scheme_ptr = Some(check_scheme);

    SUN_SUCCESS
}

pub fn SUNAdjointCheckpointScheme_NeedsSaving_Fixed(
    self_: &SUNAdjointCheckpointScheme,
    step_num: suncountertype,
    _stage_num: suncountertype,
    _t: sunrealtype,
    yes_or_no: &mut sunbooleantype,
) -> SUNErrCode {
    let interval = content_mut(self_).interval;

    if step_num % interval == 0 {
        *yes_or_no = SUNTRUE;
    } else {
        *yes_or_no = SUNFALSE;
    }

    SUN_SUCCESS
}

pub fn SUNAdjointCheckpointScheme_InsertVector_Fixed(
    self_: &SUNAdjointCheckpointScheme,
    step_num: suncountertype,
    _stage_num: suncountertype,
    t: sunrealtype,
    y: &N_Vector,
) -> SUNErrCode {
    /* C `SUNCTX_` — the SUNFunctionBegin local for `self->sunctx` */
    let sunctx = self_.sunctx.borrow().clone();

    /* If this is the first state for a step, then we need to create a
    list node first to store the step and all stage solutions in.
    We keep a pointer to the list node until this step is over for
    fast access when inserting stages. */
    let step_data_node: SUNDataNode;
    let (io_mode, step_num_of_current_insert) = {
        let content = content_mut(self_);
        (content.io_mode, content.step_num_of_current_insert)
    };
    if step_num != step_num_of_current_insert {
        let mut new_step_node: Option<SUNDataNode> = None;
        let _ = SUNDataNode_CreateList(io_mode, 0, &sunctx, &mut new_step_node);
        step_data_node = new_step_node.expect("SUNDataNode_CreateList returned NULL");
        {
            let mut content = content_mut(self_);
            content.current_insert_step_node = Some(step_data_node.clone());
            content.step_num_of_current_insert = step_num;
        }

        /* Store the step node in the root node object. */
        let key = sunSignedToString(step_num);
        let root_node = content_mut(self_)
            .root_node
            .clone()
            .expect("SUNAdjointCheckpointScheme_Fixed root node");
        let _ = SUNDataNode_AddNamedChild(&root_node, &key, &step_data_node);
        drop(key); /* C: free(key) */
    } else {
        step_data_node = content_mut(self_)
            .current_insert_step_node
            .clone()
            .expect("SUNAdjointCheckpointScheme_Fixed current insert step node");
    }

    /* Add the state data as a leaf node in the step node's list of children. */
    let mut solution_node: Option<SUNDataNode> = None;
    let (io_mode, mem_helper) = {
        let content = content_mut(self_);
        (content.io_mode, content.mem_helper.clone())
    };
    let _ = SUNDataNode_CreateLeaf(io_mode, &mem_helper, &sunctx, &mut solution_node);
    let solution_node = solution_node.expect("SUNDataNode_CreateLeaf returned NULL");
    let _ = SUNDataNode_SetDataNvector(&solution_node, y, t);

    let _ = SUNDataNode_AddChild(&step_data_node, &solution_node);

    SUN_SUCCESS
}

pub fn SUNAdjointCheckpointScheme_LoadVector_Fixed(
    self_: &SUNAdjointCheckpointScheme,
    step_num: suncountertype,
    stage_num: suncountertype,
    peek: sunbooleantype,
    yout: &mut N_Vector,
    tout: &mut sunrealtype,
) -> SUNErrCode {
    /* C: `SUNErrCode errcode = SUN_SUCCESS;` — that initial value is dead on
    every path (always overwritten before it is read), so it is left
    unassigned here to keep the build warning-free. */
    let mut errcode: SUNErrCode;

    /* If we are trying to load the step solution, we need to load the list which holds
    the step and stage solutions. We keep a pointer to the list node until
    this step is over for fast access when loading stages. */
    let mut step_data_node: Option<SUNDataNode> = None;
    let step_num_of_current_load = content_mut(self_).step_num_of_current_load;
    if step_num != step_num_of_current_load {
        let key = sunSignedToString(step_num);
        let root_node = content_mut(self_)
            .root_node
            .clone()
            .expect("SUNAdjointCheckpointScheme_Fixed root node");
        errcode = SUNDataNode_GetNamedChild(&root_node, &key, &mut step_data_node);
        drop(key); /* C: free(key) */
        if errcode == SUN_SUCCESS {
            let mut content = content_mut(self_);
            content.current_load_step_node = step_data_node.clone();
            content.step_num_of_current_load = step_num;
        } else if errcode == SUN_ERR_DATANODE_NODENOTFOUND {
            step_data_node = None;
        } else {
            /* C: SUNCheckCall(errcode) — a no-op in release builds */
        }
    } else {
        step_data_node = content_mut(self_).current_load_step_node.clone();
    }

    let step_data_node = match step_data_node {
        Some(node) => node,
        None => return SUN_ERR_CHECKPOINT_NOT_FOUND,
    };

    let mut solution_node: Option<SUNDataNode> = None;
    let keep = content_mut(self_).keep;
    if keep || peek {
        errcode = SUNDataNode_GetChild(&step_data_node, stage_num, &mut solution_node);
        if errcode == SUN_ERR_DATANODE_NODENOTFOUND {
            solution_node = None;
        } else {
            /* C: SUNCheckCall(errcode) */
        }
    } else {
        let mut has_children = SUNFALSE;
        let _ = SUNDataNode_HasChildren(&step_data_node, &mut has_children);

        if has_children {
            errcode = SUNDataNode_RemoveChild(&step_data_node, stage_num, &mut solution_node);
            if errcode == SUN_ERR_DATANODE_NODENOTFOUND {
                solution_node = None;
            } else {
                /* C: SUNCheckCall(errcode) */
            }
        }

        /* If we just removed the last stage (so has_children==false),
        then we should remove the step too. */
        let _ = SUNDataNode_HasChildren(&step_data_node, &mut has_children);
        if !has_children {
            let key = sunSignedToString(step_num);
            let root_node = content_mut(self_)
                .root_node
                .clone()
                .expect("SUNAdjointCheckpointScheme_Fixed root node");
            /* C reassigns its `step_data_node` local through this out-param
            (the local is dead afterwards) */
            let mut removed_step_node = Some(step_data_node.clone());
            let _ = SUNDataNode_RemoveNamedChild(&root_node, &key, &mut removed_step_node);
            drop(key); /* C: free(key) */
            let _ = SUNDataNode_Destroy(&mut removed_step_node);
        }
    }

    let solution_node = match solution_node {
        Some(node) => node,
        None => return SUN_ERR_CHECKPOINT_NOT_FOUND,
    };

    let _ = SUNDataNode_GetDataNvector(&solution_node, yout, tout);

    /* Cleanup the checkpoint memory if need be */
    let keep = content_mut(self_).keep;
    if !(keep || peek) {
        let mut solution_node = Some(solution_node);
        let _ = SUNDataNode_Destroy(&mut solution_node);
    }

    SUN_SUCCESS
}

pub fn SUNAdjointCheckpointScheme_Destroy_Fixed(
    self_ptr: &mut Option<SUNAdjointCheckpointScheme>,
) -> SUNErrCode {
    /* C dereferences `(*self_ptr)->sunctx` in SUNFunctionBegin */
    let self_ = self_ptr
        .as_ref()
        .expect("SUNAdjointCheckpointScheme_Destroy_Fixed: NULL checkpoint scheme")
        .clone();

    /* C: SUNCheckCall(SUNDataNode_Destroy(&IMPL_MEMBER(self, root_node))); */
    let mut root_node = content_mut(&self_).root_node.take();
    let _ = SUNDataNode_Destroy(&mut root_node);
    content_mut(&self_).root_node = root_node;

    /* C: free(self->content); free(self->ops); free(self); */
    drop(self_);
    *self_ptr = None;

    SUN_SUCCESS
}

pub fn SUNAdjointCheckpointScheme_EnableDense_Fixed(
    check_scheme: &SUNAdjointCheckpointScheme,
    on_or_off: sunbooleantype,
) -> SUNErrCode {
    let mut guard = content_mut(check_scheme);
    let content = &mut *guard;

    if on_or_off {
        content.backup_interval = content.interval;
        content.interval = 1;
    } else {
        content.interval = content.backup_interval;
    }

    SUN_SUCCESS
}
