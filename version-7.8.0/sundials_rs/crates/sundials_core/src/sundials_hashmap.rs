//! Port of `src/sundials/sundials_hashmap.c` +
//! `src/sundials/sundials_hashmap_impl.h` (char* keys, void* values →
//! generic `V`; the C `destroyKeyValue` callback maps to `Drop`).

use crate::sundials_errors::{
    SUN_ERR_ARG_CORRUPT, SUN_ERR_ARG_OUTOFRANGE, SUN_ERR_MALLOC_FAIL, SUN_SUCCESS,
};
use crate::sundials_types::*;
use crate::sunstl_vector::{SUNStlVector, SUNSTLVECTOR_GROWTH_FACTOR};

pub const SUNHASHMAP_ERROR: i64 = -99;
pub const SUNHASHMAP_KEYNOTFOUND: i64 = -1;
pub const SUNHASHMAP_DUPLICATE: i64 = -2;

pub struct SUNHashMapKeyValue_<V> {
    pub key: String,
    pub value: V,
}

pub type SUNHashMapKeyValue<V> = Box<SUNHashMapKeyValue_<V>>;

pub struct SUNHashMap<V> {
    pub buckets: SUNStlVector<Option<SUNHashMapKeyValue<V>>>,
}

const HASH_PRIME: u64 = 14695981039346656037;
const HASH_OFFSET_BASIS: u64 = 1099511628211;

/// 64-bit FNV1-a (as upstream: note the swapped prime/offset constants are
/// preserved verbatim).
fn fnv1a_hash(str_: &str) -> u64 {
    let mut hash = HASH_OFFSET_BASIS;
    for c in str_.bytes() {
        /* C `char` is signed here; non-ASCII keys never occur */
        hash = (hash ^ (c as u64)).wrapping_mul(HASH_PRIME);
    }
    hash
}

fn sunHashMapIdxFromKey<V>(map: &SUNHashMap<V>, key: &str) -> i64 {
    /* We want the index to be in [0, SUNHashMap_Capacity(map)) */
    let end = SUNHashMap_Capacity(map) - 1;
    if end == 0 {
        end
    } else {
        (fnv1a_hash(key) % (end as u64)) as i64
    }
}

pub fn SUNHashMap_New<V>(capacity: i64) -> Result<SUNHashMap<V>, SUNErrCode> {
    if capacity <= 0 {
        return Err(SUN_ERR_ARG_OUTOFRANGE);
    }
    let mut buckets =
        SUNStlVector::<Option<SUNHashMapKeyValue<V>>>::New(capacity).ok_or(SUN_ERR_MALLOC_FAIL)?;
    /* Initialize all buckets to NULL */
    for _ in 0..capacity {
        let err = buckets.PushBack(None);
        if err != SUN_SUCCESS {
            return Err(err);
        }
    }
    Ok(SUNHashMap { buckets })
}

pub fn SUNHashMap_Capacity<V>(map: &SUNHashMap<V>) -> i64 {
    map.buckets.Capacity()
}

/// C `SUNHashMap_Iterate`: walk `[start, size)`, calling `yieldfn`;
/// `SUNHASHMAP_ERROR` from the callback means "keep looking".
pub fn SUNHashMap_Iterate<V>(
    map: &SUNHashMap<V>,
    start: i64,
    yieldfn: impl Fn(i64, Option<&SUNHashMapKeyValue<V>>) -> i64,
) -> i64 {
    for i in start..map.buckets.Size() {
        let retval = yieldfn(i, map.buckets.At(i).and_then(|kv| kv.as_ref()));
        if retval == SUNHASHMAP_ERROR {
            continue; /* keep looking */
        } else {
            return retval; /* yieldfn indicates the loop should break */
        }
    }
    SUNHashMap_Capacity(map)
}

fn sunHashMapResize<V>(map: &mut SUNHashMap<V>) -> SUNErrCode {
    let old_capacity = SUNHashMap_Capacity(map);
    let new_capacity = if old_capacity == 0 {
        2
    } else {
        ((old_capacity as f64) * SUNSTLVECTOR_GROWTH_FACTOR).ceil() as i64
    };

    let mut old_buckets = std::mem::replace(
        &mut map.buckets,
        match SUNStlVector::New(new_capacity) {
            Some(v) => v,
            None => return SUN_ERR_MALLOC_FAIL,
        },
    );

    /* Set all buckets to NULL */
    for _ in 0..new_capacity {
        let err = map.buckets.PushBack(None);
        if err != SUN_SUCCESS {
            return err;
        }
    }

    /* Rehash and reinsert (from the highest old index down, as upstream) */
    for i in (0..old_capacity).rev() {
        if let Some(slot) = old_buckets.AtMut(i) {
            if let Some(kvp) = slot.take() {
                SUNHashMap_Insert(map, &kvp.key, kvp.value);
            }
        }
        let err = old_buckets.PopBack();
        if err != SUN_SUCCESS {
            return err;
        }
    }

    SUN_SUCCESS
}

/// Returns `0` on success, `SUNHASHMAP_ERROR`, or `SUNHASHMAP_DUPLICATE`.
pub fn SUNHashMap_Insert<V>(map: &mut SUNHashMap<V>, key: &str, value: V) -> i64 {
    let mut idx = sunHashMapIdxFromKey(map, key);

    /* Check if the bucket is already filled (i.e., we might have had a
    collision) */
    let occupied = map.buckets.At(idx).and_then(|kv| kv.as_ref());
    if let Some(kvp) = occupied {
        /* Determine if key is actually a duplicate (not allowed) */
        if kvp.key == key {
            return SUNHASHMAP_DUPLICATE;
        }

        /* OK, it was a real collision, so find the next open spot */
        let retval = SUNHashMap_Iterate(map, idx + 1, |i, kv| {
            if kv.is_none() {
                i /* open spot found at i */
            } else {
                SUNHASHMAP_ERROR /* keep looking */
            }
        });
        if retval == SUNHASHMAP_ERROR {
            return retval;
        } else if retval == SUNHashMap_Capacity(map) {
            /* the map is out of empty buckets, so we grow it */
            let err = sunHashMapResize(map);
            if err != SUN_SUCCESS {
                return err as i64;
            }
            return SUNHashMap_Insert(map, key, value);
        }
        idx = retval;
    }

    let kvp = Box::new(SUNHashMapKeyValue_ {
        key: key.to_string(),
        value,
    });
    SUNStlVector::Set(&mut map.buckets, idx, Some(kvp)) as i64
}

/// Locate the bucket index for `key`; C `GetValue`/`Remove` share this probe.
fn sunHashMapFindIndex<V>(map: &SUNHashMap<V>, key: &str) -> i64 {
    let idx = sunHashMapIdxFromKey(map, key);

    let kvp = map.buckets.At(idx).and_then(|kv| kv.as_ref());
    /* Check for a collision (an empty bucket means there was a collision at
    one point, but the colliding key has since been removed) */
    let collision = match kvp {
        Some(kvp) => kvp.key != key,
        None => true,
    };

    if collision {
        let retval = SUNHashMap_Iterate(map, idx + 1, |i, kv| match kv {
            None => SUNHASHMAP_ERROR, /* keep looking: bucket is empty */
            Some(kv) => {
                if kv.key == key {
                    i /* found it at i */
                } else {
                    SUNHASHMAP_ERROR /* keep looking */
                }
            }
        });
        if retval == SUNHASHMAP_ERROR {
            return SUNHASHMAP_ERROR;
        }
        retval
    } else {
        idx
    }
}

/// Returns `(0, Some(&V))` on success, else `(code, None)` with
/// `SUNHASHMAP_ERROR` or `SUNHASHMAP_KEYNOTFOUND`.
pub fn SUNHashMap_GetValue<'a, V>(map: &'a SUNHashMap<V>, key: &str) -> (i64, Option<&'a V>) {
    let idx = sunHashMapFindIndex(map, key);
    if idx == SUNHASHMAP_ERROR {
        return (SUNHASHMAP_ERROR, None);
    }
    match map.buckets.At(idx).and_then(|kv| kv.as_ref()) {
        Some(kvp) => (0, Some(&kvp.value)),
        None => (SUNHASHMAP_KEYNOTFOUND, None),
    }
}

/// Mutable access variant used by the profiler timers.
pub fn SUNHashMap_GetValueMut<'a, V>(
    map: &'a mut SUNHashMap<V>,
    key: &str,
) -> (i64, Option<&'a mut V>) {
    let idx = sunHashMapFindIndex(map, key);
    if idx == SUNHASHMAP_ERROR {
        return (SUNHASHMAP_ERROR, None);
    }
    match map.buckets.AtMut(idx).and_then(|kv| kv.as_mut()) {
        Some(kvp) => (0, Some(&mut kvp.value)),
        None => (SUNHASHMAP_KEYNOTFOUND, None),
    }
}

/// Returns `(0, Some(value))` on success.
pub fn SUNHashMap_Remove<V>(map: &mut SUNHashMap<V>, key: &str) -> (i64, Option<V>) {
    let idx = sunHashMapFindIndex(map, key);
    if idx == SUNHASHMAP_ERROR {
        return (SUNHASHMAP_ERROR, None);
    }
    match map.buckets.AtMut(idx) {
        Some(slot) => match slot.take() {
            Some(kvp) => (0, Some(kvp.value)),
            None => (SUNHASHMAP_KEYNOTFOUND, None),
        },
        None => (SUNHASHMAP_KEYNOTFOUND, None),
    }
}

/// C `SUNHashMap_Sort`: copy bucket references into a new array sorted with
/// `compar` (C `qsort` — unstable sort).
pub fn SUNHashMap_Sort<'a, V>(
    map: &'a SUNHashMap<V>,
    compar: impl Fn(
        Option<&SUNHashMapKeyValue_<V>>,
        Option<&SUNHashMapKeyValue_<V>>,
    ) -> std::cmp::Ordering,
) -> Result<Vec<Option<&'a SUNHashMapKeyValue_<V>>>, SUNErrCode> {
    let mut sorted: Vec<Option<&SUNHashMapKeyValue_<V>>> =
        Vec::with_capacity(SUNHashMap_Capacity(map) as usize);
    for i in 0..SUNHashMap_Capacity(map) {
        sorted.push(map.buckets.At(i).and_then(|kv| kv.as_deref()));
    }
    sorted.sort_unstable_by(|l, r| compar(*l, *r));
    Ok(sorted)
}

/// C `SUNHashMap_PrintKeys`.
pub fn SUNHashMap_PrintKeys<V>(map: &SUNHashMap<V>, file: &crate::sundials_utils::SUNFile) {
    file.write_str("[");
    for i in 0..SUNHashMap_Capacity(map) {
        if let Some(kvp) = map.buckets.At(i).and_then(|kv| kv.as_ref()) {
            file.write_str(&format!("{}, ", kvp.key));
        }
    }
    file.write_str("]\n");
}

const _: SUNErrCode = SUN_ERR_ARG_CORRUPT;
