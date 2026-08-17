//! Port of `src/sundials/sundials_profiler.c` +
//! `include/sundials/sundials_profiler.h` + `sundials_profiler_impl.h`
//! (serial branch; POSIX `clock_gettime(CLOCK_MONOTONIC)` maps to
//! `std::time::Instant` against a process-wide epoch).
//!
//! The reference builds have `SUNDIALS_ENABLE_PROFILING` off, so the
//! `SUNDIALS_MARK_*` macros are no-ops at solver call sites and
//! `SUNContext_Create` does not create a profiler. The module is still a
//! complete, usable port of the profiler API.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use crate::sundials_errors::*;
use crate::sundials_hashmap::*;
use crate::sundials_math::SUNMAX;
use crate::sundials_types::*;
use crate::sundials_utils::SUNFile;

pub const SUNDIALS_ROOT_TIMER: &str = "From profiler epoch";

#[derive(Clone, Copy, Default)]
pub struct sunTimespec {
    pub tv_sec: i64,
    pub tv_nsec: i64,
}

#[derive(Clone, Default)]
pub struct sunTimerStruct {
    pub tic: sunTimespec,
    pub toc: sunTimespec,
    pub average: f64,
    pub maximum: f64,
    pub elapsed: f64,
    pub count: i64,
}

pub struct SUNProfiler_ {
    pub comm: SUNComm,
    pub title: String,
    pub map: SUNHashMap<sunTimerStruct>,
    pub overhead: sunTimerStruct,
    pub sundials_time: f64,
}

pub type SUNProfiler = Rc<RefCell<SUNProfiler_>>;

fn monotonic_epoch() -> Instant {
    use std::sync::OnceLock;
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

fn sunclock_gettime_monotonic(ts: &mut sunTimespec) -> i32 {
    let d = monotonic_epoch().elapsed();
    ts.tv_sec = d.as_secs() as i64;
    ts.tv_nsec = d.subsec_nanos() as i64;
    0
}

fn sunTimerStructNew() -> sunTimerStruct {
    sunTimerStruct::default()
}

fn sunStartTiming(entry: &mut sunTimerStruct) {
    sunclock_gettime_monotonic(&mut entry.tic);
}

fn sunStopTiming(entry: &mut sunTimerStruct) {
    sunclock_gettime_monotonic(&mut entry.toc);

    let mut s_difference = entry.toc.tv_sec - entry.tic.tv_sec;
    let mut ns_difference = entry.toc.tv_nsec - entry.tic.tv_nsec;
    if ns_difference < 0 {
        s_difference -= 1;
        ns_difference = 1000000000 + entry.toc.tv_nsec - entry.tic.tv_nsec;
    }

    entry.elapsed += (s_difference as f64) + (ns_difference as f64) * 1e-9;
    entry.average = entry.elapsed;
    entry.maximum = entry.elapsed;
}

fn sunResetTiming(entry: &mut sunTimerStruct) {
    *entry = sunTimerStruct::default();
}

pub fn SUNProfiler_Create(comm: SUNComm, title: &str, p: &mut Option<SUNProfiler>) -> SUNErrCode {
    if comm != SUN_COMM_NULL {
        *p = None;
        return -1;
    }

    let mut overhead = sunTimerStructNew();
    sunStartTiming(&mut overhead);

    /* Check to see if max entries env variable was set, and use if it was. */
    let mut max_entries: i64 = 2560;
    if let Ok(v) = std::env::var("SUNPROFILER_MAX_ENTRIES") {
        max_entries = crate::sundials_utils::atol(&v);
    }
    if max_entries <= 0 {
        max_entries = 2560;
    }

    /* Create the hashmap used to store the timers */
    let map = match SUNHashMap_New::<sunTimerStruct>(max_entries) {
        Ok(m) => m,
        Err(_) => {
            *p = None;
            return SUN_ERR_MALLOC_FAIL;
        }
    };

    let profiler = Rc::new(RefCell::new(SUNProfiler_ {
        comm: SUN_COMM_NULL,
        title: title.to_string(),
        map,
        overhead,
        /* Initialize the overall timer to 0. */
        sundials_time: 0.0,
    }));

    SUNProfiler_Begin(&profiler, SUNDIALS_ROOT_TIMER);
    sunStopTiming(&mut profiler.borrow_mut().overhead);

    *p = Some(profiler);
    SUN_SUCCESS
}

pub fn SUNProfiler_Free(p: &mut Option<SUNProfiler>) -> SUNErrCode {
    if let Some(prof) = p.as_ref() {
        SUNProfiler_End(prof, SUNDIALS_ROOT_TIMER);
    }
    *p = None;
    SUN_SUCCESS
}

pub fn SUNProfiler_Begin(p: &SUNProfiler, name: &str) -> SUNErrCode {
    let mut prof = p.borrow_mut();
    let mut overhead = std::mem::take(&mut prof.overhead);
    sunStartTiming(&mut overhead);

    let mut ret = SUN_SUCCESS;
    let missing = {
        let (getval, _) = SUNHashMap_GetValueMut(&mut prof.map, name);
        getval != 0
    };
    if missing {
        let timer = sunTimerStructNew();
        let ier = SUNHashMap_Insert(&mut prof.map, name, timer);
        if ier != 0 {
            if ier == SUNHASHMAP_ERROR {
                ret = SUN_ERR_PROFILER_MAPINSERT;
            } else if ier == SUNHASHMAP_DUPLICATE {
                ret = SUN_ERR_PROFILER_MAPFULL;
            }
        }
    }

    if ret == SUN_SUCCESS {
        let (_, timer) = SUNHashMap_GetValueMut(&mut prof.map, name);
        if let Some(timer) = timer {
            timer.count += 1;
            sunStartTiming(timer);
        }
    }

    sunStopTiming(&mut overhead);
    prof.overhead = overhead;
    ret
}

pub fn SUNProfiler_End(p: &SUNProfiler, name: &str) -> SUNErrCode {
    let mut prof = p.borrow_mut();
    let mut overhead = std::mem::take(&mut prof.overhead);
    sunStartTiming(&mut overhead);

    let ier = {
        let (ier, timer) = SUNHashMap_GetValueMut(&mut prof.map, name);
        if ier == 0 {
            if let Some(timer) = timer {
                sunStopTiming(timer);
            }
        }
        ier
    };

    sunStopTiming(&mut overhead);
    prof.overhead = overhead;

    if ier == SUNHASHMAP_ERROR {
        return SUN_ERR_PROFILER_MAPGET;
    }
    if ier == SUNHASHMAP_KEYNOTFOUND {
        return SUN_ERR_PROFILER_MAPKEYNOTFOUND;
    }
    SUN_SUCCESS
}

pub fn SUNProfiler_GetTimerResolution(_p: &SUNProfiler, resolution: &mut f64) -> SUNErrCode {
    /* CLOCK_MONOTONIC resolution: Instant is nanosecond-resolution */
    *resolution = 1e-9;
    SUN_SUCCESS
}

pub fn SUNProfiler_GetElapsedTime(p: &SUNProfiler, name: &str, time: &mut f64) -> SUNErrCode {
    let prof = p.borrow();
    let (ier, timer) = SUNHashMap_GetValue(&prof.map, name);
    if ier != 0 {
        return -1;
    }
    if let Some(timer) = timer {
        *time = timer.elapsed;
    }
    SUN_SUCCESS
}

pub fn SUNProfiler_Reset(p: &SUNProfiler) -> SUNErrCode {
    {
        let mut prof = p.borrow_mut();

        /* Reset the overhead timer */
        let mut overhead = std::mem::take(&mut prof.overhead);
        sunResetTiming(&mut overhead);
        sunStartTiming(&mut overhead);
        prof.overhead = overhead;

        /* Reset all timers */
        for i in 0..SUNHashMap_Capacity(&prof.map) {
            if let Some(slot) = prof.map.buckets.AtMut(i) {
                if let Some(kvp) = slot.as_mut() {
                    sunResetTiming(&mut kvp.value);
                }
            }
        }

        /* Reset the overall timer. */
        prof.sundials_time = 0.0;
    }

    SUNProfiler_Begin(p, SUNDIALS_ROOT_TIMER);
    let mut prof = p.borrow_mut();
    let mut overhead = std::mem::take(&mut prof.overhead);
    sunStopTiming(&mut overhead);
    prof.overhead = overhead;

    SUN_SUCCESS
}

/// Print the: timer name, percentage of exec time (based on the max),
/// max across ranks, average across ranks, and the timer counter.
fn sunPrintTimer(key: &str, ts: &sunTimerStruct, fp: &SUNFile, sundials_time: f64) {
    let maximum = ts.maximum;
    let average = ts.average;
    let percent = if key != SUNDIALS_ROOT_TIMER {
        maximum / sundials_time * 100.0
    } else {
        100.0
    };
    fp.write_str(&format!(
        "{:<40}\t {:>6}% \t         {}s \t {}s \t {}\n",
        key,
        crate::sundials_utils::fmt_f(percent, 2),
        crate::sundials_utils::fmt_f(maximum, 6),
        crate::sundials_utils::fmt_f(average, 6),
        ts.count
    ));
}

pub fn SUNProfiler_Print(p: &SUNProfiler, fp: &SUNFile) -> SUNErrCode {
    let rank = 0;

    {
        let mut prof = p.borrow_mut();
        let mut overhead = std::mem::take(&mut prof.overhead);
        sunStartTiming(&mut overhead);
        prof.overhead = overhead;
    }

    /* Get the total SUNDIALS time up to this point */
    SUNProfiler_End(p, SUNDIALS_ROOT_TIMER);
    SUNProfiler_Begin(p, SUNDIALS_ROOT_TIMER);

    {
        let mut prof = p.borrow_mut();
        let (ier, timer) = SUNHashMap_GetValue(&prof.map, SUNDIALS_ROOT_TIMER);
        if ier == SUNHASHMAP_ERROR {
            return SUN_ERR_PROFILER_MAPGET;
        }
        if ier == SUNHASHMAP_KEYNOTFOUND {
            return SUN_ERR_PROFILER_MAPKEYNOTFOUND;
        }
        let elapsed = timer.expect("checked above").elapsed;
        prof.sundials_time = elapsed;
    }

    if rank == 0 {
        let prof = p.borrow();
        let mut resolution = 0.0;
        SUNProfiler_GetTimerResolution(p, &mut resolution);
        /* Sort the timers in descending order */
        let sorted = match SUNHashMap_Sort(&prof.map, |l, r| {
            match (l, r) {
                (None, None) => std::cmp::Ordering::Equal,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (Some(_), None) => std::cmp::Ordering::Less,
                (Some(l), Some(r)) => {
                    let left_max = l.value.maximum;
                    let right_max = r.value.maximum;
                    if left_max < right_max {
                        std::cmp::Ordering::Greater
                    } else if left_max > right_max {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Equal
                    }
                }
            }
        }) {
            Ok(s) => s,
            Err(_) => return SUN_ERR_PROFILER_MAPSORT,
        };
        fp.write_str(
            "\n============================================================\
             ====================================================\n",
        );
        fp.write_str(&format!(
            "SUNDIALS GIT VERSION: {}\n",
            crate::sundials_version::SUNDIALS_GIT_VERSION
        ));
        fp.write_str(&format!("SUNDIALS PROFILER: {}\n", prof.title));
        fp.write_str(&format!(
            "TIMER RESOLUTION: {}s\n",
            crate::sundials_utils::fmt_g(resolution, 6)
        ));
        fp.write_str(&format!(
            "{:<40}\t % time (inclusive) \t max/rank \t average/rank \t count \n",
            "RESULTS:"
        ));
        fp.write_str(
            "==============================================================\
             ==================================================\n",
        );

        /* Print all the other timers out */
        for kv in sorted.iter().flatten() {
            sunPrintTimer(&kv.key, &kv.value, fp, prof.sundials_time);
        }
    }

    {
        let mut prof = p.borrow_mut();
        let mut overhead = std::mem::take(&mut prof.overhead);
        sunStopTiming(&mut overhead);
        prof.overhead = overhead;
    }

    if rank == 0 {
        /* Print out the total time and the profiler overhead */
        let prof = p.borrow();
        fp.write_str(&format!(
            "{:<40}\t {:>6}% \t         {}s \t -- \t\t -- \n",
            "Est. profiler overhead",
            crate::sundials_utils::fmt_f(prof.overhead.elapsed / prof.sundials_time, 2),
            crate::sundials_utils::fmt_f(prof.overhead.elapsed, 6)
        ));

        /* End of output */
        fp.write_str("\n");
    }

    SUN_SUCCESS
}

const _: fn(f64, f64) -> f64 = SUNMAX::<f64>;
