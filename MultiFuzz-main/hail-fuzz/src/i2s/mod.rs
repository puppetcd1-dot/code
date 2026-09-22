pub(crate) use colorization::ColorizationStage;
pub(crate) use finder::Comparisons;
pub(crate) use replacement::{I2SRandomReplacement, I2SReplaceStage};

mod analysis;
mod colorization;
mod finder;
mod replacement;

use anyhow::Context;
use icicle_fuzzing::cmplog2::CmpLog2Ref;
use icicle_vm::Vm;

/// The maximum number of times (per-stream) the I2S stage will attempt to replace a target
/// destination value.
const MAX_ONE_BYTE_MATCHES: usize = 16;

/// The maximum number of times (per-stream) the I2S stage will attempt to use a target source
/// value.
const MAX_ONE_BYTE_REPLACEMENTS: usize = 16;

/// Dump the CmpLog data to `path`, in a canonical order.
///
/// ## Schema
///
/// One block per comparison location, and within a block: the location line
/// (`<pc>: <op>`), the operand-kind line, then one line per recorded value pair.
/// The order is **canonical** and a consumer may rely on it:
///
///   * locations are sorted by `(pc, op display)` -- iteration order of the
///     cmplog's own tables is not stable across runs;
///   * within a location, the value pairs are sorted, and each pair is written
///     with `a <= b` so that a comparison recorded from the other side compares
///     equal;
///   * the call log follows the instruction log, sorted the same way.
///
/// Artifacts written before this ordering existed are *not* normalized: two runs
/// of the same seed differ in line order and in which side of a pair was written
/// first, which is exactly what a parent/mutant diff would have tripped over.
/// Compare them by content, not by position.
///
/// The `op` line is the sleigh rendering of the comparison as the CmpLog recorded
/// it, so its two operands can still be mirrored between runs (the recorder, not
/// this writer, decides that); the (pc, normalized value pairs) are what a
/// comparison should key on.
pub fn log_cmplog_data(
    vm: &mut Vm,
    cmplog: CmpLog2Ref,
    path: &std::path::Path,
) -> anyhow::Result<()> {
    use pcode::PcodeDisplay;
    use std::io::Write;

    // @debugging: save CmpLog data
    let mut log = std::io::BufWriter::new(
        std::fs::File::create(path)
            .with_context(|| format!("failed to create `{}.txt`", path.display()))?,
    );
    // Sorted by address, then by the rendered comparison: the recorder's own
    // iteration order varies between runs, and a diff has to be about content.
    let mut locations = cmplog.get_inst_log(&mut vm.cpu).to_vec();
    locations.sort_by_key(|location| {
        // `display()` is a lazy `Display` wrapper, not a comparable value: render
        // it once so the key can be ordered.
        (location.addr, location.op.display(&vm.cpu.arch.sleigh).to_string())
    });
    for location in locations {
        writeln!(log, "{:#x}: {}", location.addr, location.op.display(&vm.cpu.arch.sleigh))?;
        let (a_kind, b_kind) = analysis::analyse_comparisons(&location);
        writeln!(log, "\t{a_kind:x?},{b_kind:x?}")?;

        // Ascending inside a pair, and the pairs themselves sorted: the same
        // comparison recorded with its operands swapped is the same comparison.
        let mut values: Vec<(i64, i64)> =
            location.values.into_iter().map(|(a, b)| if a <= b { (a, b) } else { (b, a) }).collect();
        values.sort_unstable();
        values.dedup();
        for (a, b) in values {
            writeln!(log, "\t{a:#x}, {b:#x}")?;
        }
    }
    // The call log is handed out as a slice rather than an iterator, so it is
    // sorted in place: a permutation of the recorder's own record, which every
    // consumer reads order-insensitively.
    let calls = cmplog.get_call_log(&mut vm.cpu);
    calls.sort_by_key(|location| (location.addr, location.has_invalid, location.is_indirect));
    for location in calls.iter() {
        writeln!(
            log,
            "{:#x}, has_invalid={}, is_indirect={}",
            location.addr, location.has_invalid, location.is_indirect
        )?;
        let (a_kind, b_kind) = analysis::analyse_call_parameters(location);
        writeln!(log, "\t{a_kind:x?}\n\t{b_kind:x?}")?;
        let mut values: Vec<(&[u8], &[u8])> = location
            .values
            .iter()
            .map(|(a, b)| {
                if a.as_slice() <= b.as_slice() {
                    (a.as_slice(), b.as_slice())
                }
                else {
                    (b.as_slice(), a.as_slice())
                }
            })
            .collect();
        values.sort_unstable();
        values.dedup();
        for (a, b) in values {
            writeln!(log, "\t{}, {}", a.escape_ascii(), b.escape_ascii())?;
        }
    }

    Ok(())
}
