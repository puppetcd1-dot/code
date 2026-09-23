use std::{
    collections::BTreeMap,
    io::{Read, Write},
    path::{Path, PathBuf},
};

use anyhow::Context;
use hashbrown::HashMap;

use icicle_fuzzing::{FuzzTarget, Runnable, parse_addr_or_symbol, parse_u64_with_prefix};
use icicle_vm::{Vm, VmExit, cpu::ExceptionCode};

use crate::{
    Config, config,
    coverage::Coverage,
    debugging::{modify_input, trace},
    i2s::log_cmplog_data,
    input::{CortexmMultiStream, MultiStream},
    queue::InputMetadata,
    setup_vm,
    utils::load_json,
};

pub enum SaveMode {
    Full,
    BlocksOnly,
}

impl std::str::FromStr for SaveMode {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "full" => Self::Full,
            "blocks" => Self::BlocksOnly,
            _ => Self::BlocksOnly,
        })
    }
}

pub fn save_block_coverage(mut config: Config, mode: SaveMode) -> anyhow::Result<()> {
    let mut testcases: Vec<InputMetadata> = load_json(&config.workdir.join("testcases.json"))?;
    testcases.sort_by_key(|x| x.found_at);

    let features = config::EnabledFeatures::from_env()?;
    let (mut target, mut vm) = setup_vm(&mut config, &features)?;
    target.initialize_vm(&config.fuzzer, &mut vm)?;

    let mut coverage =
        crate::coverage::BlockCoverage::init(&mut vm, crate::coverage::BucketStrategy::Any, true);

    let snapshot = vm.snapshot();

    let mut all_blocks = vec![];
    let mut block_map = HashMap::new();
    let mut output = vec![];
    for case in testcases {
        vm.restore(&snapshot);
        coverage.reset(&mut vm);

        let input = MultiStream::from_path(&config.workdir.join(format!("queue/{}.bin", case.id)))?;
        target.get_mmio_handler(&mut vm).unwrap().clone_from(&input);

        target.run(&mut vm)?;

        let blocks = coverage.get_blocks(&mut vm);
        let hits = blocks
            .into_iter()
            .map(|addr| {
                *block_map.entry(addr).or_insert_with(|| {
                    let id = all_blocks.len();
                    all_blocks.push((addr, case.found_at, case.id));
                    id
                })
            })
            .collect::<Vec<_>>();

        output.push(serde_json::json!({
            "id": case.id,
            "time_ms": case.found_at,
            "hits": hits,
        }));
    }

    let out = match mode {
        SaveMode::Full => serde_json::json!({ "blocks": all_blocks, "inputs": output }),
        SaveMode::BlocksOnly => serde_json::json!(all_blocks),
    };
    write!(std::io::stdout(), "{out}")?;

    Ok(())
}

pub fn replay(mut config: Config, path: &str) -> anyhow::Result<()> {
    let features = config::EnabledFeatures::from_env()?;
    let (mut target, mut vm) = setup_vm(&mut config, &features)?;
    target.initialize_vm(&config.fuzzer, &mut vm)?;

    let mut input = MultiStream::from_bytes(&InputSource::from_str(path).read()?)
        .with_context(|| format!("Invalid file format: {path}"))?;
    modify_input(&mut input);
    target.get_mmio_handler(&mut vm).unwrap().clone_from(&input);

    //
    // Now we launch in one of three modes depending on what environment variables are configured:
    //

    // GDB mode.
    if let Ok(addr) = std::env::var("GDB_BIND") {
        let elf_path = vm
            .env
            .as_any()
            .downcast_ref::<icicle_cortexm::FuzzwareEnvironment>()
            .unwrap()
            .elf_path
            .clone();
        let mut state = icicle_gdb::ArmStub::new(&mut vm);
        if let Some(elf_path) = elf_path {
            let is_loopback =
                addr.parse::<std::net::SocketAddr>().map_or(false, |x| x.ip().is_loopback());
            // If we are running on a loopback address, assume that GDB will be executed locally
            // and we can use the local ELF path (greatly improves performance).
            let path = match is_loopback {
                false => icicle_gdb::ExePath::Remote(elf_path),
                true => icicle_gdb::ExePath::Local(elf_path),
            };
            tracing::info!("Setting executable path: {path:?}");
            state.set_exe_path(path);
        }
        return icicle_gdb::listen(&addr, state, icicle_gdb::CustomCommands::default());
    }

    // Benchmarking mode.
    if let Ok(trials_str) = std::env::var("TRIALS") {
        let trials = trials_str
            .parse::<u64>()
            .with_context(|| format!("failed to parse TRIALS={trials_str}"))?;
        return replay_bench(vm, target, trials);
    }

    // Tracing dumping mode.
    replay_trace(vm, target)
}

fn replay_bench(mut vm: Vm, mut target: CortexmMultiStream, trials: u64) -> anyhow::Result<()> {
    let core_ids = core_affinity::get_core_ids().unwrap_or(vec![]);
    if let Some(core_id) = core_ids.first() {
        eprintln!("pinning active thread to core: {core_id:?}");
        core_affinity::set_for_current(*core_id);
    }

    let mut snapshot = vm.snapshot();
    let mut cursor_snapshot = target.get_mmio_handler(&mut vm).unwrap().source.snapshot_cursors();

    // Perform a dry run of the input to warm up the VM.
    let exit = target.run(&mut vm);
    vm.recompile();

    // Dump JIT function ID -> guest address mapping for analysis.
    if let Err(e) = vm.jit.dump_jit_mapping("jit_table.txt".as_ref(), vm.env.debug_info().unwrap())
    {
        tracing::warn!("Failed to dump JIT table: {e}")
    }

    let mut expected_icount = vm.cpu.icount();
    eprintln!("[icicle] exited with: {exit:?} (icount = {expected_icount})");

    // We allow replaying from part way through the input if provided from an environment variable.
    if let Some(bp) =
        std::env::var("REPLAY_FROM").ok().and_then(|addr| parse_addr_or_symbol(&addr, &mut vm))
    {
        vm.restore(&snapshot);
        target.get_mmio_handler(&mut vm).unwrap().source.restore_cursors(&cursor_snapshot);

        vm.add_breakpoint(bp);
        let exit = target.run(&mut vm);
        let pc = vm.cpu.read_pc();
        anyhow::ensure!(
            matches!(exit, Ok(VmExit::Breakpoint)) && pc == bp,
            "Failed to hit breakpoint at: {bp:#x} when `REPLAY_FROM` is set (exited with: {exit:?} at: {pc:#x})",
        );
        vm.remove_breakpoint(bp);

        eprintln!("[icicle] hit break: pc={pc:#x} {exit:?} (icount = {})", vm.cpu.icount());
        expected_icount += 1;

        snapshot = vm.snapshot();
        cursor_snapshot = target.get_mmio_handler(&mut vm).unwrap().source.snapshot_cursors();
    }

    let start = std::time::Instant::now();

    for _ in 0..trials {
        vm.restore(&snapshot);
        // Restoring currently triggers a lookup flush, but we know the code will not change between
        // execs so we can skip it here.
        vm.cpu.mem.mapping_changed = false;
        target.get_mmio_handler(&mut vm).unwrap().source.restore_cursors(&cursor_snapshot);
        let _ = target.run(&mut vm);

        // Ensure execution doesn't diverge.
        let icount = vm.cpu.icount();
        anyhow::ensure!(
            icount == expected_icount,
            "Execution diverged during benchmarking:\n\
            expected execution to end at icount={expected_icount}, but ended at icount={icount} instead."
        )
    }

    let elapsed = start.elapsed().as_secs_f64();
    eprintln!("{trials} trials executed in {elapsed:.2} seconds");
    eprintln!("{:.2} trials/second", trials as f64 / elapsed);
    eprintln!("{:.2} ms per trial", (elapsed / trials as f64) * 1000.0);

    Ok(())
}

fn replay_trace(mut vm: Vm, mut target: CortexmMultiStream) -> anyhow::Result<()> {
    let semantics_enabled =
        icicle_fuzzing::parse_bool_env("SAVE_SEMANTIC_TAINT")?.unwrap_or(true);

    // The provenance ledger is fed by the existing MultiStream tracer pipeline:
    // every MMIO read reports the exact input offset it consumed, which is what
    // lets the dynamic pass attribute a value to bytes instead of to a value.
    let ledger = crate::semantic_taint::ReadLedger::new();
    let mut listeners: Vec<Box<dyn trace::IoTracerAny>> = Vec::new();
    if semantics_enabled {
        listeners.push(Box::new(crate::semantic_taint::ReadLedgerTracer::new(ledger.clone())));
    }
    let path_tracer = trace::add_path_tracer(&mut vm, target.mmio_handler.unwrap(), listeners)?;

    let mmio_ranges = taint_mmio_ranges()?;
    let mut collector = crate::semantic_taint::RoleCollector::new(
        ledger.clone(),
        crate::semantic_taint::SinkClassifier::from_env(),
    );

    // Install the dynamic taint pass.  It is disarmed until the analysis pass
    // starts, so it has no effect on ordinary replay.
    let phase_b = if semantics_enabled {
        Some(crate::phase_b::install(
            &mut vm,
            mmio_ranges.clone(),
            Some(Box::new(collector.clone())),
        ))
    }
    else {
        None
    };

    let mut cmplog = None;
    if icicle_fuzzing::parse_bool_env("REPLAY_CMPLOG")?.unwrap_or(false) {
        let cmplog_ref =
            icicle_fuzzing::cmplog2::CmpLog2Builder::new().instrument_calls(true).finish(&mut vm);
        cmplog_ref.set_enabled(&mut vm.cpu, true);
        cmplog = Some(cmplog_ref);
    }

    icicle_fuzzing::add_debug_instrumentation(&mut vm);

    // Discovery pass.
    //
    // The dynamic pass needs the pc -> stream map of MMIO read sites, but that
    // map only exists once the firmware has actually read something.  Replay is
    // deterministic, so one disarmed pass is enough to discover the sites and
    // lift the blocks they live in; the analysis pass then runs against a
    // complete picture.
    let mut passes = 0usize;

    let snapshot = vm.snapshot();
    let cursors = target.get_mmio_handler(&mut vm).unwrap().source.snapshot_cursors();

    let mut read_sites: HashMap<u64, Vec<crate::input::StreamKey>> = HashMap::new();
    if semantics_enabled {
        begin_discovery_pass(&phase_b, &ledger);
        if let Err(error) = target.run(&mut vm) {
            eprintln!("[taint] discovery pass failed: {error:#}");
        }
        read_sites = ledger.read_sites();
        // A PC that read more than one stream is not a problem: the analysis keys
        // every binding by `(pc, stream)`, so each pair becomes its own pseudo-site
        // with its own read sequence.  It is worth naming anyway -- a shared
        // `read_byte()` helper and a register sweep over a bank of ports look
        // alike here and only the roles tell them apart.
        let multiplexed = ledger.multiplexed_read_sites();
        if !multiplexed.is_empty() {
            eprintln!(
                "[taint] {} read site(s) serve more than one stream; each (pc, stream) is \
                 analysed as its own pseudo-site: {multiplexed:x?}",
                multiplexed.len()
            );
        }
        eprintln!(
            "[taint] discovery pass: {} reads across {} site(s) in {} pc(s)",
            ledger.len(),
            read_sites.values().map(|streams| streams.len()).sum::<usize>(),
            read_sites.len()
        );

        vm.restore(&snapshot);
        target.get_mmio_handler(&mut vm).unwrap().source.restore_cursors(&cursors);
        // The path tracer is *not* part of the VM snapshot (only the fuzzer's own
        // Snapshot captures it), so the discovery pass's blocks would otherwise
        // survive into the analysis pass: the trace would mix two executions, and
        // the debug builds' `icount >= prev` assertion inside the hook -- which is
        // not covered by the taint pass's panic guard -- would abort the process.
        path_tracer.clear(&mut vm);
    }

    begin_analysis_pass(&phase_b, &ledger, &mut collector, read_sites, &mmio_ranges);
    passes += 1;

    let exit = target.run(&mut vm)?;

    if let Some(state) = &phase_b {
        state.borrow_mut().flush_pending();
        // Loop-bound observations need the whole run's counts, so they are
        // emitted once execution has finished.
        state.borrow_mut().emit_observations();
    }

    let xpsr = vm.cpu.arch.sleigh.get_varnode("xpsr").unwrap();
    let active_irq = vm.cpu.read_reg(xpsr) & 0x1ff;
    eprintln!(
        "\n[icicle] exited with: {} (icount = {}), active_irq = {active_irq}",
        target.exit_string(exit),
        vm.cpu.icount()
    );
    eprintln!("[icicle] callstack:\n{}", icicle_vm::debug::backtrace(&mut vm));

    let print_count = match std::env::var("PRINT_LAST_BLOCKS").ok() {
        Some(c) => c.parse().context("Failed to parse `PRINT_LAST_BLOCKS` environment variable")?,
        None => 10,
    };
    eprintln!("[icicle] last blocks:\n{}", path_tracer.print_last_blocks(&mut vm, print_count));

    let reglist = icicle_vm::debug::get_debug_regs(&vm.cpu);
    eprintln!("registers:\n{}", icicle_vm::debug::print_regs(&vm, &reglist));

    if let Some(cmplog) = cmplog {
        // Written in a canonical order (see `log_cmplog_data`): the recorder's own
        // iteration order varies between runs, so two runs of the same seed used to
        // differ in line order and in which side of a value pair came first.
        // Artifacts from before that ordering are *not* normalized -- do not mix
        // them with new ones in a diff.
        log_cmplog_data(&mut vm, cmplog, "cmplog.txt".as_ref())?;
    }
    if icicle_fuzzing::parse_bool_env("SAVE_TRACE")?.unwrap_or(true) {
        let symbolize = icicle_fuzzing::parse_bool_env("SYMBOLIZE_TRACE")?.unwrap_or(false);
        path_tracer.save_trace(&mut vm, "trace.txt".as_ref(), symbolize);
    }
    if icicle_fuzzing::parse_bool_env("SAVE_MMIO_READS")?.unwrap_or(false) {
        trace::save_mmio_reads("mmio_reads.txt".as_ref(), &path_tracer.get_mmio_reads(&mut vm));
    }
    if icicle_fuzzing::parse_bool_env("SAVE_SEMANTIC_TAINT")?.unwrap_or(true) {
        let counters = phase_b
            .as_ref()
            .map(|state| state.borrow().counters())
            .unwrap_or_default();
        let output = collector.output(passes);
        crate::semantic_taint::save_reads("semantic_reads.jsonl".as_ref(), &ledger.records())?;
        crate::semantic_taint::save_role_map("role_map.json".as_ref(), &output)?;

        // Two different checks, answering different questions:
        //   * `loads_skew` is a *count*: interpretation claimed a different number of
        //     MMIO reads than the collector saw, so the two sides diverged somewhere;
        //   * `occ_mismatch` is per fragment: the ledger's own occurrence for the
        //     read was not the one the analysis asked for, so that fragment's bytes
        //     belong to a different read than the one that consumed them.  A
        //     positional queue cannot see this at all -- it hands over the next
        //     record and looks consistent.
        let loads_skew =
            (counters.loads_interpreted as i64 - collector.loads_observed() as i64).abs();
        let occ_mismatches = collector.occ_mismatches();
        let unconsumed = ledger.pending();
        // Fingerprint of an unmodelled checksum: one magic comparison whose
        // provenance spans many read sites.
        let magic_max_width = output.magic_sites.iter().map(|site| site.width).max().unwrap_or(0);
        // How to read these counters (the directions are not interchangeable):
        //   * the register-reuse fix moves magic up and checksum down (recalling a
        //     separator constant that a stale accumulator marker had suppressed);
        //   * the masked-checksum fix moves magic down and checksum up (a masked
        //     CRC comparison was being reported as a protocol constant);
        //   * together the aggregate direction depends on the target's mix of the
        //     two shapes, so neither counter validates either fix on its own --
        //     read individual entries instead (small separator-sized constants vs
        //     masked comparisons);
        //   * context-level gating raises `bounded_reads` (same-stream bounds used
        //     to be skipped entirely) while the terminator admission rule makes
        //     `loop_bounds` drop sharply (one per loop instead of one per byte);
        //   * the site-keyed gate fix removes double counting: two gates live at
        //     once on one source used to credit every read twice, which is how a
        //     row could report `gated_count = 274` against `count = 140`.  Now
        //     `loop_bounds` has one row per (source, target) site pair and
        //     `bounded_reads` one mark per read, so both fall to their true values
        //     and every row satisfies `gated <= count` (asserted in
        //     `emit_observations`).  Falling here is the fix working, not recall
        //     lost: compare the per-row `gated` values, not the totals alone.
        //   * `skipped_space` is zero *by construction* in replay: the coverage
        //     bitmap that lives in the extra memory spaces is only installed by the
        //     fuzzing loop's injector, so replay has no non-guest access to skip.
        //     It is not a health signal here.
        //   * `width_mismatch` compares the MMIO model's byte count with the width
        //     of the p-code load.  The model hands over only the bytes whose bits
        //     the firmware uses, so a 32-bit `ldr` that tests one bit is served
        //     from a single byte and counts as a mismatch: a unit difference, not
        //     an alignment error.
        //   * `unconsumed` is split into reads at dropped ambiguous sites (expected)
        //     and leftovers from an early stop (the actual drift signal).
        //   * `pool_discriminants` sizes the literal-pool path (the comparison had
        //     no constant operand, so the value came from the other operand): it
        //     says how much of the magic map rests on a claim the firmware never
        //     stated as an immediate.  Set against `magic` it is the open-path
        //     budget; `MagicSite::from_pool` names the comparison points.
        //   * `pruned_magic` counts magic entries the output stage dropped because
        //     every comparison behind them was demoted (a pointer, a derived
        //     value's boundary, a flag).  A drop here is the output half of the
        //     demotion working, not recall lost -- the events remain in
        //     `magic_sites`.
        //   * `device_writes` counts tainted stores that went into a peripheral
        //     register instead of memory: the interrupt-clear read-modify-write in
        //     the UART handler, the status-bit clears.  A non-zero value is the
        //     separation working -- those writes are neither payload nor checksum,
        //     which is why the status bytes stopped being reported as either.
        //   * `checksum` counts candidate arms, and the arm breakdown printed
        //     below says which one spoke.  A conditional branch lifts to a flag
        //     *algebra* (`bgt` is `NG == OV`), so a flag pair must not count as a
        //     checksum of the data the flags were derived from; a run whose
        //     `equal_both_tainted` total moves with the number of input bytes is
        //     the signature of that mistake.
        //   * `checksum_rejected` counts the shapes the checksum rule considered
        //     and refused (a flag pair, or a comparison of one read with itself).
        //     They stay in `checksum_sites` with the reason, and they mark no
        //     stream as consumed -- which is what keeps them out of the protocol
        //     budget below.
        //   * `bounded_reads_device` counts reads a live gate swept in from a
        //     stream that carries no data role: the firmware's own register polls.
        //     They are read while the bound is in effect, but they are not the
        //     payload the bound was about, so they produce no payload role.
        eprintln!(
            "[taint] schema={} reads={} sites={} fragments={} roles={} blocks={} stores={} \
             magic={} loop_bounds={} checksum={} table_loads={} bounded_reads={} \
             skipped_space={} stack_skipped={} unbound={} width_mismatch={} occ_mismatch={} \
             early_stops={} unconsumed={} dropped_sites={} leftovers={} demoted={} \
             pool_discriminants={} pruned_magic={} device_writes={} panic_disarmed={} \
             magic_max_width={} checksum_rejected={} bounded_reads_device={} \
             stores_value_unknown={} gate_masks={} gate_tests={} gate_exits={} \
             gate_near_misses={} unmodelled_tainted={} phase_b={} loads_skew={}",
            output.report.schema,
            output.report.mmio_reads,
            output.report.read_sites,
            output.report.source_fragments,
            output.report.role_entries,
            output.report.data_blocks,
            output.report.tainted_stores,
            output.report.magic_compares,
            output.report.loop_bounds,
            output.report.checksum_events,
            output.report.table_loads,
            output.report.bounded_reads_marked,
            counters.skipped_space_ops,
            counters.stack_stores_skipped,
            output.report.unbound_loads,
            collector.size_mismatches(),
            occ_mismatches,
            counters.early_stops,
            unconsumed,
            output.report.dropped_site_reads,
            output.report.early_stop_leftovers,
            counters.demoted_discriminants,
            counters.pool_discriminants,
            output.report.pruned_magic_entries,
            counters.device_writes,
            counters.panic_disarmed,
            magic_max_width,
            output.report.rejected_checksum_events,
            output.report.device_state_bounded_reads,
            output.report.store_values_unknown,
            counters.test_masks_derived,
            output.report.gate_tests,
            counters.gate_exits_tainted,
            counters.gate_near_misses,
            counters.unmodelled_ops_tainted,
            crate::phase_b::PHASE_B_REVISION,
            loads_skew
        );
        // The demotion total is an argument only when split: which mechanism took
        // the constant away (a negated add form, a ones-run at the comparison's
        // width, a pointer, or a boundary of a derived value).
        if !counters.demoted_by_reason.is_empty() {
            let parts: Vec<String> = counters
                .demoted_by_reason
                .iter()
                .map(|(why, n)| format!("{why}:{n}"))
                .collect();
            eprintln!("[taint] demoted by reason: {}", parts.join(" "));
        }
        // Where the literal-pool fallback contributed: the attribution a per-site
        // allow-list policy would be written from.
        if !counters.pool_by_pc.is_empty() {
            let parts: Vec<String> = counters
                .pool_by_pc
                .iter()
                .map(|(pc, n)| format!("{pc:#x}:{n}"))
                .collect();
            eprintln!("[taint] pool discriminants by site: {}", parts.join(" "));
        }
        // Where a bit test that never became a gate broke.  The aggregate
        // (`gate_near_misses`) says how many blocks were in that state; only the
        // split says which of the four links is missing, and they need different
        // fixes.
        if !counters.gate_near_miss_by_reason.is_empty() {
            let parts: Vec<String> = counters
                .gate_near_miss_by_reason
                .iter()
                .map(|(why, n)| format!("{why}:{n}"))
                .collect();
            eprintln!("[taint] gate near misses by reason: {}", parts.join(" "));
        }
        // Gates that only the register-scoped view could explain: the measure of
        // what the cross-block fallback buys (and of how much the run depends on
        // it, since that view is not scoped to the function the test came from).
        if counters.gate_test_from_regs > 0 {
            eprintln!(
                "[taint] gate tests from register-scoped bit tests: {}",
                counters.gate_test_from_regs
            );
        }
        // A condition that reached its branch with no bit test, with the chain of
        // operations that defined it and the mask the rules would have inferred.
        // The next engine fix is read off this table: the chain names the arm that
        // drops the bit test, and `would-be mask` saying a bit of the *input* byte
        // means the exit can consult the mask rules as a last resort, while a mask
        // on an intermediate value's top bit means the bit has to be pulled back
        // through the chain that produced it.
        if !counters.gate_near_miss_nodes.is_empty() {
            let mut nodes = counters.gate_near_miss_nodes.clone();
            nodes.sort_by_key(|(_, _, _, _,_, count)| std::cmp::Reverse(*count));
            eprintln!("[taint] conditions with no bit test (branch <- definer):");
            for (branch, def, chain, size, mask, count) in nodes.iter() {
                eprintln!(
                    "[taint]   {:#x} <- {:#x} [{chain}] width={} would-be mask {:#x}: {}",
                    *branch, *def, *size, *mask, *count
                );
            }
        }
        // The bits the firmware branches on.  A gate with high coverage is a
        // constraint ("do not randomise this bit"); a weak one is the firmware
        // serving itself and is reported at 0.5.
        if !output.gate_constraints.is_empty() {
            eprintln!("[taint] gate constraints: {}", output.gate_constraints.len());
            for gate in &output.gate_constraints {
                eprintln!(
                    "[taint]   stream={:#x} off={:?} mask={:#x} branch_pc={:#x} \
                     confirmed={} weak={} unobserved={} confidence={:.2}",
                    gate.stream, gate.offset_range, gate.mask, gate.branch_pc,
                    gate.confirmed, gate.weak, gate.unobserved, gate.confidence
                );
            }
        }
        // Which rule claimed the checksum role, and where.  The confidence does
        // not identify the rule (two arms share 0.5) and each arm has a different
        // false-positive cost, so a run has to name the arm before anything is
        // suppressed.  `width` is how many read sites the event drew on: a width
        // of one means both sides resolved to the same read.
        if !output.checksum_sites.is_empty() {
            let mut by_arm: BTreeMap<&str, u32> = BTreeMap::new();
            for site in &output.checksum_sites {
                *by_arm.entry(site.arm).or_insert(0) += site.events;
            }
            let totals: Vec<String> =
                by_arm.iter().map(|(arm, n)| format!("{arm}:{n}")).collect();
            eprintln!("[taint] checksum arms: {}", totals.join(" "));
            for site in &output.checksum_sites {
                eprintln!(
                    "[taint]   arm={} producer_pc={:#x} width={} events={}",
                    site.arm, site.producer_pc, site.width, site.events
                );
            }
        }
        // `occ_mismatch == 0` on its own proves nothing: both counters advance
        // together even when the two sides are shifted.  Reads the interpreter
        // never claimed that do *not* belong to a dropped site are the actual
        // misalignment signal, because an early stop leaves them in the queue.
        if output.report.early_stop_leftovers > 0 {
            eprintln!(
                "[taint] WARNING: interpretation stopped early {} time(s) and {} read(s) that \
                 belong to no dropped site were never bound; fragment offsets for those sites \
                 may be shifted",
                counters.early_stops, output.report.early_stop_leftovers
            );
        }
        if output.report.dropped_site_reads > 0 {
            eprintln!(
                "[taint] note: {} read(s) belong to read sites that served more than one stream \
                 and were dropped; those bytes are consumed by nothing, which is expected",
                output.report.dropped_site_reads
            );
        }
        if counters.panic_disarmed {
            eprintln!(
                "[taint] WARNING: the taint hook panicked and disarmed; the result is partial"
            );
        }
        if counters.evictions > 0 {
            eprintln!(
                "[taint] WARNING: {} byte(s) were evicted from shadow memory (taint lost); \
                 raise TAINT_MEM_* or narrow the analysis window",
                counters.evictions
            );
        }
    }
    if icicle_fuzzing::parse_bool_env("DEBUG_IL")?.unwrap_or(false) {
        std::fs::write("il.pcode", icicle_vm::debug::dump_semantics(&vm)?)?;
    }

    Ok(())
}

/// Start one analysis pass cleanly.
///
/// Clearing the ledger, resetting the collector and re-arming the engine have to
/// happen together: a pass that starts with reads still queued from the previous
/// one attributes fragments to the wrong offsets, and no counter would catch it
/// (both sides keep counting consistently, just about the wrong bytes).  Keeping
/// the three steps in one place makes it a property of the code rather than a
/// convention the next driver has to remember.
fn begin_analysis_pass(
    phase_b: &Option<std::rc::Rc<std::cell::RefCell<crate::phase_b::PhaseBState>>>,
    ledger: &crate::semantic_taint::ReadLedger,
    collector: &mut crate::semantic_taint::RoleCollector,
    read_sites: HashMap<u64, Vec<crate::input::StreamKey>>,
    mmio_ranges: &[std::ops::Range<u64>],
) {
    begin_pass_state(ledger, collector, read_sites.clone());
    if let Some(state) = phase_b {
        let mut state = state.borrow_mut();
        state.reset_pass(read_sites, mmio_ranges.to_vec());
        state.set_armed(true);
    }
}

/// Start the discovery pass: the same "from zero" guarantee, but disarmed.
///
/// The discovery pass exists only to learn which pcs read which streams, so the
/// engine must collect nothing; what it shares with the analysis pass is the
/// property that the *previous* pass left nothing behind, which is why it goes
/// through a helper rather than a hand-written pair of calls that the next driver
/// has to remember.  It deliberately does not touch the collector: this pass
/// produces no roles, and `begin_analysis_pass` resets the collector once the
/// discovered read sites have been read out.
fn begin_discovery_pass(
    phase_b: &Option<std::rc::Rc<std::cell::RefCell<crate::phase_b::PhaseBState>>>,
    ledger: &crate::semantic_taint::ReadLedger,
) {
    if let Some(state) = phase_b {
        state.borrow_mut().set_armed(false);
    }
    ledger.clear();
}

/// The part of a pass start that needs no VM: the ledger, the collector and the
/// read-site map.
///
/// Split out so the property can be *tested* -- a parent/mutant harness runs
/// several passes in one process, and "the second pass sees only its own reads" is
/// exactly the kind of thing that is obvious in prose and wrong in code (the
/// collector used to keep its sink set across passes, which silently reclassified
/// the next pass's bytes under the previous pass's streams).
fn begin_pass_state(
    ledger: &crate::semantic_taint::ReadLedger,
    collector: &mut crate::semantic_taint::RoleCollector,
    read_sites: HashMap<u64, Vec<crate::input::StreamKey>>,
) {
    ledger.clear();
    collector.reset_pass();
    collector.set_read_sites(read_sites);
}

/// Peripheral address ranges, used both to reject MMIO as taint-addressable
/// memory and to recognise absolute-addressed peripheral loads.
fn taint_mmio_ranges() -> anyhow::Result<Vec<std::ops::Range<u64>>> {
    let start = parse_u64_with_prefix(
        &std::env::var("TAINT_MMIO_START").unwrap_or_else(|_| "0x40000000".to_string()),
    )
    .ok_or_else(|| anyhow::format_err!("invalid TAINT_MMIO_START"))?;
    let end = parse_u64_with_prefix(
        &std::env::var("TAINT_MMIO_END").unwrap_or_else(|_| "0x60000000".to_string()),
    )
    .ok_or_else(|| anyhow::format_err!("invalid TAINT_MMIO_END"))?;
    let ppb_start = parse_u64_with_prefix(
        &std::env::var("TAINT_PPB_START").unwrap_or_else(|_| "0xE0000000".to_string()),
    )
    .ok_or_else(|| anyhow::format_err!("invalid TAINT_PPB_START"))?;
    let ppb_end = parse_u64_with_prefix(
        &std::env::var("TAINT_PPB_END").unwrap_or_else(|_| "0xF0000000".to_string()),
    )
    .ok_or_else(|| anyhow::format_err!("invalid TAINT_PPB_END"))?;

    let mut ranges = Vec::new();
    if start < end {
        ranges.push(start..end);
    }
    if ppb_start < ppb_end {
        ranges.push(ppb_start..ppb_end);
    }
    Ok(ranges)
}

pub fn analyze_crashes(mut config: Config, path: &str) -> anyhow::Result<()> {
    let features = config::EnabledFeatures::from_env()?;
    let (mut target, mut vm) = setup_vm(&mut config, &features)?;
    target.initialize_vm(&config.fuzzer, &mut vm)?;

    let path_tracer = trace::add_path_tracer(&mut vm, target.mmio_handler.unwrap(), Vec::new())?;
    let xpsr_reg = vm.cpu.arch.sleigh.get_varnode("xpsr").unwrap();

    let input_src = InputSource::from_str(path);

    let snapshot = vm.snapshot();
    for entry in input_src.list_files().with_context(|| format!("failed list files in: {path}"))? {
        vm.restore(&snapshot);
        path_tracer.clear(&mut vm);

        let input = MultiStream::from_bytes(&input_src.read_child(&entry)?)
            .with_context(|| format!("Invalid file format: {}", entry.display()))?;
        target.get_mmio_handler(&mut vm).unwrap().clone_from(&input);

        eprintln!("-------------------\n{}", entry.display());
        let exit = target.run(&mut vm)?;
        let active_irq = vm.cpu.read_reg(xpsr_reg) & 0x1ff;
        eprintln!(
            "\n[icicle] exited with: {} (icount = {}), active_irq = {active_irq}",
            target.exit_string(exit),
            vm.cpu.icount()
        );

        if !matches!(exit, VmExit::UnhandledException((ExceptionCode::Environment, _))) {
            // Print additional information for unknown crashes.
            eprintln!("[icicle] callstack:\n{}", icicle_vm::debug::backtrace(&mut vm));
            eprintln!("[icicle] last blocks:\n{}", path_tracer.print_last_blocks(&mut vm, 10));
        }

        eprintln!("-------------------\n");
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum Compression {
    None,
    Gz,
}

impl Compression {
    fn open_reader(&self, path: &Path) -> std::io::Result<Box<dyn std::io::Read>> {
        let file = std::fs::File::open(path)?;
        match self {
            Self::None => Ok(Box::new(std::io::BufReader::new(file))),
            Self::Gz => Ok(Box::new(flate2::read::GzDecoder::new(file))),
        }
    }
}

enum PathKind {
    Path,
    Tar { subpath: Option<PathBuf> },
}

struct InputSource {
    path: PathBuf,
    kind: PathKind,
    compression: Compression,
}

impl InputSource {
    pub fn from_str(path: &str) -> Self {
        let (file_path, subpath) = match path.rsplit_once(":") {
            Some((file_path, subpath)) => (file_path, Some(PathBuf::from(subpath))),
            None => (path, None),
        };

        let (stem, compression) =
            match file_path.strip_suffix(".gz").or_else(|| file_path.strip_suffix(".gzip")) {
                Some(stem) => (stem, Compression::Gz),
                None if file_path.ends_with(".tgz") => (file_path, Compression::None),
                None => (file_path, Compression::None),
            };

        if stem.ends_with(".tgz") || stem.ends_with(".tar") {
            Self { path: PathBuf::from(file_path), kind: PathKind::Tar { subpath }, compression }
        }
        else {
            Self { path: PathBuf::from(file_path), kind: PathKind::Path, compression }
        }
    }

    pub fn list_files(&self) -> anyhow::Result<Vec<PathBuf>> {
        match &self.kind {
            PathKind::Path => {
                anyhow::ensure!(
                    matches!(self.compression, Compression::None),
                    "attempted to read compressed file as a directory: {}",
                    self.path.display()
                );

                let mut entries = vec![];
                for entry in std::fs::read_dir(&self.path)
                    .with_context(|| format!("failed to read: {}", self.path.display()))?
                {
                    entries.push(
                        entry?
                            .path()
                            .file_name()
                            .ok_or_else(|| anyhow::format_err!("invalid file name"))?
                            .into(),
                    );
                }

                Ok(entries)
            }
            PathKind::Tar { subpath } => {
                let subpath = subpath.as_deref().unwrap_or(Path::new(""));
                let mut archive = tar::Archive::new(self.compression.open_reader(&self.path)?);
                Ok(archive
                    .entries()?
                    .flat_map(|x| x)
                    .filter(|x| x.header().entry_type().is_file())
                    .flat_map(|x| Some(x.path().ok()?.into_owned()))
                    .filter(|x| x.parent().unwrap_or(Path::new("")) == subpath)
                    .flat_map(|x| Some(x.file_name()?.into()))
                    .collect())
            }
        }
    }

    pub fn read(&self) -> anyhow::Result<Vec<u8>> {
        match &self.kind {
            PathKind::Path => {
                let mut out = vec![];
                self.compression
                    .open_reader(&self.path)
                    .with_context(|| format!("failed to read: {}", self.path.display()))?
                    .read_to_end(&mut out)?;
                Ok(out)
            }
            PathKind::Tar { subpath } => {
                let file_path = subpath.as_deref().ok_or_else(|| {
                    anyhow::format_err!(
                        "expected subpath for .tar archive: {}",
                        self.path.display()
                    )
                })?;
                read_within(&self.path, file_path, &self.compression)
            }
        }
    }

    pub fn read_child(&self, file: &Path) -> anyhow::Result<Vec<u8>> {
        match &self.kind {
            PathKind::Path => {
                anyhow::ensure!(
                    matches!(self.compression, Compression::None),
                    "attempted to read compressed file as a directory: {}",
                    self.path.display()
                );

                let file_path = self.path.join(file);
                std::fs::read(&file_path)
                    .with_context(|| format!("failed to read: {}", file_path.display()))
            }
            PathKind::Tar { subpath } => {
                let file_path = subpath.as_deref().unwrap_or(Path::new("")).join(file);
                read_within(&self.path, &file_path, &self.compression)
            }
        }
    }
}

fn read_within(
    tar_path: &Path,
    file_path: &Path,
    compression: &Compression,
) -> anyhow::Result<Vec<u8>> {
    let mut archive = tar::Archive::new(compression.open_reader(tar_path)?);
    let Some(mut entry) =
        archive.entries()?.flatten().find(|x| x.path().map_or(false, |path| path == file_path))
    else {
        anyhow::bail!("failed to find {} in {}", file_path.display(), tar_path.display());
    };
    let mut buf = vec![];
    entry.read_to_end(&mut buf).with_context(|| {
        format!("error reading: {} from {}", file_path.display(), tar_path.display())
    })?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    // `AccessContext` and `MagicEvidence` are declared elsewhere: the collector
    // only *imports* them, and a private import is not a path a sibling module can
    // resolve.
    use crate::{
        // The observer's methods are *trait* methods: without the trait in scope
        // they are not callable at all.
        phase_b::{MagicEvidence, PhaseBObserver},
        semantic_taint::{ReadLedger, RoleCollector, SinkClassifier},
        taint::AccessContext,
    };

    /// A pass must start from zero.  This is the invariant the parent/mutant
    /// harness rests on: evidence from the previous pass must not survive into the
    /// next one, or the two analyses being compared are a mixture.
    #[test]
    fn a_second_pass_sees_only_its_own_reads() {
        let ledger = ReadLedger::new();
        let mut collector = RoleCollector::new(ledger.clone(), SinkClassifier::new());

        // First pass: a byte on stream 0x4000 is compared against a constant, so
        // that stream counts as consumed and its bytes as protocol.
        begin_pass_state(&ledger, &mut collector, HashMap::new());
        ledger.record_for_test(0x100, 0x4000, &[0x41]);
        let judged = AccessContext::new(0x100, 0x4000);
        collector.on_source_load(judged, 1);
        collector.on_magic_compare(&[judged], 0x41, 0.85, MagicEvidence::at(0x200));

        let first = collector.output(1);
        assert_eq!(first.report.role_entries, 1);
        assert_eq!(first.report.protocol_bytes, 1);
        assert_eq!(first.report.mmio_reads, 1);

        // Second pass: a read on the *same* stream, but nothing consumes it.
        // Carrying the first pass's sink set over would classify this byte as
        // protocol bytes under the previous pass's evidence.
        begin_pass_state(&ledger, &mut collector, HashMap::new());
        ledger.record_for_test(0x300, 0x4000, &[0x42]);
        collector.on_source_load(AccessContext::new(0x300, 0x4000), 1);

        let second = collector.output(2);
        assert_eq!(second.report.mmio_reads, 1, "the ledger started empty");
        assert!(
            second.role_map.entries.is_empty(),
            "the previous pass's roles survived: {:?}",
            second.role_map.entries
        );
        assert_eq!(second.report.protocol_bytes, 0, "no sink was noted in this pass");
        assert_eq!(second.report.device_state_bytes, 1);
        assert_eq!(second.report.passes, 2);
    }
}
