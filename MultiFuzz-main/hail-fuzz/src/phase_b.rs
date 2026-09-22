//! Phase B: dynamic P-code taint pass.
//!
//! Interprets cached P-code blocks against a [`ShadowState`] driven by live
//! `Cpu` values.  P-code is the ground truth for data flow: interpreting the
//! same statements the JIT executes gives exact, value-independent provenance,
//! so a tainted value points at the MMIO read that produced it rather than at
//! bytes that merely happen to hold the same number.
//!
//! The pass reports exactly the observations the role map is inferred from:
//!
//!  * [`PhaseBObserver::on_source_load`] - a known read site executed, which
//!    binds that read site to the exact input bytes it consumed.
//!  * [`PhaseBObserver::on_tainted_store`] - a sink consumed tainted data.
//!  * [`PhaseBObserver::on_magic_compare`] - a tainted value was compared for
//!    equality against a constant, i.e. the firmware stated the value it
//!    expects to read.
//!  * [`PhaseBObserver::on_loop_bound`] - a tainted value gated a loop that
//!    read another stream repeatedly, i.e. it bounds how much is consumed.
//!
//! Everything here runs out of band (replay); the fuzzing loop never sees it.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::rc::Rc;

use hashbrown::{HashMap, HashSet};
use icicle_vm::{
    BlockTable, Vm,
    cpu::lifter::{Block, BlockExit, Target},
    cpu::{BlockGroup, Cpu},
};
use pcode::{MemId, Op, Value, VarId, VarNode};

use crate::{
    input::StreamKey,
    taint::{AccessContext, ShadowState, TaintTag},
};

pub trait ConcreteEnv {
    fn read_value(&mut self, v: Value) -> Option<u64>;
    fn read_mem(&mut self, addr: u64, size: u8) -> Option<u64>;
}

/// Production `ConcreteEnv` backed by the live emulator CPU.
pub struct LiveEnv<'a> {
    pub cpu: &'a mut Cpu,
    pub mmio_ranges: &'a [Range<u64>],
}

impl<'a> ConcreteEnv for LiveEnv<'a> {
    fn read_value(&mut self, v: Value) -> Option<u64> {
        Some(icicle_vm::cpu::read_value_zxt(self.cpu, v))
    }

    fn read_mem(&mut self, addr: u64, size: u8) -> Option<u64> {
        // >u32::MAX = JIT host pointer in Regs temp slot; would panic is_regular_region.
        if addr > u64::from(u32::MAX) {
            return None;
        }
        let len = size.max(1) as u64;
        // The taint interpreter computes `addr` from possibly-stale concrete state,
        // so it can be any 32-bit value.  This read fires from a block-entry hook,
        // i.e. *in the middle of interpreting a real firmware block*, so it MUST be
        // purely observational: it may not consume fuzz bytes, allocate guest
        // pages, or otherwise mutate the emulator out from under the interpreter.
        //
        //  * `mmio_ranges` skip keeps us out of fuzz-consuming IoMemory.
        //  * `is_regular_region` is NOT sufficient: it returns `true` for
        //    `Unallocated` regions, and reading one drives `init_physical`, which
        //    allocates a page and rewrites the page table / TLB.  It also dispatches
        //    through `read_tlb_miss`, whose `last_io_handler` fast-path can route a
        //    regular address into the MMIO handler off a stale cached range.
        //
        // Require every byte of the span to already resolve to a present *physical*
        // page; otherwise treat the value as unknown (`None`).  `get_physical_addr`
        // returns `Some` only for `MemoryMapping::Physical`, never
        // `Unallocated`/`Io`/unmapped, so this read cannot allocate or dispatch I/O.
        let end = addr.checked_add(len - 1)?;
        if self.mmio_ranges.iter().any(|r| r.contains(&addr) || r.contains(&end))
            || self.cpu.mem.get_physical_addr(addr).is_none()
            || self.cpu.mem.get_physical_addr(end).is_none()
        {
            return None;
        }
        read_regular_memory(self.cpu, addr, size)
    }
}

/// Read `size` bytes from already-mapped physical memory.
///
/// Uses the generic `read::<N>` entry point rather than per-width helpers so the
/// pass does not depend on optional convenience methods of the memory backend.
fn read_regular_memory(cpu: &mut Cpu, addr: u64, size: u8) -> Option<u64> {
    use icicle_vm::cpu::mem::perm;
    match size {
        1 => cpu.mem.read::<1>(addr, perm::NONE).ok().map(|b| b[0] as u64),
        2 => cpu.mem.read::<2>(addr, perm::NONE).ok().map(|b| u16::from_le_bytes(b) as u64),
        4 => cpu.mem.read::<4>(addr, perm::NONE).ok().map(|b| u32::from_le_bytes(b) as u64),
        8 => cpu.mem.read::<8>(addr, perm::NONE).ok().map(u64::from_le_bytes),
        _ => None,
    }
}

/// Receives the observations the role map is inferred from.
///
/// Every method carries *provenance* (the read sites a value came from), never a
/// byte value, so the consumer never has to guess which input bytes were meant.
/// Where an equality comparison happened, and how the compared value's taint was
/// resolved.
///
/// Both fields are audit evidence rather than inputs to the decision: a magic
/// event without its comparison PC cannot be traced back to the instruction that
/// stated the constant, and that is what makes a suspicious discriminant
/// attributable at all.  `from_fallback` says whether the taint came from a
/// definition inside the group being interpreted or from the register file's
/// shadow.  A fallback-sourced tag is legitimate -- a call returns into a new
/// group, so the return register has no definition there -- but it is also how a
/// stale value from an unrelated block can reach an unrelated comparison, so the
/// distinction is recorded rather than silently trusted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MagicEvidence {
    pub compare_pc: u64,
    pub from_fallback: bool,
}

impl MagicEvidence {
    pub const fn at(compare_pc: u64) -> Self {
        Self { compare_pc, from_fallback: false }
    }
}

pub trait PhaseBObserver {
    /// A load at a known MMIO read site executed while interpreting this block.
    fn on_source_load(&mut self, _context: AccessContext, _size: u8) {}

    /// A store whose stored value or address derived from MMIO input.
    fn on_tainted_store(
        &mut self,
        _sources: &[AccessContext],
        _pc: u64,
        _addr: u64,
        _size: u8,
        _value: u64,
    ) {
    }

    /// A tainted value was compared for equality against a constant.
    ///
    /// This is the magic rule: the firmware is stating the value it expects to
    /// read, and the sources say which input bytes it expects it from.
    fn on_magic_compare(
        &mut self,
        _sources: &[AccessContext],
        _value: u64,
        _confidence: f32,
        _evidence: MagicEvidence,
    ) {
    }

    /// A tainted value was written to a peripheral register.
    ///
    /// This is the firmware driving its own device, not a use of the input as
    /// data, so it produces no role -- but it is the counter-evidence that keeps a
    /// status register's own read-modify-write from being read as a payload copy.
    fn on_device_write(&mut self, _sources: &[AccessContext], _pc: u64, _addr: u64) {}

    /// A tainted value gated a loop that read `target` `count` times.
    ///
    /// This is the length rule: the source bounds how much of the target is
    /// consumed.  `count` is the *site total* -- how many times the target was
    /// read in this pass -- which is the quantity comparable with the length
    /// field's own value.  `gated_count` counts only the reads that actually
    /// happened while the gate was in effect, which is what identifies the
    /// payload bytes; the two differ for a loop whose first read precedes the
    /// branch that establishes the gate.
    ///
    /// A loop whose body re-reads its own length field reports a bound where
    /// `target` is the source's own site.  That is the same-site approximation the
    /// pass relies on for helper-style byte reads (one read site reused for every
    /// byte), which cannot be told apart from a self-referential loop; consumers
    /// should expect it rather than treat it as a parsing mistake.
    fn on_loop_bound(
        &mut self,
        _source: AccessContext,
        _source_occs: &[u32],
        _target: AccessContext,
        _count: u64,
        _gated_count: u64,
    ) {
    }

    /// One read of `target` that happened while a loop bound was in effect.
    ///
    /// These are the bytes a length field pays for, i.e. the payload.
    fn on_bounded_read(&mut self, _read: AccessContext) {}

    /// A load whose *address* derived from tainted input: a table lookup indexed
    /// by input bytes.  The value read out is clean, so without this the taint
    /// chain dies here; the caller can also use the event as checksum evidence.
    fn on_table_load(&mut self, _addr_sources: &[AccessContext], _pc: u64) {}

    /// A value that went through a mixing chain (>= 2 combining steps) and is
    /// therefore checksum-like rather than a plain payload byte.
    fn on_checksum(&mut self, _sources: &[AccessContext], _confidence: f32) {}
}

/// What is known about one (source, target) gating pair.
///
/// Only loops are recorded here: a forward branch (an IT-block guard, an
/// interrupt re-entry) gates a decision, not an amount, so it carries no length
/// information and is filtered out at the branch itself.
///
/// Both sides of the key are read *sites* ([`AccessContext::site`], `occ == 0`):
/// the relation "this field's value decides how much of that site is read" is a
/// property of the sites, and a length field re-read inside the loop it gates
/// would otherwise produce one row per occurrence of the same bound.  Which
/// occurrences opened a gate is carried in `source_occs` instead.
#[derive(Debug, Clone, Default)]
struct CtrlObs {
    /// Occurrences of the source *site* that opened a gate on this target.
    source_occs: BTreeSet<u32>,
    /// Reads of the target that happened while a gate was in effect.
    ///
    /// A *set*, not a list: two gates can be live at once on the same source
    /// site (two nested back edges whose conditions both derive from it, which
    /// the per-branch keys cannot deduplicate because their pcs differ), and one
    /// read must count once towards the bound, not once per live gate.  It also
    /// makes `gated_count <= count` structural: every entry is a distinct read of
    /// one site, so the set can never be larger than the number of reads.
    bounded_reads: BTreeSet<AccessContext>,
}

/// How a user op (custom pcode operation) may move taint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UserOpKind {
    /// Pure function of its inputs (e.g. `lzcount`): taint flows input -> output.
    PureData,
    /// No data output at all (e.g. `coprocessor_store`): cannot move taint, and
    /// must not be treated as an opaque call.
    NoOutput,
    /// Unmodelled: conservatively treated as an opaque call (kills argument
    /// registers) rather than guessing a propagation rule.
    Unknown,
}

/// Classify a user op by name.
///
/// The default is deliberately [`UserOpKind::Unknown`]: breaking a taint chain
/// loses a role, but inventing a propagation rule for an operation whose
/// semantics we do not know would fabricate one.
fn classify_user_op(name: &str) -> UserOpKind {
    match name {
        // Bit-manipulation helpers that are pure functions of their input.
        "lzcount" | "clz" | "ctz" | "popcount" | "rev" | "byteswap" | "bswap" => {
            UserOpKind::PureData
        }
        // Coprocessor transfers write a register/memory but produce no p-code
        // output, so there is nothing to propagate and nothing to kill.
        "coprocessor_store" | "coprocessor_load" => UserOpKind::NoOutput,
        _ => UserOpKind::Unknown,
    }
}

/// Upper bound on sub-block steps interpreted per group, guarding against an
/// intra-group back-edge looping forever.  Groups are small (a handful of
/// sub-blocks per guest instruction), so this is never reached in practice.
const MAX_GROUP_STEPS: usize = 256;

/// Dynamic taint interpreter operating one block at a time.
pub struct PhaseBEngine {
    shadow: ShadowState,
    /// Which streams each PC was seen reading, from the discovery pass.
    ///
    /// A PC is not a site by itself: a shared `read_byte()` helper serves whichever
    /// peripheral pointer its caller passed, so one PC can read several streams,
    /// each at its own address.  Keying by PC alone collapsed those into one, which
    /// is why such sites used to be dropped outright.  The pair is the real
    /// identity: every `(pc, stream)` is its own pseudo-site with its own read
    /// sequence.
    read_sites: HashMap<u64, Vec<StreamKey>>,
    mmio_ranges: Vec<Range<u64>>,
    /// The only memory space that belongs to the guest.  Everything at or above
    /// `pcode::RESERVED_SPACE_END` is emulator bookkeeping (the coverage bitmap
    /// and other trace stores), which must never touch guest taint.
    ram_space: MemId,
    def_concrete: HashMap<VarId, Option<u64>>,
    def_taint: HashMap<VarId, TaintTag>,
    /// How each variable in the current block was derived from a subtraction or
    /// addition, so a following `x == 0` can recover the value that was
    /// compared.  Only valid within one block (see `begin_block`).
    def_sub: HashMap<VarId, SubInfo>,
    /// Variables produced by mixing operations (xor/shift/multiply/non-constant
    /// and), i.e. candidates for a checksum accumulator.
    def_mixed: HashSet<VarId>,
    /// Registers that hold a mixed value.  Unlike `def_mixed` this is scoped to
    /// the pass, not the block: a CRC accumulator is built inside a loop and
    /// compared after the loop has exited, in a different group, where the
    /// per-block definition table has long been cleared.
    mixed_regs: HashSet<VarId>,
    /// Variables produced by `value & mask`, mapped to that mask.  A following
    /// `== mask` is a position test, not a protocol constant; a comparison against
    /// any other constant is still a real discriminant.
    def_masked: HashMap<VarId, u64>,
    /// Flag registers of the loaded spec (`ZR`, `CY`, `NG`, `OV`, ...).
    ///
    /// Every conditional branch lifts to a comparison against a flag (`beq` is
    /// `ZR == 0x1`, `bpl` is `NG == 0x0`), so a flag whose taint came from input
    /// would otherwise report 0 or 1 as a value the input should take.
    flag_vars: HashSet<VarId>,
    /// Whether a table lookup happened in the current block, which is the
    /// checksum signal for a following mixed store.
    table_load_in_block: bool,
    cur_pc: u64,
    /// Address of the group currently being interpreted; the reference point for
    /// deciding whether a branch is a loop back-edge.
    cur_group_entry: u64,
    /// Stack pointer at group entry, and the register it was read from.
    cur_sp: Option<u64>,
    sp_varnode: VarNode,
    /// Depth below `sp` that counts as stack.
    stack_window: u64,
    /// Height above `sp` that also counts as stack (a frame stores incoming
    /// arguments just above `sp`).
    stack_window_up: u64,
    /// How many times each read site has been read.
    target_reads: HashMap<AccessContext, u64>,
    /// Entry address of every lifted block, so a `Target::Internal` outside the
    /// cached group can still be resolved to an address.
    block_addrs: HashMap<usize, u64>,
    /// High-water mark of the block table, used to notice a code cache flush.
    max_block_index: usize,
    /// Tainted branch conditions currently gating subsequent reads.
    ///
    /// Keyed by the branch that established the gate as well as the source it came
    /// from: one branch instruction belongs to exactly one loop, so a gate can be
    /// retired precisely when that branch takes its exit, and the same source
    /// gating two different loops is no longer deduplicated away.
    gating: Vec<(AccessContext, u64)>,
    gating_set: HashSet<(AccessContext, u64)>,
    /// Every read that a gate has ever bounded (append-only).
    ///
    /// Used to reject a *new* gate whose source is a byte some earlier gate paid
    /// for: in a terminator loop (`while (buf[i] != 0)`) the byte just read becomes
    /// the next iteration's gate source, which would otherwise turn the whole
    /// payload into a length field.  A genuine second length field is read after
    /// the previous loop has exited and is therefore inside no bound.
    bounded_read_set: HashSet<AccessContext>,
    /// One entry per loop bound: which source site gated which target site.
    ///
    /// Keyed by sites (see [`CtrlObs`]): an occurrence-keyed map would report a
    /// re-read length field once per occurrence and, when two nested back edges
    /// derive their condition from the same read, record every bounded read twice.
    ctrl_obs: HashMap<(AccessContext, AccessContext), CtrlObs>,
    /// User ops classified as pure data / no-output, by id.
    pure_user_ops: HashSet<pcode::PcodeOpId>,
    no_output_user_ops: HashSet<pcode::PcodeOpId>,
    /// Read sites whose load has been interpreted but not yet executed by the
    /// CPU.  The observer is notified at the next block entry, when the MMIO
    /// read has actually happened and its input offset is known.
    pending_source_loads: Vec<(AccessContext, u8)>,
    /// Totals for the report.
    loads_interpreted: u64,
    skipped_space_ops: u64,
    stack_stores_skipped: u64,
    table_loads: u64,
    /// Times interpretation stopped part-way through a group while the CPU kept
    /// executing it (an unresolvable internal branch, or the step cap).  Each one
    /// means the two sides may have drifted apart.
    early_stops: u64,
    /// Set when a panic in the taint hook forced the pass to disarm.
    panic_disarmed: bool,
    observer: Option<Box<dyn PhaseBObserver>>,
}

/// How a variable was produced by an add/subtract, for magic backtracking.
///
/// `cmp a, b` lifts to `tmp = a - b; tmpZR = tmp == 0`, so the value that was
/// actually compared lives in the operands of the subtraction, not in the
/// comparison.
#[derive(Debug, Clone, Copy)]
struct SubInfo {
    /// The operand whose taint we attribute the comparison to.
    base: Value,
    /// The other operand when it is a constant: this is the compared value.
    constant: Option<u64>,
    /// The other operand when it is not a constant (literal-pool path).
    ///
    /// Only meaningful when `constant` is `None`: for the constant form there is
    /// no second operand, and this field holds the base itself.
    other: Value,
    is_add: bool,
}

impl PhaseBEngine {
    pub fn new(read_sites: HashMap<u64, Vec<StreamKey>>, mmio_ranges: Vec<Range<u64>>) -> Self {
        Self {
            shadow: ShadowState::new(),
            read_sites,
            mmio_ranges,
            ram_space: pcode::RAM_SPACE,
            def_concrete: HashMap::new(),
            def_taint: HashMap::new(),
            def_sub: HashMap::new(),
            def_mixed: HashSet::new(),
            mixed_regs: HashSet::new(),
            def_masked: HashMap::new(),
            flag_vars: HashSet::new(),
            table_load_in_block: false,
            cur_pc: 0,
            cur_group_entry: 0,
            cur_sp: None,
            // Replaced by `install` with the real stack-pointer varnode.
            sp_varnode: VarNode::NONE,
            stack_window: default_stack_window(),
            stack_window_up: default_stack_window_up(),
            target_reads: HashMap::new(),
            block_addrs: HashMap::new(),
            max_block_index: 0,
            gating: Vec::new(),
            gating_set: HashSet::new(),
            bounded_read_set: HashSet::new(),
            ctrl_obs: HashMap::new(),
            pure_user_ops: HashSet::new(),
            no_output_user_ops: HashSet::new(),
            pending_source_loads: Vec::new(),
            loads_interpreted: 0,
            skipped_space_ops: 0,
            stack_stores_skipped: 0,
            device_writes: 0,
            demoted_discriminants: 0,
            table_loads: 0,
            early_stops: 0,
            panic_disarmed: false,
            observer: None,
        }
    }

    /// Install the stack pointer varnode and the window below it that counts as
    /// stack (calibrated by `TAINT_STACK_WINDOW`, default 0x800).
    pub fn set_stack_pointer(&mut self, sp: VarNode) {
        self.sp_varnode = sp;
    }

    /// Declare one read site: `pc` reads `stream`.
    ///
    /// Calling it twice with different streams makes the PC a multiplexed site,
    /// which the engine resolves per read by the concrete load address.
    pub fn pin_site(&mut self, pc: u64, stream: StreamKey) {
        let streams = self.read_sites.entry(pc).or_default();
        if !streams.contains(&stream) {
            streams.push(stream);
        }
    }

    /// Register the flag registers resolved from the loaded spec (see `flag_vars`).
    pub fn set_flag_vars(&mut self, ids: impl IntoIterator<Item = VarId>) {
        self.flag_vars.extend(ids);
    }

    /// Install the user ops that are safe to interpret as pure data movement,
    /// resolved by name from the loaded SLEIGH spec.
    fn set_user_ops(&mut self, ops: impl IntoIterator<Item = (pcode::PcodeOpId, UserOpKind)>) {
        for (id, kind) in ops {
            match kind {
                UserOpKind::PureData => {
                    self.pure_user_ops.insert(id);
                }
                UserOpKind::NoOutput => {
                    self.no_output_user_ops.insert(id);
                }
                UserOpKind::Unknown => {}
            }
        }
    }

    /// Install the observer that receives load/store/compare/loop observations.
    /// It survives [`Self::reset_pass`] because it owns cross-pass state.
    pub fn set_observer(&mut self, observer: Box<dyn PhaseBObserver>) {
        self.observer = Some(observer);
    }

    /// Notify the observer about source loads still pending at the end of a run.
    ///
    /// A source load is reported at the *next* block entry, because the
    /// block-entry hook fires before the block executes and the MMIO read (and
    /// hence its input offset) does not exist at interpretation time.  The last
    /// batch of loads therefore has no following entry, so the driver flushes it
    /// here once the run has finished.
    pub fn flush_pending_source_loads(&mut self) {
        if self.pending_source_loads.is_empty() {
            return;
        }
        let pending = std::mem::take(&mut self.pending_source_loads);
        if let Some(observer) = self.observer.as_mut() {
            for (ctx, size) in pending {
                observer.on_source_load(ctx, size);
            }
        }
    }

    /// Reset all per-pass state and refresh the runtime config for a new pass.
    pub fn reset_pass(
        &mut self,
        read_sites: HashMap<u64, Vec<StreamKey>>,
        mmio_ranges: Vec<Range<u64>>,
    ) {
        self.shadow = ShadowState::new();
        self.read_sites = read_sites;
        self.mmio_ranges = mmio_ranges;
        self.def_concrete.clear();
        self.def_taint.clear();
        self.def_sub.clear();
        self.def_mixed.clear();
        self.mixed_regs.clear();
        self.def_masked.clear();
        self.table_load_in_block = false;
        self.cur_pc = 0;
        self.cur_group_entry = 0;
        self.cur_sp = None;
        self.target_reads.clear();
        // `block_addrs` is a map of the lifted program, not of this pass: keep it.
        self.gating.clear();
        self.gating_set.clear();
        // Same lifetime as `ctrl_obs`: the bounded reads it mirrors are part of the
        // pass's evidence and are never removed during it.
        self.bounded_read_set.clear();
        self.ctrl_obs.clear();
        self.pending_source_loads.clear();
        self.loads_interpreted = 0;
        self.device_writes = 0;
        self.demoted_discriminants = 0;
    }

    /// Concrete value of an input, consulting in-block defs first, then `env`.
    fn concrete_in(&self, v: Value, env: &mut dyn ConcreteEnv) -> Option<u64> {
        match v {
            Value::Const(c, s) => Some(mask_to_size(c, s)),
            Value::Var(vn) => {
                if vn.is_invalid() {
                    return None;
                }
                if let Some(c) = self.def_concrete.get(&vn.id) {
                    return *c;
                }
                if vn.id > 0 {
                    // Mask so JIT host pointers in Regs temp slots can't escape as addresses.
                    env.read_value(Value::Var(vn)).map(|val| mask_to_size(val, vn.size))
                }
                else {
                    None
                }
            }
        }
    }

    fn taint_in(&self, v: Value) -> TaintTag {
        self.resolve_taint(v).0
    }

    /// Taint of a value, plus whether it came from the register file's shadow.
    ///
    /// `from_fallback` means the value had no definition in the group currently
    /// being interpreted, so its tag is whatever the shadow holds from the block
    /// that last wrote that register.  That is required for anything crossing a
    /// call or a block boundary -- a callee's return value has no local
    /// definition -- and it is also how a stale tag can reach an unrelated
    /// comparison, so the distinction is reported rather than hidden.
    fn resolve_taint(&self, v: Value) -> (TaintTag, bool) {
        match v {
            Value::Const(..) => (TaintTag::Clean, false),
            Value::Var(vn) => {
                if vn.is_invalid() {
                    return (TaintTag::Clean, false);
                }
                if let Some(t) = self.def_taint.get(&vn.id) {
                    return (t.clone(), false);
                }
                if vn.id > 0 {
                    (self.shadow.reg_tag(vn.id), true)
                }
                else {
                    (TaintTag::Clean, false)
                }
            }
        }
    }

    fn set_def(&mut self, out: pcode::VarNode, concrete: Option<u64>, taint: TaintTag) {
        if out.is_invalid() {
            return;
        }
        // A definition replaces whatever the destination held before, so the
        // "derived from a mixing chain" marker is reset here and re-applied by the
        // arms that actually mix.  A monotonically growing set would instead let a
        // CRC register keep its marker after being reloaded with fresh input, and
        // every later comparison on that register -- a `\r` separator test, say --
        // would then be dismissed as a checksum.  Registers are reused constantly
        // at -O2, and a CRC accumulator and a separator compare often share the
        // same low register number.
        self.def_mixed.remove(&out.id);
        // The same reasoning applies to the other per-variable derivation notes:
        // sleigh reuses temporary ids heavily within a block, and a stale entry
        // would make a later comparison look like it came from an earlier
        // subtraction or mask.
        self.def_masked.remove(&out.id);
        self.def_sub.remove(&out.id);
        if out.id > 0 {
            self.mixed_regs.remove(&out.id);
        }
        self.def_concrete.insert(out.id, concrete);
        self.def_taint.insert(out.id, taint.clone());
        if out.id > 0 {
            self.shadow.set_reg_tag(out.id, taint);
        }
    }

    /// Record whether the value just defined came out of a mixing chain.
    ///
    /// Called after [`Self::set_def`] by the arms that either mix or merely move a
    /// value; everything else keeps the cleared (fresh) state.
    fn note_mixed(&mut self, out: VarNode, is_mixed: bool) {
        if out.is_invalid() {
            return;
        }
        if is_mixed {
            // Block-local for every variable (a temporary only lives that long),
            // and additionally remembered across blocks for registers.
            self.def_mixed.insert(out.id);
            if out.id > 0 {
                self.mixed_regs.insert(out.id);
            }
        }
        else {
            self.def_mixed.remove(&out.id);
            self.mixed_regs.remove(&out.id);
        }
    }

    fn contexts_of(&self, tag: &TaintTag) -> Vec<AccessContext> {
        self.shadow.index.contexts_in_tag(tag)
    }

    /// How many times a read *site* was read over the whole pass.
    ///
    /// `target_reads` holds one counter per site, stored under that site's first
    /// occurrence, so a site (`occ == 0`) has to be translated before lookup.
    fn site_read_count(&self, site: AccessContext) -> u64 {
        self.target_reads.get(&AccessContext::new(site.pc, site.addr)).copied().unwrap_or(0)
    }

    #[cfg(test)]
    fn read_count(&self, context: AccessContext) -> u64 {
        self.target_reads.get(&context).copied().unwrap_or(0)
    }

    /// Emit one observation per loop-bound pair found during the pass.
    ///
    /// One event per (source *site*, target *site*) pair, carrying the
    /// occurrences of the source that opened a gate, the target's total read
    /// count, and the reads that happened while the gate was in effect.
    /// `gated_count` is never greater than `count`: the reads are held as a set,
    /// so a read gated by two live gates is still one read (see [`CtrlObs`]).
    ///
    /// Called after the run: the number of target reads is only complete once
    /// execution has finished.
    pub fn emit_observations(&mut self) {
        let mut events: Vec<(AccessContext, Vec<u32>, AccessContext, u64, Vec<AccessContext>)> =
            Vec::new();
        for (&(source, target), obs) in &self.ctrl_obs {
            // `count` is the number of reads of the target *site* over the whole
            // pass, which is the quantity a length field's value can be compared
            // against (see the confirmation rule in `semantic_taint`).
            let count = self.site_read_count(target);
            // Require >= 3 reads so a two-iteration prologue cannot masquerade as
            // a length field.
            if count < 3 {
                continue;
            }
            // `bounded_reads` is a set of distinct reads of that one site, so a
            // bound can never claim more reads than the site ever had.  Two live
            // gates on one source used to be able to double this; that was the
            // bug this invariant now guards against.
            debug_assert!(
                obs.bounded_reads.len() as u64 <= count,
                "a bound cannot cover more reads than its target site had: {} > {count} \
                 (source {source:?}, target {target:?})",
                obs.bounded_reads.len(),
            );
            let bounded: Vec<AccessContext> = obs.bounded_reads.iter().copied().collect();
            let occs: Vec<u32> = obs.source_occs.iter().copied().collect();
            events.push((source, occs, target, count, bounded));
        }
        // Canonical order: these events become a document that the aligned
        // parent/mutant harness diffs, so the order must not depend on the hash
        // map's per-process seed.
        events.sort_unstable_by_key(|(source, _, target, _, _)| (*source, *target));
        if let Some(observer) = self.observer.as_mut() {
            for (source, occs, target, count, bounded) in events {
                let gated_count = bounded.len() as u64;
                observer.on_loop_bound(source, &occs, target, count, gated_count);
                for read in bounded {
                    observer.on_bounded_read(read);
                }
            }
        }
    }

    /// Totals for the analysis report.
    pub fn counters(&self) -> PhaseBCounters {
        PhaseBCounters {
            loads_interpreted: self.loads_interpreted,
            skipped_space_ops: self.skipped_space_ops,
            stack_stores_skipped: self.stack_stores_skipped,
            device_writes: self.device_writes,
            demoted_discriminants: self.demoted_discriminants,
            table_loads: self.table_loads,
            early_stops: self.early_stops,
            panic_disarmed: self.panic_disarmed,
            evictions: self.shadow.evictions(),
        }
    }
}

impl PhaseBEngine {
    /// Interpret a single basic block, updating shadow state and observations.
    ///
    /// Resets per-block definition state, then interprets the block standalone.
    /// For multi-block groups use [`Self::run_group`], which preserves definition
    /// state across the group's internal sub-blocks.
    #[allow(dead_code)]
    pub fn run_block(&mut self, block: &Block, env: &mut dyn ConcreteEnv) {
        self.begin_block(block.start);
        self.capture_stack_pointer(env);
        self.interpret_block(block, env);
    }

    /// Interpret a whole `BlockGroup` by following its internal control flow.
    ///
    /// The JIT only fires the entry-block hook (injecting `Op::Hook` into a
    /// non-entry sub-block corrupts the JIT), so the engine must itself walk the
    /// group's remaining sub-blocks to observe taint that flows through P-code
    /// generated past an internal branch (ARM IT-blocks, divide zero-checks,
    /// multi-register loads, etc.).
    ///
    /// `base` is the global block index of `blocks[0]` (i.e. `group.blocks.0`),
    /// used to translate `Target::Internal(global_idx)` into a position in
    /// `blocks`.
    ///
    /// Only the sub-block the real CPU would take is interpreted: each internal
    /// branch is resolved by evaluating its condition concretely.  If a condition
    /// cannot be resolved concretely, or control leaves the group (Call/Return/an
    /// external target), interpretation stops: an accepted under-taint, never a
    /// guess down an untaken path (which would manufacture spurious roles).
    pub fn run_group(&mut self, blocks: &[Block], base: usize, env: &mut dyn ConcreteEnv) {
        let Some(first) = blocks.first() else { return };
        let first_start = first.start;
        self.begin_block(first_start);
        self.capture_stack_pointer(env);

        let mut pos = 0usize;
        let mut stopped_cleanly = false;
        for _ in 0..MAX_GROUP_STEPS {
            let block = &blocks[pos];
            self.interpret_block(block, env);

            let next = match &block.exit {
                BlockExit::Jump { target } => Self::internal_pos(target, base, blocks.len()),
                BlockExit::Branch { cond, target, fallthrough } => {
                    match self.concrete_in(*cond, env) {
                        Some(0) => Self::internal_pos(fallthrough, base, blocks.len()),
                        Some(_) => Self::internal_pos(target, base, blocks.len()),
                        // Non-concrete condition: we can't know the CPU's path, so
                        // the rest of the group goes uninterpreted while the CPU
                        // still executes it.
                        None => {
                            self.early_stops += 1;
                            None
                        }
                    }
                }
                // Call/Return transfer control out of the group.
                BlockExit::Call { .. } | BlockExit::Return { .. } => None,
            };

            match next {
                Some(p) => pos = p,
                None => {
                    stopped_cleanly = true;
                    break;
                }
            }
        }
        if !stopped_cleanly {
            // The step cap was hit: interpretation gave up part-way.
            self.early_stops += 1;
        }
    }

    /// Translate a `Target::Internal(global_idx)` into a position within the
    /// captured group slice, or `None` if it leaves the group / isn't internal.
    fn internal_pos(target: &Target, base: usize, len: usize) -> Option<usize> {
        match target {
            Target::Internal(global) => {
                let p = global.checked_sub(base)?;
                (p < len).then_some(p)
            }
            _ => None,
        }
    }

    /// Reset per-block definition state and flush loads deferred from the
    /// previous group (they have now executed on the real CPU).
    fn begin_block(&mut self, start: u64) {
        self.def_concrete.clear();
        self.def_taint.clear();
        self.def_sub.clear();
        self.def_mixed.clear();
        self.def_masked.clear();
        self.cur_pc = start;
        self.cur_group_entry = start;

        // Notify the observer about source loads only now: the block-entry hook
        // fires *before* the block runs, so the corresponding MMIO read (and
        // therefore its input offset) does not exist yet at interpretation time.
        if !self.pending_source_loads.is_empty() {
            let pending = std::mem::take(&mut self.pending_source_loads);
            if let Some(observer) = self.observer.as_mut() {
                for (ctx, size) in pending {
                    observer.on_source_load(ctx, size);
                }
            }
        }
    }

    /// Capture the stack pointer at group entry.
    ///
    /// The interpreter already computes every store address concretely, so the
    /// only thing needed to recognise a stack spill is a reference point; the
    /// group is short enough that the stack pointer cannot move far within it.
    fn capture_stack_pointer(&mut self, env: &mut dyn ConcreteEnv) {
        if self.sp_varnode.is_invalid() {
            self.cur_sp = None;
            return;
        }
        self.cur_sp = env.read_value(Value::Var(self.sp_varnode));
    }

    /// Whether `addr` falls in the stack window below the group-entry `sp`.
    ///
    /// ARM stacks grow down, so a spill lands just below `sp`; the window is
    /// generous because `sp` may move a little within a group.
    fn in_stack_window(&self, addr: u64) -> bool {
        let Some(sp) = self.cur_sp else { return false };
        let (addr, sp) = (addr as i64, sp as i64);
        // Spills land below `sp`, and the frame stores incoming arguments just
        // above it; neither is a consumption of the value.
        addr <= sp.saturating_add(self.stack_window_up as i64)
            && addr > sp.saturating_sub(self.stack_window as i64)
    }

    /// Whether a p-code memory space belongs to the guest.
    ///
    /// Guest data lives in RAM and, for a handful of special registers, in the
    /// register space.  Everything from `RESERVED_SPACE_END` upwards is emulator
    /// bookkeeping allocated by tooling -- the coverage bitmap
    /// (`StoreRef::get_store_id() == id + RESERVED_SPACE_END`) and the other trace
    /// stores.  Those accesses always carry clean values, and a clean store is a
    /// strong update, so interpreting them would wipe the taint of whatever guest
    /// address happens to share the shadow key -- quite apart from the wasted work.
    fn is_guest_space(&self, space: MemId) -> bool {
        space == self.ram_space || space == pcode::REGISTER_SPACE
    }

    /// Whether an address names a peripheral register rather than memory.
    fn is_device_address(&self, addr: u64) -> bool {
        self.mmio_ranges.iter().any(|range| range.contains(&addr))
    }

    /// Interpret one block's P-code and its exit gating, WITHOUT resetting the
    /// per-block definition state (so it can be chained across a group's
    /// sub-blocks).  `cur_pc` is advanced only by `InstructionMarker`s, so a
    /// marker-less continuation sub-block keeps the instruction PC of its parent.
    fn interpret_block(&mut self, block: &Block, env: &mut dyn ConcreteEnv) {
        for stmt in &block.pcode.instructions {
            match stmt.op {
                Op::InstructionMarker => {
                    // Guard against synthetic blocks with a VarNode input;
                    // `.as_u64()` would panic through the extern "C" trampoline.
                    if let Value::Const(pc, _) = stmt.inputs.first() {
                        self.cur_pc = pc;
                    }
                }

                Op::Load(space) => {
                    if !self.is_guest_space(space) {
                        self.skipped_space_ops += 1;
                        continue;
                    }

                    let size = stmt.output.size.max(1);
                    let addr_input = stmt.inputs.first();
                    let addr = self.concrete_in(addr_input, env);

                    // Which stream this read belongs to.  A PC that serves one
                    // stream is that site's stream; a PC that serves several is
                    // resolved per *read* by the concrete load address, which keeps
                    // a shared helper's callers' streams apart instead of merging
                    // or dropping them.
                    let key = self.read_sites.get(&self.cur_pc).and_then(|streams| {
                        match streams.len() {
                            0 => None,
                            1 => streams.first().copied(),
                            _ => addr.and_then(|addr| {
                                streams
                                    .iter()
                                    .copied()
                                    .find(|stream| *stream & 0xffff_ffff == addr & 0xffff_ffff)
                            }),
                        }
                    });
                    if let Some(key) = key {
                        // `occ` is assigned in interpretation order, which is the
                        // order the CPU performs the reads in.  Note that `occ`,
                        // not the address, is what separates two reads: a FIFO read
                        // at two different PC values shares one input cell, so the
                        // cell offset cannot distinguish them.
                        let counter = self
                            .target_reads
                            .entry(AccessContext::new(self.cur_pc, key))
                            .or_insert(0);
                        *counter += 1;
                        let ctx = AccessContext::at(self.cur_pc, key, *counter as u32);

                        // A tainted loop gate seen earlier bounds how much of this
                        // stream is consumed.  Only loop gates reach `gating` (see
                        // the exit handling), so every entry here is a real bound.
                        //
                        // The live gates are folded by source *site* first: two
                        // nested back edges can both take their condition from the
                        // same read site, and they describe the same bound, so one
                        // read must be credited to it once rather than once per
                        // live gate.  Folding here (and not only at the key) is
                        // what keeps `gated_count` a count of reads.
                        let mut gating_sites: BTreeMap<(u64, StreamKey), BTreeSet<u32>> =
                            BTreeMap::new();
                        for (src, _) in &self.gating {
                            gating_sites.entry((src.pc, src.addr)).or_default().insert(src.occ);
                        }
                        let ctx_site = AccessContext::site(ctx.pc, ctx.addr);
                        for ((site_pc, site_stream), occs) in gating_sites {
                            // Only the reads that *opened* a gate are excluded: a
                            // gate cannot bound its own source read.  Later
                            // occurrences of the same site are exactly the helper
                            // style payload (`bl read_byte` reusing one site) that
                            // has to stay bounded, and a second length field read at
                            // another context of the same peripheral must stay
                            // visible too.
                            let source_site = AccessContext::site(site_pc, site_stream);
                            if source_site == ctx_site && occs.contains(&ctx.occ) {
                                continue;
                            }
                            // Key by the target *site* so a loop accumulates all of
                            // its bounded reads into one observation, while the
                            // individual occurrences are kept for payload marking.
                            let obs = self.ctrl_obs.entry((source_site, ctx_site)).or_default();
                            obs.source_occs.extend(occs);
                            obs.bounded_reads.insert(ctx);
                            self.bounded_read_set.insert(ctx);
                        }

                        // LiveEnv returns None for MMIO reads (it must not consume
                        // fuzz bytes), so the loaded value is unknown here.
                        let loaded = addr.and_then(|a| env.read_mem(a, size));

                        // Hand the read site to the observer at the next block
                        // entry, once this load has executed on the real CPU and
                        // its input offset can be read from the tracer pipeline.
                        self.pending_source_loads.push((ctx, size));
                        self.loads_interpreted += 1;

                        let tag = self.shadow.source_tag(ctx);
                        self.set_def(stmt.output, loaded, tag);
                    }
                    else {
                        let val_tag = match addr {
                            Some(a) => self.shadow.mem_tag_range(a as u32, size as usize),
                            None => TaintTag::Clean,
                        };
                        // A load whose *address* is tainted is a table lookup: the
                        // table contents are clean, so unless the address
                        // provenance is folded in the chain dies here.  This is the
                        // indexed-table pattern (e.g. a CRC table driven by input).
                        let addr_tag = self.taint_in(addr_input);
                        if !addr_tag.is_clean() {
                            self.table_loads += 1;
                            self.table_load_in_block = true;
                            let sources = self.contexts_of(&addr_tag);
                            let load_pc = self.cur_pc;
                            if let Some(observer) = self.observer.as_mut() {
                                observer.on_table_load(&sources, load_pc);
                            }
                        }
                        let tag = val_tag.union(&addr_tag);
                        let val = addr.and_then(|a| env.read_mem(a, size));
                        self.set_def(stmt.output, val, tag);
                    }
                }

                Op::Store(space) => {
                    if !self.is_guest_space(space) {
                        self.skipped_space_ops += 1;
                        continue;
                    }

                    let size = value_size(stmt.inputs.second()).max(1);
                    let addr = self.concrete_in(stmt.inputs.first(), env);
                    if let Some(a) = addr {
                        let val_tag = self.taint_in(stmt.inputs.second());
                        let addr_tag = self.taint_in(stmt.inputs.first());
                        let tag = val_tag.union(&addr_tag);
                        self.shadow.set_mem_tag_range(a as u32, size as usize, tag.clone());

                        // Tainted store: hand the precise provenance to the
                        // observer, which attaches the sink role to it.
                        if !tag.is_clean() {
                            // A stack spill is not a consumption: the value goes
                            // out unchanged and comes back later.  Taint still
                            // flows through memory, it just produces no role.
                            if self.in_stack_window(a) {
                                self.stack_stores_skipped += 1;
                            }
                            else if self.is_device_address(a) {
                                // A write to a peripheral register is the firmware
                                // driving its own device, not a use of the input as
                                // data.  It is reported separately for exactly that
                                // reason: a read-modify-write of an interrupt-clear
                                // register is neither a payload copy nor a checksum,
                                // and calling it one is what made the target's
                                // status bytes look like payload.
                                self.device_writes += 1;
                                let sources = self.contexts_of(&tag);
                                let pc = self.cur_pc;
                                if let Some(observer) = self.observer.as_mut() {
                                    observer.on_device_write(&sources, pc, a);
                                }
                            }
                            else {
                                let sources = self.contexts_of(&tag);
                                if self.operand_mixed(stmt.inputs.second()) {
                                    // A mixed value being stored is a checksum
                                    // (or a hash) being finalised, not a payload
                                    // copy: the sink's role would be wrong.
                                    let confidence =
                                        if self.table_load_in_block { 0.7 } else { 0.6 };
                                    if let Some(observer) = self.observer.as_mut() {
                                        observer.on_checksum(&sources, confidence);
                                    }
                                }
                                else {
                                    let value =
                                        self.concrete_in(stmt.inputs.second(), env).unwrap_or(0);
                                    let pc = self.cur_pc;
                                    if let Some(observer) = self.observer.as_mut() {
                                        observer.on_tainted_store(&sources, pc, a, size, value);
                                    }
                                }
                            }
                        }
                    }
                }

                Op::Copy | Op::ZeroExtend | Op::SignExtend => {
                    let src = stmt.inputs.first();
                    let c = self.concrete_in(src, env);
                    let t = self.taint_in(src);
                    self.set_def(stmt.output, c, t);

                    // These are pure moves of the low bits, so a derived-value
                    // property travels with them: `t = x & 0xF0; r2 = t` is still
                    // a mask test, and a moved accumulator is still a checksum.
                    if !stmt.output.is_invalid() {
                        if let Value::Var(vn) = src {
                            if let Some(mask) = self.def_masked.get(&vn.id).copied() {
                                self.def_masked.insert(stmt.output.id, mask);
                            }
                        }
                        let mixed = self.operand_mixed(src);
                        self.note_mixed(stmt.output, mixed);
                    }
                }

                Op::Subpiece(offset) => {
                    let src = stmt.inputs.first();
                    let t = self.taint_in(src);
                    let c = self.concrete_in(src, env).map(|v| {
                        // `offset` is a byte offset (u8), so the shift amount can
                        // legally reach 8*255 = 2040 bits.  A plain `v >> n` with
                        // n >= 64 panics in debug builds and yields an unspecified
                        // value in release; `checked_shr` returns None past the
                        // word width, which for a SUBPIECE beyond the source means
                        // all selected bits are zero.
                        let shifted = v.checked_shr((offset as u32) * 8).unwrap_or(0);
                        mask_to_size(shifted, stmt.output.size)
                    });
                    self.set_def(stmt.output, c, t);
                    // A slice of a mixed value is still that value's low bits.
                    let mixed = self.operand_mixed(src);
                    self.note_mixed(stmt.output, mixed);
                }

                Op::IntAdd | Op::IntSub => {
                    let a = stmt.inputs.first();
                    let b = stmt.inputs.second();
                    let ta = self.taint_in(a);
                    let tb = self.taint_in(b);
                    let c = match (self.concrete_in(a, env), self.concrete_in(b, env)) {
                        (Some(x), Some(y)) => {
                            Some(mask_to_size(eval_binop(stmt.op, x, y), stmt.output.size))
                        }
                        _ => None,
                    };
                    self.set_def(stmt.output, c, ta.union(&tb));

                    // Remember how this value was derived.  A `cmp` lowers to
                    // `tmp = a - b; tmp == 0`, so the value that was compared
                    // lives in these operands, not in the comparison.
                    let is_add = stmt.op == Op::IntAdd;
                    let info = match (a, b) {
                        (base, Value::Const(c, s)) | (Value::Const(c, s), base) => SubInfo {
                            base,
                            constant: Some(mask_to_size(c, s)),
                            other: base,
                            is_add,
                        },
                        (x, y) => SubInfo { base: x, constant: None, other: y, is_add },
                    };
                    if !stmt.output.is_invalid() {
                        self.def_sub.insert(stmt.output.id, info);
                    }
                }

                Op::IntAnd | Op::IntOr | Op::IntXor | Op::IntMul | Op::IntLeft | Op::IntRight
                | Op::IntSignedRight => {
                    let a = stmt.inputs.first();
                    let b = stmt.inputs.second();
                    let ta = self.taint_in(a);
                    let tb = self.taint_in(b);
                    let c = match (self.concrete_in(a, env), self.concrete_in(b, env)) {
                        (Some(x), Some(y)) => {
                            Some(mask_to_size(eval_binop(stmt.op, x, y), stmt.output.size))
                        }
                        _ => None,
                    };
                    self.set_def(stmt.output, c, ta.union(&tb));

                    if !stmt.output.is_invalid() {
                        let mask = constant_of(a).or_else(|| constant_of(b));
                        if stmt.op == Op::IntAnd && mask.is_some() {
                            // `value & mask` extracts bits: the result is a
                            // position test, not the value the firmware compares.
                            self.def_masked.insert(stmt.output.id, mask.expect("checked"));
                            // A mask does not undo a mixing chain, though: the final
                            // `crc & 0xFFFF` before a comparison is still a checksum.
                            // Only an *already mixed* operand propagates here --
                            // tainting alone must not, or `input & 0xF0 == 0xA0`
                            // would be demoted from a discriminant to a checksum.
                            let mixed = self.operand_mixed(a) || self.operand_mixed(b);
                            self.note_mixed(stmt.output, mixed);
                        }
                        else if creates_mixing(stmt.op)
                            && (!ta.is_clean()
                                || !tb.is_clean()
                                || self.operand_mixed(a)
                                || self.operand_mixed(b))
                        {
                            // A combining step applied to tainted or already-mixed
                            // data: this is a candidate checksum accumulator.  The
                            // register copy of the marker outlives the block so the
                            // check still works after the loop exits into another
                            // group.
                            //
                            // Only operations that fold two values together qualify.
                            // `& mask`, `| bit` and `<< n` merely move bits, so they
                            // preserve the marker (above) but must not create one: a
                            // UART byte masked with the port table entry is a mask,
                            // and treating it as a mixing step turned the byte's own
                            // store into checksum evidence.
                            self.note_mixed(stmt.output, true);
                        }
                        else {
                            self.note_mixed(stmt.output, false);
                        }
                    }
                }

                Op::IntEqual => {
                    let a = stmt.inputs.first();
                    let b = stmt.inputs.second();
                    let ta = self.taint_in(a);
                    let tb = self.taint_in(b);

                    // Magic rule: an equality comparison against a constant states
                    // the value the firmware expects, and the provenance says which
                    // input bytes it expects it from.  The real constant almost
                    // never appears here directly though -- see `report_compare`.
                    //
                    // `IntNotEqual` is deliberately excluded: in firmware it is
                    // dominated by loop terminators (`while (b != 0)`), whose
                    // constant is a sentinel rather than a protocol value.
                    match (a, b) {
                        (Value::Var(_), Value::Const(c, s)) => {
                            self.report_compare(ta.clone(), a, mask_to_size(c, s), env)
                        }
                        (Value::Const(c, s), Value::Var(_)) => {
                            self.report_compare(tb.clone(), b, mask_to_size(c, s), env)
                        }
                        // A bare variable-vs-variable equality.  This is rare in
                        // lifted output (ARM `cmp r,r` becomes subtract-and-test,
                        // which the backtrack path covers), but when it does occur
                        // it is an echo or a verification, never a constant.
                        (Value::Var(_), Value::Var(_))
                            if !ta.is_clean() && !tb.is_clean() =>
                        {
                            let sources = self.contexts_of(&ta.union(&tb));
                            if let Some(observer) = self.observer.as_mut() {
                                observer.on_checksum(&sources, 0.5);
                            }
                        }
                        _ => {}
                    }

                    let size = value_size(a);
                    let c = match (self.concrete_in(a, env), self.concrete_in(b, env)) {
                        (Some(x), Some(y)) => eval_cmp(stmt.op, x, y, size),
                        _ => None,
                    };
                    self.set_def(stmt.output, c, ta.union(&tb));
                }

                Op::IntNotEqual
                | Op::IntLess
                | Op::IntSignedLess
                | Op::IntLessEqual
                | Op::IntSignedLessEqual
                | Op::IntCarry
                | Op::IntSignedCarry
                | Op::IntSignedBorrow
                | Op::BoolAnd
                | Op::BoolOr
                | Op::BoolXor => {
                    let a = stmt.inputs.first();
                    let b = stmt.inputs.second();
                    let ta = self.taint_in(a);
                    let tb = self.taint_in(b);
                    let size = value_size(a);
                    // Concretely resolve comparison/boolean ops so internal branch
                    // conditions can be evaluated by `run_group`.
                    let c = match (self.concrete_in(a, env), self.concrete_in(b, env)) {
                        (Some(x), Some(y)) => eval_cmp(stmt.op, x, y, size),
                        _ => None,
                    };
                    self.set_def(stmt.output, c, ta.union(&tb));
                }

                Op::IntDiv | Op::IntSignedDiv | Op::IntRem | Op::IntSignedRem
                | Op::IntRotateLeft | Op::IntRotateRight => {
                    let ta = self.taint_in(stmt.inputs.first());
                    let tb = self.taint_in(stmt.inputs.second());
                    self.set_def(stmt.output, None, ta.union(&tb));
                }

                Op::IntNot | Op::IntNegate | Op::IntCountOnes | Op::IntCountLeadingZeroes => {
                    let t = self.taint_in(stmt.inputs.first());
                    self.set_def(stmt.output, None, t);
                    // Bit manipulation of a mixed value stays part of that value's
                    // mixing chain.
                    let mixed = self.operand_mixed(stmt.inputs.first());
                    self.note_mixed(stmt.output, mixed);
                }

                Op::BoolNot => {
                    let src = stmt.inputs.first();
                    let t = self.taint_in(src);
                    let c = self.concrete_in(src, env).map(|v| (v == 0) as u64);
                    self.set_def(stmt.output, c, t);
                }

                Op::Select(cond) => {
                    // `select(cond)(a, b)`: the result is one of the two data
                    // inputs, so its provenance is their union.  Which input was
                    // taken only affects the concrete value.
                    let a = stmt.inputs.first();
                    let b = stmt.inputs.second();
                    let t = self.taint_in(a).union(&self.taint_in(b));
                    let cond = self.concrete_in(Value::Var(VarNode::new(cond, 1)), env);
                    let c = match (cond, self.concrete_in(a, env), self.concrete_in(b, env)) {
                        (Some(cond), Some(x), Some(y)) => {
                            let chosen = if cond != 0 { x } else { y };
                            Some(mask_to_size(chosen, stmt.output.size))
                        }
                        _ => None,
                    };
                    self.set_def(stmt.output, c, t);
                    let mixed =
                        self.operand_mixed(a) || self.operand_mixed(b);
                    self.note_mixed(stmt.output, mixed);
                }

                Op::PcodeOp(id) => {
                    if self.pure_user_ops.contains(&id) {
                        // A pure function of its input (e.g. `lzcount`): the
                        // result carries the input's provenance.  The concrete
                        // value is left unknown rather than guessed.
                        let t = self.taint_in(stmt.inputs.first());
                        self.set_def(stmt.output, None, t);
                        let mixed = self.operand_mixed(stmt.inputs.first());
                        self.note_mixed(stmt.output, mixed);
                    }
                    else if self.no_output_user_ops.contains(&id) {
                        // Produces no p-code output; it must not be mistaken for an
                        // opaque call either, or a coprocessor transfer would kill
                        // the argument registers on every execution.
                    }
                    else {
                        self.shadow.kill_call_regs();
                    }
                }

                // Instrumentation hooks are invisible to the guest: they must not
                // define anything, kill anything, or produce observations.
                Op::Hook(_) | Op::HookIf(_) => {}

                _ => {
                    if !stmt.output.is_invalid() {
                        self.set_def(stmt.output, None, TaintTag::Clean);
                    }
                }
            }
        }

        if let Some(Value::Var(cond)) = block.exit.cond() {
            if !cond.is_invalid() {
                let cond_tag = self.taint_in(Value::Var(cond));
                // Only a loop gate is a bound.  A forward branch -- an IT-block
                // guard, an interrupt re-entry, the same function called twice --
                // gates a decision, and the fact that it was reached twice says
                // nothing about how much input is consumed.
                if !cond_tag.is_clean() {
                    let branch_pc = self.cur_pc;
                    if let Some(loops_back) = self.branch_loops_back(block, cond, env) {
                        if loops_back {
                            for src in self.contexts_of(&cond_tag) {
                                // Already an active gate for this loop.
                                if self.gating_set.contains(&(src, branch_pc)) {
                                    continue;
                                }
                                // A byte some earlier gate already paid for is the
                                // next iteration of that same loop, not a new field.
                                if self.bounded_read_set.contains(&src) {
                                    continue;
                                }
                                self.gating_set.insert((src, branch_pc));
                                self.gating.push((src, branch_pc));
                            }
                        }
                        else {
                            // The loop exited, so the gate is over.  Without this
                            // the bound would keep labelling every later read of
                            // every other stream as payload.  Matching on the branch
                            // pc retires every gate that branch opened in one go,
                            // which is what a terminator loop needs: its exit
                            // condition carries only the last iteration's byte,
                            // while the gate that is still live was opened by the
                            // first one.
                            self.gating.retain(|(_, pc)| *pc != branch_pc);
                            self.gating_set.retain(|(_, pc)| *pc != branch_pc);
                        }
                    }
                }
            }
        }
    }

    /// Whether this branch goes back to the loop head, i.e. whether the gate it
    /// establishes stays in effect.
    ///
    /// Returns `None` when the branch is not a loop candidate at all (neither
    /// successor goes backwards) or when its condition cannot be resolved
    /// concretely, in which case the gate list is left untouched.
    ///
    /// Only conditional exits are ever seen here: the caller reaches this through
    /// `BlockExit::cond()`, which is `Some` for a branch and `None` for everything
    /// else.  A loop that closes with an unconditional jump (a `while` head whose
    /// tainted test sits in a *forward* branch, then `b top`) therefore never opens
    /// a gate at all -- a known under-approximation, symmetric in both directions
    /// (no gate is pushed, and none is left behind to be popped).
    fn branch_loops_back(
        &self,
        block: &Block,
        cond: VarNode,
        env: &mut dyn ConcreteEnv,
    ) -> Option<bool> {
        let backward = |target: &Target| {
            self.target_addr(target).map_or(false, |addr| addr <= self.cur_group_entry)
        };
        let (target_back, fallthrough_back) = match &block.exit {
            BlockExit::Branch { target, fallthrough, .. } => {
                (backward(target), backward(fallthrough))
            }
            _ => return None,
        };
        if !target_back && !fallthrough_back {
            return None;
        }
        // Which successor is taken decides whether the next iteration happens;
        // for the common counted loop (`subs; bne top`) a zero condition is
        // exactly the exit.
        let taken = self.concrete_in(Value::Var(cond), env)?;
        Some(if taken == 0 { fallthrough_back } else { target_back })
    }

    /// Resolve a branch target to an address.
    ///
    /// `Target::Internal` indexes the global block table, which the engine does
    /// not own; the injector records every lifted block's entry address as it
    /// caches groups, so the lookup succeeds even for targets outside the group
    /// currently being interpreted.
    fn target_addr(&self, target: &Target) -> Option<u64> {
        match target {
            Target::External(Value::Const(addr, _)) => Some(*addr),
            Target::Internal(index) => self.block_addrs.get(index).copied(),
            _ => None,
        }
    }

    /// Whether `v` is a value that has already been through a mixing chain.
    fn operand_mixed(&self, v: Value) -> bool {
        match v {
            Value::Var(vn) => {
                !vn.is_invalid()
                    && (self.def_mixed.contains(&vn.id) || self.mixed_regs.contains(&vn.id))
            }
            Value::Const(..) => false,
        }
    }

    /// Report the constant of an equality comparison, if it is one.
    ///
    /// Real lifted code rarely compares an input against a magic value directly:
    /// `cmp a, b` becomes `tmp = a - b; tmp == 0`, so the interesting constant
    /// lives in the operands of the subtraction.  Recovering it is the point of
    /// tracking `def_sub`.
    fn report_compare(
        &mut self,
        tag: TaintTag,
        var: Value,
        constant: u64,
        env: &mut dyn ConcreteEnv,
    ) {
        let (_, var_from_fallback) = self.resolve_taint(var);

        if constant != 0 {
            // Direct form, e.g. `x == 0x41`.  A bit test (`value & 1 == 1`) is a
            // mask, not a protocol constant -- but only when the mask *is* the
            // constant being tested for.  A partial mask (`value & 0xF0 == 0xA0`)
            // is a real discriminant: setting the byte to 0xA0 passes the check.
            let Value::Var(vn) = var else { return };
            // Every conditional branch lifts to a comparison against a flag
            // (`beq` is `ZR == 0x1`), so a flag whose taint came from input says
            // nothing about what the input should contain.
            if self.flag_vars.contains(&vn.id) {
                return;
            }
            if self.def_masked.get(&vn.id).map_or(false, |mask| *mask == constant) {
                return;
            }
            if tag.is_clean() {
                return;
            }
            // A value that has already been mixed is not a plain input field, so
            // an equality against a constant there is a checksum check; labelling
            // the whole mixing chain as magic would be wrong.
            if self.operand_mixed(var) {
                let sources = self.contexts_of(&tag);
                if let Some(observer) = self.observer.as_mut() {
                    observer.on_checksum(&sources, 0.6);
                }
                return;
            }
            let sources = self.contexts_of(&tag);
            let confidence = self.magic_confidence(var, constant, 0.85);
            self.emit_magic(sources, constant, confidence, var_from_fallback);
            return;
        }

        // Backtrack form: `tmp == 0` where `tmp` came from a subtraction.
        let Value::Var(vn) = var else { return };
        let Some(info) = self.def_sub.get(&vn.id).copied() else { return };

        // The taint must come from the base side (the input being compared).  This
        // single gate rejects the thousands of flag updates (`counter == 0`,
        // `CY == 0`) without needing any instrumentation-specific blacklist.
        let (base_tag, base_from_fallback) = self.resolve_taint(info.base);
        if base_tag.is_clean() {
            return;
        }
        // A flag on the base side is the same branch idiom, one level down.
        if let Value::Var(base_vn) = info.base {
            if self.flag_vars.contains(&base_vn.id) {
                return;
            }
        }

        // A mixed base is a running checksum, so the comparison is its final
        // check -- including the hard-coded variant `tmp = crc - 0x1234; tmp == 0`.
        if self.operand_mixed(info.base) {
            let sources = self.contexts_of(&base_tag);
            if let Some(observer) = self.observer.as_mut() {
                observer.on_checksum(&sources, 0.6);
            }
            return;
        }

        // The compared value: the constant operand, or -- for the literal-pool
        // form, where ARM cannot encode the immediate -- whatever the other
        // operand concretely holds.
        //
        // The "is the other side also input?" test belongs *inside* the
        // non-constant arm and nowhere else: for the constant form `info.other`
        // is the base operand itself (there is no second operand), so testing it
        // unconditionally would reject every real `cmp r, #imm` -- the single
        // most valuable shape this rule exists to catch.
        let (value, confidence) = match info.constant {
            Some(c) => (c, 0.85),
            None => {
                // Two inputs compared against each other: an echo or a
                // verification (a received CRC against a computed one).  Neither
                // side is a value the firmware expects from us, so there is no
                // discriminant here -- that is the checksum signal.
                let other_tag = self.taint_in(info.other);
                if !other_tag.is_clean() {
                    let sources = self.contexts_of(&base_tag.union(&other_tag));
                    if let Some(observer) = self.observer.as_mut() {
                        observer.on_checksum(&sources, 0.5);
                    }
                    return;
                }
                match self.concrete_in(info.other, env) {
                    Some(c) => (c, 0.6),
                    None => return,
                }
            }
        };
        // An add-form comparison states its constant in negated form (`cmn r0, #1`
        // is `r0 + 1 == 0`), so the discriminant is the negation of the constant --
        // masked back to the width of the operands, or a four-byte `+1` comes out
        // as 0xffff_ffff_ffff_ffff and reads like a valid 64-bit constant.
        let width = value_size(info.base).max(value_size(info.other));
        let value = mask_to_size(value, width);
        let value = if info.is_add { mask_to_size(value.wrapping_neg(), width) } else { value };
        let sources = self.contexts_of(&base_tag);
        let confidence = self.magic_confidence(info.base, value, confidence);
        self.emit_magic(sources, value, confidence, base_from_fallback);
    }

    /// The confidence a discriminant deserves, after rejecting the values that are
    /// not discriminants at all.
    ///
    /// Two shapes reach this point with a value the input should never take: an
    /// address (scheduler and allocator code compares pointers, and input taint
    /// reaches those comparisons through the scheduler -- and such a value is
    /// stable across seeds, so no cross-seed filter can see it), and a constant the
    /// compared value is an *offset* of (`c + 1 == 0xe`, where the firmware is
    /// bounding a derived value).  Both stay in the evidence, at a confidence below
    /// the mutation threshold.
    fn magic_confidence(&mut self, var: Value, value: u64, confidence: f32) -> f32 {
        let offset_base = match var {
            Value::Var(vn) => self.def_sub.contains_key(&vn.id),
            _ => false,
        };
        if looks_like_address(value) || offset_base {
            self.demoted_discriminants += 1;
            return DEMOTED_CONFIDENCE;
        }
        confidence
    }

    /// Hand one magic observation to the observer.
    fn emit_magic(
        &mut self,
        sources: Vec<AccessContext>,
        value: u64,
        confidence: f32,
        from_fallback: bool,
    ) {
        let compare_pc = self.cur_pc;
        if let Some(observer) = self.observer.as_mut() {
            observer.on_magic_compare(
                &sources,
                value,
                confidence,
                MagicEvidence { compare_pc, from_fallback },
            );
        }
    }
}

/// Confidence given to a discriminant that is kept for the audit trail but is
/// known not to be a value the input should take (see `looks_like_address` and the
/// offset-base case in `report_compare`): below the mutation side's threshold, so
/// the evidence survives and the role does not.
const DEMOTED_CONFIDENCE: f32 = 0.3;

/// Whether a constant is really an address.
///
/// Calibrated on the target: the lowest code address is 0x0800_0000, and the range
/// covers SRAM (0x2000_xxxx), peripherals (0x4000_xxxx) and the NVIC
/// (0xE000_xxxx).  Pointers are *stable across seeds*, which is exactly why a
/// cross-seed filter cannot see them, so they are rejected on their shape.
fn looks_like_address(v: u64) -> bool {
    v >= 0x0800_0000
}

/// Whether an operation *creates* a mixing chain out of tainted data.
///
/// Bit manipulation (`& mask`, `| bit`, `<< n`) only moves bits around, so it
/// preserves an existing marker but must not create one; the operations that fold
/// two values together do create one.
fn creates_mixing(op: Op) -> bool {
    matches!(op, Op::IntXor | Op::IntMul | Op::IntRight | Op::IntSignedRight)
}

fn value_size(v: Value) -> u8 {
    match v {
        Value::Const(_, s) => s,
        Value::Var(vn) => vn.size,
    }
}

/// The value of a constant operand, masked to its own width.
fn constant_of(v: Value) -> Option<u64> {
    match v {
        Value::Const(c, s) => Some(mask_to_size(c, s)),
        Value::Var(_) => None,
    }
}

/// Size of the stack window below `sp` that is treated as stack, in bytes.
/// Calibrated by `TAINT_STACK_WINDOW` because frame sizes vary by target.
fn default_stack_window() -> u64 {
    std::env::var("TAINT_STACK_WINDOW")
        .ok()
        .and_then(|value| {
            let value = value.trim();
            match value.strip_prefix("0x").or_else(|| value.strip_prefix("0X")) {
                Some(hex) => u64::from_str_radix(hex, 16).ok(),
                None => value.parse().ok(),
            }
        })
        .unwrap_or(0x800)
}

/// Height above `sp` that is also treated as stack, in bytes.
///
/// A prologue stores incoming arguments at small positive offsets from `sp`
/// (`str r3, [sp, #0x8]`); those are spills too.  Device registers are not
/// addressed at small positive offsets from `sp`, so the window is safe.
fn default_stack_window_up() -> u64 {
    std::env::var("TAINT_STACK_WINDOW_UP")
        .ok()
        .and_then(|value| {
            let value = value.trim();
            match value.strip_prefix("0x").or_else(|| value.strip_prefix("0X")) {
                Some(hex) => u64::from_str_radix(hex, 16).ok(),
                None => value.parse().ok(),
            }
        })
        .unwrap_or(0x40)
}

/// Machine-level totals for the analysis report.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct PhaseBCounters {
    /// Loads interpreted at a known MMIO read site.
    pub loads_interpreted: u64,
    /// Accesses to non-guest memory spaces (instrumentation) that were skipped.
    pub skipped_space_ops: u64,
    /// Tainted stores that landed in the stack window and produced no role.
    pub stack_stores_skipped: u64,
    /// Tainted stores that went to a peripheral register instead of memory.
    pub device_writes: u64,
    /// Magic discriminants demoted for being addresses, or boundaries of a value
    /// derived from the input.  Reported so a run can be read without mistaking
    /// them for protocol constants.
    pub demoted_discriminants: u64,
    /// Loads whose address was tainted (table lookups).
    pub table_loads: u64,
    /// Groups where interpretation stopped part-way (see `PhaseBEngine`).
    pub early_stops: u64,
    /// Whether a panic forced the pass to disarm.
    pub panic_disarmed: bool,
    /// Bytes dropped from the shadow memory because it hit its bound.  Non-zero
    /// means taint was lost silently, so it is worth surfacing rather than hiding
    /// in a field nobody reads.
    pub evictions: u64,
}

fn mask_to_size(v: u64, size: u8) -> u64 {
    if size >= 8 {
        v
    }
    else {
        let bits = size as u32 * 8;
        v & ((1u64 << bits) - 1)
    }
}

fn eval_binop(op: Op, a: u64, b: u64) -> u64 {
    match op {
        Op::IntAdd => a.wrapping_add(b),
        Op::IntSub => a.wrapping_sub(b),
        Op::IntAnd => a & b,
        Op::IntOr => a | b,
        Op::IntXor => a ^ b,
        Op::IntMul => a.wrapping_mul(b),
        Op::IntLeft => a.checked_shl(b as u32).unwrap_or(0),
        Op::IntRight => a.checked_shr(b as u32).unwrap_or(0),
        Op::IntSignedRight => ((a as i64).checked_shr(b as u32).unwrap_or(0)) as u64,
        _ => 0,
    }
}

/// Sign-extend the low `size` bytes of `v` to a full `i64`.
fn sign_extend(v: u64, size: u8) -> i64 {
    let bits = (size as u32) * 8;
    if bits == 0 || bits >= 64 {
        return v as i64;
    }
    let shift = 64 - bits;
    ((v << shift) as i64) >> shift
}

/// Concretely evaluate a comparison/boolean P-code op to `0` or `1`, given the
/// operand `size` (in bytes) needed for signed comparisons and carry/borrow.
/// Returns `None` for ops this evaluator does not model (div/rem/rotate), whose
/// concrete value is left unknown.  Inputs are assumed already masked to `size`.
fn eval_cmp(op: Op, a: u64, b: u64, size: u8) -> Option<u64> {
    let bits = (size as u32) * 8;
    let r = match op {
        Op::IntEqual => a == b,
        Op::IntNotEqual => a != b,
        Op::IntLess => a < b,
        Op::IntLessEqual => a <= b,
        Op::IntSignedLess => sign_extend(a, size) < sign_extend(b, size),
        Op::IntSignedLessEqual => sign_extend(a, size) <= sign_extend(b, size),
        Op::IntCarry => bits < 64 && ((a as u128 + b as u128) >> bits) != 0,
        Op::IntSignedCarry => {
            let s = sign_extend(a, size) as i128 + sign_extend(b, size) as i128;
            let r = sign_extend(a.wrapping_add(b), size) as i128;
            s != r
        }
        Op::IntSignedBorrow => {
            let s = sign_extend(a, size) as i128 - sign_extend(b, size) as i128;
            let r = sign_extend(a.wrapping_sub(b), size) as i128;
            s != r
        }
        Op::BoolAnd => (a != 0) && (b != 0),
        Op::BoolOr => (a != 0) || (b != 0),
        Op::BoolXor => (a != 0) ^ (b != 0),
        _ => return None,
    };
    Some(r as u64)
}

// ---------------------------------------------------------------------------
// In-VM driver: block cache + block-entry hook
// ---------------------------------------------------------------------------

/// A cached `BlockGroup`: the entry address maps to all its sub-blocks (clean,
/// pre-hook) plus the global index of the first sub-block, so the engine can
/// follow internal `Target::Internal` edges when the entry-block hook fires.
struct CachedGroup {
    base: usize,
    blocks: Vec<Block>,
}

pub struct PhaseBState {
    blocks: HashMap<u64, CachedGroup>,
    engine: PhaseBEngine,
    mmio_ranges: Vec<Range<u64>>,
    armed: bool,
}

impl PhaseBState {
    pub fn reset_pass(
        &mut self,
        read_sites: HashMap<u64, Vec<StreamKey>>,
        mmio_ranges: Vec<Range<u64>>,
    ) {
        self.mmio_ranges = mmio_ranges.clone();
        self.engine.reset_pass(read_sites, mmio_ranges);
    }

    pub fn set_armed(&mut self, armed: bool) {
        self.armed = armed;
    }

    pub fn emit_observations(&mut self) {
        self.engine.emit_observations();
    }

    /// Flush source loads that had no following block entry (see
    /// [`PhaseBEngine::flush_pending_source_loads`]).
    pub fn flush_pending(&mut self) {
        self.engine.flush_pending_source_loads();
    }

    /// Install the argument-register ids the shadow state should kill on an
    /// opaque call, resolved from the loaded SLEIGH spec.
    pub fn set_call_clobber_regs(&mut self, ids: impl IntoIterator<Item = pcode::VarId>) {
        self.engine.shadow.set_call_clobber(ids);
    }

    /// Machine-level totals, for the analysis report.
    pub fn counters(&self) -> PhaseBCounters {
        self.engine.counters()
    }
}

struct PhaseBInjector {
    state: Rc<RefCell<PhaseBState>>,
    hook: pcode::HookId,
}

/// Cap on the cached-block count (~80 MB at 200k entries).
const MAX_BLOCK_CACHE: usize = 200_000;

impl icicle_vm::CodeInjector for PhaseBInjector {
    /// Inject only into the group's entry block (`group.blocks.0`), mirroring
    /// `BlockHookInjector`.  Injecting into non-entry sub-blocks corrupts the JIT.
    fn inject(&mut self, _cpu: &mut Cpu, group: &BlockGroup, code: &mut BlockTable) {
        let id = group.blocks.0;
        let entry_start = code.blocks[id].start;

        let mut st = self.state.borrow_mut();

        // A VM reset flushes the block table and re-lifts from index 0, which
        // invalidates every cached index -> address mapping (and would silently
        // point cached `Target::Internal` values at the wrong code).  The table
        // shrinking below what we have already seen is the signature of that.
        if code.blocks.len() < st.engine.max_block_index {
            st.blocks.clear();
            st.engine.block_addrs.clear();
            st.engine.max_block_index = 0;
        }

        let base = group.blocks.0;
        let blocks: Vec<Block> = group.range().map(|i| code.blocks[i].clone()).collect();
        // Record every lifted block's entry address: a branch target may be any
        // block in the table, not just one inside the cached group, and the engine
        // needs the address to tell a loop from a forward branch.  Refreshing
        // unconditionally keeps the mapping right across a re-lift.
        for (index, block) in group.range().zip(blocks.iter()) {
            st.engine.block_addrs.insert(index, block.start);
            st.engine.max_block_index = st.engine.max_block_index.max(index + 1);
        }

        // Cache ALL sub-blocks (clean, pre-hook) so the engine can follow the
        // group's internal control flow when the entry hook fires; taint flowing
        // through P-code past an internal branch would otherwise be missed.
        if !st.blocks.contains_key(&entry_start) {
            if st.blocks.len() >= MAX_BLOCK_CACHE {
                return;
            }
            st.blocks.insert(entry_start, CachedGroup { base, blocks });
        }
        drop(st);

        // Insert the hook AFTER the first InstructionMarker, never at position 0.
        // Icicle's JIT establishes the emulated PC at the InstructionMarker; an
        // Op::Hook placed before it leaves the PC unestablished, so the JIT falls
        // back to `jmp r13` and emits spurious host-memory writes that corrupt the
        // heap/stack (observed as crashes at garbage addresses such as 0x9f9f9f9f).
        // If the entry block has no InstructionMarker (synthetic/empty lifter
        // artefact), skip injection entirely: such blocks carry no MMIO loads the
        // taint engine needs, and position-0 insertion is precisely the crash cause.
        //
        // Injection is idempotent: a snapshot restore can re-lift the block from
        // the cache's clean copy, and the fresh block still needs its hook even
        // though its cached group is already present.  Re-injecting into an
        // already-hooked block would instead add a second Op::Hook.
        let entry = &mut code.blocks[id];
        if entry.pcode.instructions.iter().any(|s| s.op == pcode::Op::Hook(self.hook)) {
            return;
        }
        let insert_pos = match entry
            .pcode
            .instructions
            .iter()
            .position(|s| s.op == pcode::Op::InstructionMarker)
        {
            Some(p) => p + 1,
            None => return,
        };

        // Inject hook only into the entry block: non-entry injection corrupts the JIT.
        entry.pcode.instructions.insert(insert_pos, pcode::Op::Hook(self.hook).into());
        code.modified.insert(id);
    }
}

/// Install the Phase B block cache and hook into `vm`.
///
/// Must be called before the blocks of interest are translated.  The pass stays
/// disarmed until the driver calls [`PhaseBState::set_armed`], so installing it
/// has no effect on normal execution.
pub fn install(
    vm: &mut Vm,
    mmio_ranges: Vec<Range<u64>>,
    observer: Option<Box<dyn PhaseBObserver>>,
) -> Rc<RefCell<PhaseBState>> {
    let mut engine = PhaseBEngine::new(HashMap::new(), mmio_ranges.clone());
    if let Some(observer) = observer {
        engine.set_observer(observer);
    }

    let state = Rc::new(RefCell::new(PhaseBState {
        blocks: HashMap::new(),
        engine,
        mmio_ranges,
        armed: false,
    }));

    // Kill the ARM argument registers on an opaque call, resolved from the spec
    // rather than assumed, so a wrong VarId assignment cannot silently leak taint.
    let clobber: Vec<pcode::VarId> = ["r0", "r1", "r2", "r3"]
        .into_iter()
        .filter_map(|name| vm.cpu.arch.sleigh.get_varnode(name).map(|vn| vn.id))
        .collect();
    if !clobber.is_empty() {
        state.borrow_mut().set_call_clobber_regs(clobber);
    }

    // The stack window needs a reference point, and the only reliable one is the
    // live register: local tracking of `sp` would have to model every path.
    if let Some(sp) = vm.cpu.arch.sleigh.get_varnode("sp") {
        state.borrow_mut().engine.set_stack_pointer(sp);
    }

    // Flag registers, resolved from the spec rather than assumed: every
    // conditional branch lifts to a comparison against one of these (`beq` is
    // `ZR == 0x1`), so a flag that picked up input taint must not be reported as a
    // value the input should take.
    let flags: Vec<VarId> = ["ZR", "CY", "NG", "OV", "tmpZR", "tmpCY", "tmpNG", "tmpOV"]
        .into_iter()
        .filter_map(|name| vm.cpu.arch.sleigh.get_varnode(name).map(|vn| vn.id))
        .collect();
    state.borrow_mut().engine.set_flag_vars(flags);

    // Classify the spec's user ops by name.  Anything unrecognised stays
    // conservative (treated as an opaque call) rather than being guessed at.
    let user_ops: Vec<_> = vm
        .cpu
        .arch
        .sleigh
        .get_user_ops()
        .enumerate()
        .map(|(id, name)| (id as pcode::PcodeOpId, classify_user_op(name)))
        .collect();
    if tracing::enabled!(tracing::Level::DEBUG) {
        let names: Vec<_> = vm.cpu.arch.sleigh.get_user_ops().collect();
        tracing::debug!("Phase B user ops: {names:?}");
    }
    state.borrow_mut().engine.set_user_ops(user_ops);

    let hook_state = state.clone();
    let hook = vm.cpu.add_hook(move |cpu: &mut Cpu, addr: u64| {
        // CRITICAL: this closure is invoked through an `extern "C"` trampoline
        // (`InstHook::call` -> `(self.func)(...)`) in BOTH interpreter and JIT
        // modes, so disabling the JIT does NOT remove the C-ABI frame.  A panic
        // escaping the closure would therefore unwind across that frame, which
        // is undefined behaviour (it corrupts the unwinder/stack and has been
        // observed to crash unrelated worker threads with `rip=0x1`).
        //
        // The taint interpreter can legitimately panic on malformed firmware
        // p-code: e.g. `Regs::assert_valid` calls `panic!` (in release builds
        // too) for an out-of-bounds VarNode read.  We catch any such panic here,
        // disarm the engine so the rest of the pass is skipped, and return
        // normally: the worst case is a partial taint result for one seed, not
        // a process crash.  `try_borrow_mut` still guards against re-entrancy.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let Ok(mut st) = hook_state.try_borrow_mut() else { return };
            if !st.armed {
                return;
            }
            let st = &mut *st;
            if let Some(group) = st.blocks.get(&addr) {
                let blocks = group.blocks.clone();
                let base = group.base;
                let mut env = LiveEnv { cpu, mmio_ranges: &st.mmio_ranges };
                st.engine.run_group(&blocks, base, &mut env);
            }
        }));
        if result.is_err() {
            tracing::error!(
                "Phase B taint hook panicked at block {addr:#x}; disarming engine \
                 and skipping the rest of this pass (taint result will be partial)"
            );
            // The borrow held inside the panicking closure was released during
            // unwinding, so we can re-borrow to disarm.  Disarming makes every
            // subsequent block-entry hook in this pass a no-op until the next
            // `reset_pass` rebuilds the engine state.
            if let Ok(mut st) = hook_state.try_borrow_mut() {
                st.engine.panic_disarmed = true;
                st.armed = false;
            }
        }
    });

    vm.add_injector(PhaseBInjector { state: state.clone(), hook });
    state
}

#[cfg(test)]
mod tests {
    use super::*;
    use pcode::VarNode;

    #[derive(Default)]
    struct Recorded {
        loads: Vec<(AccessContext, u8)>,
        stores: Vec<(Vec<AccessContext>, u64, u64, u8, u64)>,
        magics: Vec<(Vec<AccessContext>, u64, f32)>,
        loop_bounds: Vec<(AccessContext, Vec<u32>, AccessContext, u64, u64)>,
        checksums: Vec<(Vec<AccessContext>, f32)>,
        table_loads: usize,
    }

    struct Recorder {
        recorded: Rc<RefCell<Recorded>>,
    }

    impl Recorder {
        fn new() -> (Self, Rc<RefCell<Recorded>>) {
            let recorded = Rc::new(RefCell::new(Recorded::default()));
            (Self { recorded: recorded.clone() }, recorded)
        }
    }

    impl PhaseBObserver for Recorder {
        fn on_source_load(&mut self, context: AccessContext, size: u8) {
            self.recorded.borrow_mut().loads.push((context, size));
        }

        fn on_tainted_store(
            &mut self,
            sources: &[AccessContext],
            pc: u64,
            addr: u64,
            size: u8,
            value: u64,
        ) {
            self.recorded
                .borrow_mut()
                .stores
                .push((sources.to_vec(), pc, addr, size, value));
        }

        fn on_magic_compare(
            &mut self,
            sources: &[AccessContext],
            value: u64,
            confidence: f32,
            _evidence: MagicEvidence,
        ) {
            self.recorded
                .borrow_mut()
                .magics
                .push((sources.to_vec(), value, confidence));
        }

        fn on_loop_bound(
            &mut self,
            source: AccessContext,
            source_occs: &[u32],
            target: AccessContext,
            count: u64,
            gated_count: u64,
        ) {
            self.recorded
                .borrow_mut()
                .loop_bounds
                .push((source, source_occs.to_vec(), target, count, gated_count));
        }

        fn on_checksum(&mut self, sources: &[AccessContext], confidence: f32) {
            self.recorded.borrow_mut().checksums.push((sources.to_vec(), confidence));
        }

        fn on_table_load(&mut self, _addr_sources: &[AccessContext], _pc: u64) {
            self.recorded.borrow_mut().table_loads += 1;
        }
    }

    struct MockEnv {
        regs: HashMap<i16, u64>,
        mem: HashMap<u64, u64>,
    }

    impl MockEnv {
        fn new() -> Self {
            Self { regs: HashMap::new(), mem: HashMap::new() }
        }
    }

    impl ConcreteEnv for MockEnv {
        fn read_value(&mut self, v: Value) -> Option<u64> {
            match v {
                Value::Const(c, _) => Some(c),
                Value::Var(vn) => self.regs.get(&vn.id).copied(),
            }
        }

        fn read_mem(&mut self, addr: u64, _size: u8) -> Option<u64> {
            self.mem.get(&addr).copied()
        }
    }

    fn lifter_block(pcode: pcode::Block, start: u64, end: u64, exit: BlockExit) -> Block {
        Block {
            pcode,
            entry: None,
            start,
            end,
            context: 0,
            exit,
            breakpoints: 0,
            num_instructions: 0,
        }
    }

    fn marker(pc: u64) -> pcode::Instruction {
        (pcode::Op::InstructionMarker, pcode::Value::Const(pc, 8)).into()
    }

    fn reg(id: i16) -> VarNode {
        VarNode::new(id, 4)
    }

    fn engine() -> PhaseBEngine {
        PhaseBEngine::new(HashMap::new(), vec![0x4000_0000..0x6000_0000])
    }

    /// A read site is reported at the next block entry, once the MMIO read (and
    /// therefore the input bytes it consumed) actually happened.
    #[test]
    fn source_load_is_reported_after_the_read_executes() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0000);

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        p.push((reg(1), Op::Load(0), reg(5)));
        let block = lifter_block(p, 0x100, 0x104, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(5, 0x5800_0000);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));
        e.run_block(&block, &mut env);
        e.flush_pending_source_loads();

        let source = AccessContext::new(0x100, 0x5800_0000);
        assert_eq!(recorded.borrow().loads, vec![(source, 4)]);
        assert_eq!(e.read_count(source), 1);
    }

    /// A store of tainted data reaches the observer with its exact provenance.
    #[test]
    fn tainted_store_reports_provenance_to_observer() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0000);

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        p.push((reg(10), Op::Load(0), reg(5)));
        p.push(marker(0x104));
        p.push((VarNode::NONE, Op::Store(0), reg(6), reg(10)));
        let block = lifter_block(p, 0x100, 0x108, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(5, 0x5800_0000);
        env.regs.insert(6, 0x2000_0000);
        env.regs.insert(10, 0xdead_beef);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));
        e.run_block(&block, &mut env);

        let recorded = recorded.borrow();
        let source = AccessContext::new(0x100, 0x5800_0000);
        assert_eq!(recorded.stores.len(), 1, "exactly one tainted store must be reported");
        let (sources, pc, addr, size, value) = &recorded.stores[0];
        assert_eq!(sources, &vec![source], "store provenance must be the MMIO read site");
        assert_eq!(*pc, 0x104);
        assert_eq!(*addr, 0x2000_0000);
        assert_eq!(*size, 4);
        assert_eq!(*value, 0xdead_beef);
        assert!(!e.shadow.mem_tag_range(0x2000_0000, 4).is_clean());
    }

    /// The magic rule: comparing a tainted value against a constant reports that
    /// constant together with the exact read site it is expected from.
    #[test]
    fn magic_compare_reports_the_expected_constant() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0000);

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        p.push((reg(1), Op::Load(0), reg(5)));
        p.push(marker(0x104));
        p.push((reg(2), Op::IntEqual, reg(1), Value::Const(0x41, 1)));
        let block = lifter_block(p, 0x100, 0x108, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(5, 0x5800_0000);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));
        e.run_block(&block, &mut env);

        let recorded = recorded.borrow();
        let source = AccessContext::new(0x100, 0x5800_0000);
        assert_eq!(recorded.magics, vec![(vec![source], 0x41, 0.85)]);
    }

    /// The constant may appear on the left of the comparison as well.
    #[test]
    fn magic_compare_handles_constant_on_either_side() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0000);

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        p.push((reg(1), Op::Load(0), reg(5)));
        p.push(marker(0x104));
        p.push((reg(2), Op::IntEqual, Value::Const(7, 2), reg(1)));
        let block = lifter_block(p, 0x100, 0x108, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(5, 0x5800_0000);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));
        e.run_block(&block, &mut env);

        let recorded = recorded.borrow();
        let source = AccessContext::new(0x100, 0x5800_0000);
        assert_eq!(recorded.magics, vec![(vec![source], 7, 0.85)]);
    }

    /// A range guard is a bound, not a discriminant, so `IntLess` must not
    /// report a magic constant.
    #[test]
    fn range_guard_does_not_report_a_magic_constant() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0000);

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        p.push((reg(1), Op::Load(0), reg(5)));
        p.push(marker(0x104));
        p.push((reg(2), Op::IntLess, reg(1), Value::Const(256, 4)));
        let block = lifter_block(p, 0x100, 0x108, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(5, 0x5800_0000);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));
        e.run_block(&block, &mut env);

        assert!(recorded.borrow().magics.is_empty(), "a bound is not a discriminant");
    }

    /// A comparison of untainted values says nothing about the input.
    #[test]
    fn clean_compare_reports_no_magic() {
        let mut e = engine();

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        p.push((reg(1), Op::Copy, Value::Const(0x41, 1)));
        p.push((reg(2), Op::IntEqual, reg(1), Value::Const(0x41, 1)));
        let block = lifter_block(p, 0x100, 0x108, BlockExit::invalid());

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));
        e.run_block(&block, &mut MockEnv::new());

        assert!(recorded.borrow().magics.is_empty());
    }

    /// The length rule: a tainted value gating a loop that reads another stream
    /// at least three times is reported as a loop bound.
    #[test]
    fn loop_bound_is_reported_for_looping_gate() {
        let mut e = engine();
        e.pin_site(0x200, 0x5800_0008);
        e.pin_site(0x300, 0x5800_0000);

        let mut p1 = pcode::Block::new();
        p1.push(marker(0x200));
        p1.push((reg(1), Op::Load(0), reg(9)));
        p1.push(marker(0x204));
        p1.push((reg(2), Op::IntLess, reg(3), reg(1)));
        let exit1 = BlockExit::Branch {
            cond: Value::Var(reg(2)),
            // A real back-edge: the loop head is this block, so the taken target
            // is at (not after) the group entry.  Nothing else counts as a loop.
            target: Target::External(Value::Const(0x200, 4)),
            fallthrough: Target::External(Value::Const(0x400, 4)),
        };
        let b1 = lifter_block(p1, 0x200, 0x208, exit1);

        let mut p2 = pcode::Block::new();
        p2.push(marker(0x300));
        p2.push((reg(5), Op::Load(0), reg(8)));
        let b2 = lifter_block(p2, 0x300, 0x304, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(9, 0x5800_0008);
        env.regs.insert(8, 0x5800_0000);
        env.regs.insert(3, 0);
        // The bound register is loaded from the stream, so its concrete value has
        // to come from there too: without it the branch condition is unresolvable,
        // `branch_loops_back` declines to act, and no gate is ever established.
        env.mem.insert(0x5800_0008, 5);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));

        for _ in 0..3 {
            e.run_block(&b1, &mut env);
            e.run_block(&b2, &mut env);
        }
        e.emit_observations();

        // Both sides are read *sites*: the bound is a relation between sites, and
        // the occurrences that opened it are reported separately.
        let source = AccessContext::site(0x200, 0x5800_0008);
        let target = AccessContext::site(0x300, 0x5800_0000);
        // Two events, and the first one is expected rather than a bug.
        //
        // The length field is re-read inside the loop body, so from round two on
        // its own reads are bounded by the gate it opened in round one: that is the
        // same-site approximation this pass relies on for helper-style byte reads
        // (`bl read_byte` reusing one read site, where the payload is marked
        // *because* later occurrences are bounded).  The engine cannot tell the two
        // shapes apart -- both are "a later occurrence of a site is gated by a gate
        // that site opened" -- so the self-site entry is the observable cost of
        // keeping helper targets working.  It is reported at the site's own stream,
        // and the second event is the one the rule is really about.
        assert_eq!(
            recorded.borrow().loop_bounds,
            vec![(source, vec![1], source, 3, 2), (source, vec![1], target, 3, 3)],
            "the self-site event is the documented same-site approximation"
        );
    }

    /// A gate evaluated only once is a decision, not a length.
    #[test]
    fn single_activation_is_not_a_loop_bound() {
        let mut e = engine();
        e.pin_site(0x200, 0x5800_0008);
        e.pin_site(0x300, 0x5800_0000);

        let mut p1 = pcode::Block::new();
        p1.push(marker(0x200));
        p1.push((reg(1), Op::Load(0), reg(9)));
        p1.push(marker(0x204));
        p1.push((reg(2), Op::IntEqual, reg(1), Value::Const(3, 4)));
        let exit1 = BlockExit::Branch {
            cond: Value::Var(reg(2)),
            // A real back-edge, so the gate *is* established: the point of this
            // test is that a loop taken once is still not a length.
            target: Target::External(Value::Const(0x200, 4)),
            fallthrough: Target::External(Value::Const(0x400, 4)),
        };
        let b1 = lifter_block(p1, 0x200, 0x208, exit1);

        let mut p2 = pcode::Block::new();
        p2.push(marker(0x300));
        p2.push((reg(5), Op::Load(0), reg(8)));
        let b2 = lifter_block(p2, 0x300, 0x304, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(9, 0x5800_0008);
        env.regs.insert(8, 0x5800_0000);
        env.mem.insert(0x5800_0008, 3);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));
        e.run_block(&b1, &mut env);
        e.run_block(&b2, &mut env);
        e.emit_observations();

        assert!(recorded.borrow().loop_bounds.is_empty(), "count=1 must not be a length");
    }

    /// A loop that reads the target fewer than three times is not a length.
    #[test]
    fn loop_with_two_target_reads_is_not_a_length() {
        let mut e = engine();
        e.pin_site(0x200, 0x5800_0008);
        e.pin_site(0x300, 0x5800_0000);

        let mut p1 = pcode::Block::new();
        p1.push(marker(0x200));
        p1.push((reg(1), Op::Load(0), reg(9)));
        p1.push(marker(0x204));
        p1.push((reg(2), Op::IntLess, reg(3), reg(1)));
        let exit1 = BlockExit::Branch {
            cond: Value::Var(reg(2)),
            // A real back-edge: the gate must exist for the assertion to mean
            // "two reads is not enough", rather than "no gate at all".
            target: Target::External(Value::Const(0x200, 4)),
            fallthrough: Target::External(Value::Const(0x400, 4)),
        };
        let b1 = lifter_block(p1, 0x200, 0x208, exit1);

        let mut p2 = pcode::Block::new();
        p2.push(marker(0x300));
        p2.push((reg(5), Op::Load(0), reg(8)));
        let b2 = lifter_block(p2, 0x300, 0x304, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(9, 0x5800_0008);
        env.regs.insert(8, 0x5800_0000);
        env.regs.insert(3, 0);
        env.mem.insert(0x5800_0008, 5);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));
        for _ in 0..2 {
            e.run_block(&b1, &mut env);
            e.run_block(&b2, &mut env);
        }
        e.emit_observations();

        assert!(recorded.borrow().loop_bounds.is_empty(), "count=2 must not be a length");
    }

    /// A forward-only branch is never a length, however often it is reached.
    ///
    /// This is the IT-block / repeated-call case: the guard is derived from
    /// flags that an earlier input comparison may have tainted, so it would enter
    /// `gating` under a "reached twice means loop" rule and manufacture a bound
    /// that does not exist.
    #[test]
    fn forward_branch_is_not_a_loop_bound_even_when_repeated() {
        let mut e = engine();
        e.pin_site(0x200, 0x5800_0008);
        e.pin_site(0x300, 0x5800_0000);

        let mut p1 = pcode::Block::new();
        p1.push(marker(0x200));
        p1.push((reg(1), Op::Load(0), reg(9)));
        p1.push(marker(0x204));
        p1.push((reg(2), Op::IntEqual, reg(1), Value::Const(3, 4)));
        let exit1 = BlockExit::Branch {
            cond: Value::Var(reg(2)),
            // Both successors are ahead of this block: a decision, not a loop.
            target: Target::External(Value::Const(0x300, 4)),
            fallthrough: Target::External(Value::Const(0x400, 4)),
        };
        let b1 = lifter_block(p1, 0x200, 0x208, exit1);

        let mut p2 = pcode::Block::new();
        p2.push(marker(0x300));
        p2.push((reg(5), Op::Load(0), reg(8)));
        let b2 = lifter_block(p2, 0x300, 0x304, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(9, 0x5800_0008);
        env.regs.insert(8, 0x5800_0000);
        // Resolvable condition, so the branch really is evaluated: the point is
        // that a forward branch is rejected on its shape, not on ignorance.
        env.mem.insert(0x5800_0008, 3);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));
        for _ in 0..3 {
            e.run_block(&b1, &mut env);
            e.run_block(&b2, &mut env);
        }
        e.emit_observations();

        assert!(
            recorded.borrow().loop_bounds.is_empty(),
            "a forward branch must never produce a length"
        );
    }

    /// The terminator loop (`while (buf[i] != 0)`) is the shape that makes a naive
    /// gate rule collapse the whole payload into length fields.
    ///
    /// Every iteration's byte becomes the next iteration's branch condition, so
    /// each round would open a fresh gate and be reported as a length field.  The
    /// admission rule keeps exactly one gate (the byte read before the loop was
    /// ever bounded), and the exit branch retires it.
    #[test]
    fn terminator_loop_opens_one_gate_and_retires_it() {
        let mut e = engine();
        e.pin_site(0x200, 0x5800_0000);
        e.pin_site(0x400, 0x5800_0004);

        // do { r5 = LOAD[DR]; } while (r5 != 0);
        let mut p = pcode::Block::new();
        p.push(marker(0x200));
        p.push((reg(5), Op::Load(0), reg(8)));
        p.push(marker(0x204));
        p.push((reg(6), Op::IntNotEqual, reg(5), Value::Const(0, 4)));
        let exit = BlockExit::Branch {
            cond: Value::Var(reg(6)),
            target: Target::External(Value::Const(0x200, 4)),
            fallthrough: Target::External(Value::Const(0x400, 4)),
        };
        let body = lifter_block(p, 0x200, 0x208, exit);

        // The field that follows the loop, read from a different site.
        let mut p2 = pcode::Block::new();
        p2.push(marker(0x400));
        p2.push((reg(7), Op::Load(0), reg(9)));
        let after = lifter_block(p2, 0x400, 0x408, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(8, 0x5800_0000);
        env.regs.insert(9, 0x5800_0004);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));

        // Three iterations; the third byte terminates the string.
        env.mem.insert(0x5800_0000, 5);
        e.run_block(&body, &mut env);
        e.run_block(&body, &mut env);
        env.mem.insert(0x5800_0000, 0);
        e.run_block(&body, &mut env);
        // The field after the loop must not be swept into the retired bound.
        e.run_block(&after, &mut env);
        e.emit_observations();

        // `byte` names a *site* (the bound's two sides are sites); `following` is
        // an occurrence, because that is what the bounded-read set stores.
        let byte = AccessContext::site(0x200, 0x5800_0000);
        let following = AccessContext::new(0x400, 0x5800_0004);
        assert_eq!(
            recorded.borrow().loop_bounds,
            vec![(byte, vec![1], byte, 3, 2)],
            "exactly one gate: 3 reads total, 2 of them after the gate existed"
        );
        assert!(e.gating.is_empty(), "the exit branch must retire the gate");
        assert!(
            !e.bounded_read_set.contains(&following),
            "a field read after the loop is not inside any bound"
        );
        assert!(
            e.ctrl_obs.values().all(|obs| !obs.bounded_reads.contains(&following)),
            "the following field must not appear in any bounded span"
        );
    }

    /// Two loops driven by the *same* length field keep independent gates.
    ///
    /// This is the whole reason the gate key carries the branch pc: if retiring a
    /// gate matched only on the source context, exiting the second loop would also
    /// retire the first one, and that loop would silently stop bounding its reads
    /// (its payload would revert to `Propagated`).
    #[test]
    fn two_loops_sharing_one_length_keep_separate_gates() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0008);
        e.pin_site(0x200, 0x5800_0000);
        e.pin_site(0x300, 0x5800_000c);

        // The length field, read once and shared by both loops.
        let mut p0 = pcode::Block::new();
        p0.push(marker(0x100));
        p0.push((reg(1), Op::Load(0), reg(10)));
        let length = lifter_block(p0, 0x100, 0x108, BlockExit::invalid());

        // Loop 1 (branch at 0x204): reads one stream.
        let mut p1 = pcode::Block::new();
        p1.push(marker(0x200));
        p1.push((reg(5), Op::Load(0), reg(8)));
        p1.push(marker(0x204));
        p1.push((reg(2), Op::IntLess, reg(3), reg(1)));
        let loop1 = lifter_block(
            p1,
            0x200,
            0x208,
            BlockExit::Branch {
                cond: Value::Var(reg(2)),
                target: Target::External(Value::Const(0x200, 4)),
                fallthrough: Target::External(Value::Const(0x300, 4)),
            },
        );

        // Loop 2 (branch at 0x304): reads a different stream, same length field.
        let mut p2 = pcode::Block::new();
        p2.push(marker(0x300));
        p2.push((reg(6), Op::Load(0), reg(9)));
        p2.push(marker(0x304));
        p2.push((reg(2), Op::IntLess, reg(3), reg(1)));
        let loop2 = lifter_block(
            p2,
            0x300,
            0x308,
            BlockExit::Branch {
                cond: Value::Var(reg(2)),
                target: Target::External(Value::Const(0x300, 4)),
                fallthrough: Target::External(Value::Const(0x400, 4)),
            },
        );

        let mut env = MockEnv::new();
        env.regs.insert(10, 0x5800_0008);
        env.regs.insert(8, 0x5800_0000);
        env.regs.insert(9, 0x5800_000c);
        // The length register's concrete value: both branches compare counter < 5.
        env.regs.insert(1, 5);
        env.regs.insert(3, 0);

        let (observer, _recorded) = Recorder::new();
        e.set_observer(Box::new(observer));

        e.run_block(&length, &mut env);
        e.run_block(&loop1, &mut env); // loop 1's gate opens
        e.run_block(&loop2, &mut env); // loop 2's gate opens alongside it
        assert_eq!(e.gating.len(), 2, "both loops hold a gate");

        // Loop 2 exits: only its own gate may be retired.
        env.regs.insert(3, 5);
        e.run_block(&loop2, &mut env);
        assert!(
            e.gating.iter().any(|(_, pc)| *pc == 0x204),
            "retiring loop 2 must not retire loop 1's gate"
        );
        assert!(
            !e.gating.iter().any(|(_, pc)| *pc == 0x304),
            "loop 2's own gate must be retired"
        );

        // Loop 1 keeps running, so its reads must still be bounded.
        env.regs.insert(3, 0);
        e.run_block(&loop1, &mut env);
        e.emit_observations();

        let source = AccessContext::site(0x100, 0x5800_0008);
        let target1 = AccessContext::site(0x200, 0x5800_0000);
        let bounded = e
            .ctrl_obs
            .get(&(source, target1))
            .map(|obs| obs.bounded_reads.clone())
            .unwrap_or_default();
        assert!(
            bounded.contains(&AccessContext::at(0x200, 0x5800_0000, 2)),
            "loop 1's read after loop 2 exited must still be bounded, got {bounded:?}"
        );
    }

    /// Two gates live at once on one source *site* are still one bound.
    ///
    /// This is the shape the dump showed: the `occ2` rows reported a
    /// `gated_count` of exactly twice `count` (274 against 140), because two back
    /// edges -- an inner and an outer loop, at different pcs -- both derive their
    /// condition from the same read, so both gates were live and every target read
    /// was appended to the same observation once per gate.  The gate key carries
    /// the branch pc, which is what lets the two be retired independently, so the
    /// deduplication has to happen where the read is recorded: a read is credited
    /// to a bound once, however many live gates state it.  That is also what makes
    /// `gated_count <= count` structural rather than a hope.
    #[test]
    fn two_gates_on_one_source_share_one_bound() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0008);
        e.pin_site(0x200, 0x5800_0000);

        // The shared source: read once, gating both loops.
        let mut p0 = pcode::Block::new();
        p0.push(marker(0x100));
        p0.push((reg(1), Op::Load(0), reg(10)));
        let length = lifter_block(p0, 0x100, 0x108, BlockExit::invalid());

        // The bounded site: read once per round.
        let mut p1 = pcode::Block::new();
        p1.push(marker(0x200));
        p1.push((reg(5), Op::Load(0), reg(8)));
        let target = lifter_block(p1, 0x200, 0x208, BlockExit::invalid());

        // Two back edges at different pcs, both conditioned on the same value.
        fn back_edge(pc: u64, fallthrough: u64) -> Block {
            let mut p = pcode::Block::new();
            p.push(marker(pc));
            p.push((reg(2), Op::IntLess, reg(3), reg(1)));
            lifter_block(
                p,
                pc,
                pc + 8,
                BlockExit::Branch {
                    cond: Value::Var(reg(2)),
                    target: Target::External(Value::Const(pc, 4)),
                    fallthrough: Target::External(Value::Const(fallthrough, 4)),
                },
            )
        }
        let inner = back_edge(0x300, 0x400);
        let outer = back_edge(0x400, 0x500);

        let mut env = MockEnv::new();
        env.regs.insert(10, 0x5800_0008);
        env.regs.insert(8, 0x5800_0000);
        env.regs.insert(3, 0);
        // Concrete `0 < 5`, so both back edges are taken and both gates open.
        env.regs.insert(1, 5);
        env.mem.insert(0x5800_0008, 5);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));

        e.run_block(&length, &mut env);
        for _ in 0..3 {
            e.run_block(&target, &mut env);
            e.run_block(&inner, &mut env);
            e.run_block(&outer, &mut env);
        }
        e.emit_observations();

        let source = AccessContext::site(0x100, 0x5800_0008);
        let bounded = AccessContext::site(0x200, 0x5800_0000);
        assert_eq!(
            recorded.borrow().loop_bounds,
            vec![(source, vec![1], bounded, 3, 2)],
            "one bound for the source site: 3 reads, 2 of them with both gates live"
        );
    }

    /// A tainted store inside the stack window is a spill, not a consumption.
    #[test]
    fn stack_spill_is_not_reported_as_a_sink() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0000);
        e.set_stack_pointer(VarNode::new(13, 4));
        e.stack_window = 0x100;

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        p.push((reg(1), Op::Load(0), reg(5)));
        // mult_addr = sp - 4; ram[mult_addr] = r1   (the `push` shape)
        p.push((reg(6), Op::IntSub, reg(13), Value::Const(4, 4)));
        p.push((VarNode::NONE, Op::Store(0), reg(6), reg(1)));
        let block = lifter_block(p, 0x100, 0x110, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(5, 0x5800_0000);
        env.regs.insert(13, 0x2000_1000);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));
        e.run_block(&block, &mut env);

        assert!(recorded.borrow().stores.is_empty(), "a spill must not be a sink");
        assert_eq!(e.counters().stack_stores_skipped, 1);
        // The value still flows through memory: filtering only stops the role.
        assert!(!e.shadow.mem_tag_range(0x2000_0ffc, 4).is_clean());
    }

    /// A prologue stores incoming arguments just *above* `sp`; that is a spill
    /// too and must not become a role.
    #[test]
    fn prologue_store_above_sp_is_a_spill() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0000);
        e.set_stack_pointer(VarNode::new(13, 4));
        e.stack_window = 0x100;
        e.stack_window_up = 0x40;

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        p.push((reg(1), Op::Load(0), reg(5)));
        // str r1,[sp,#0x8]
        p.push((reg(6), Op::IntAdd, reg(13), Value::Const(8, 4)));
        p.push((VarNode::NONE, Op::Store(0), reg(6), reg(1)));
        let block = lifter_block(p, 0x100, 0x110, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(5, 0x5800_0000);
        env.regs.insert(13, 0x2000_1000);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));
        e.run_block(&block, &mut env);

        assert!(recorded.borrow().stores.is_empty(), "a prologue spill is not a sink");
        assert_eq!(e.counters().stack_stores_skipped, 1);
    }

    /// An echo comparison: two input-derived values compared against each other.
    /// Neither is a value the firmware expects *from us*, so there is no magic
    /// constant here -- it is a verification, i.e. checksum evidence.
    #[test]
    fn echo_comparison_is_checksum_not_magic() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0000);
        e.pin_site(0x104, 0x5800_0004);

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        p.push((reg(1), Op::Load(0), reg(5)));
        p.push(marker(0x104));
        p.push((reg(2), Op::Load(0), reg(7)));
        p.push(marker(0x108));
        p.push((reg(3), Op::IntSub, reg(1), reg(2)));
        p.push((reg(4), Op::IntEqual, reg(3), Value::Const(0, 4)));
        let block = lifter_block(p, 0x100, 0x110, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(5, 0x5800_0000);
        env.regs.insert(7, 0x5800_0004);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));
        e.run_block(&block, &mut env);

        let recorded = recorded.borrow();
        assert!(recorded.magics.is_empty(), "an echo is not a magic constant");
        assert_eq!(recorded.checksums.len(), 1);
        assert_eq!(recorded.checksums[0].1, 0.5);
    }

    /// A bare variable-vs-variable equality is the same signal.
    #[test]
    fn var_vs_var_equality_is_checksum() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0000);
        e.pin_site(0x104, 0x5800_0004);

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        p.push((reg(1), Op::Load(0), reg(5)));
        p.push(marker(0x104));
        p.push((reg(2), Op::Load(0), reg(7)));
        p.push(marker(0x108));
        p.push((reg(3), Op::IntEqual, reg(1), reg(2)));
        let block = lifter_block(p, 0x100, 0x110, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(5, 0x5800_0000);
        env.regs.insert(7, 0x5800_0004);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));
        e.run_block(&block, &mut env);

        assert!(recorded.borrow().magics.is_empty());
        assert_eq!(recorded.borrow().checksums.len(), 1);
    }

    /// A hard-coded checksum comparison: `tmp = crc - 0x1234; tmp == 0`.  The
    /// constant is a checksum, not a protocol field, because the base is a mixed
    /// value rather than a plain input byte.
    #[test]
    fn hardcoded_checksum_comparison_is_not_magic() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0000);

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        p.push((reg(1), Op::Load(0), reg(5)));
        p.push((reg(2), Op::IntXor, reg(1), Value::Const(0xff, 4)));
        p.push((reg(3), Op::IntSub, reg(2), Value::Const(0x1234, 4)));
        p.push((reg(4), Op::IntEqual, reg(3), Value::Const(0, 4)));
        let block = lifter_block(p, 0x100, 0x110, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(5, 0x5800_0000);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));
        e.run_block(&block, &mut env);

        let recorded = recorded.borrow();
        assert!(recorded.magics.is_empty(), "a checksum constant is not a magic field");
        assert_eq!(recorded.checksums.len(), 1);
        assert_eq!(recorded.checksums[0].1, 0.6);
    }

    /// A partial mask comparison is a real discriminant: only a test for the mask
    /// itself (`x & 1 == 1`) is a mere position test.
    #[test]
    fn partial_mask_comparison_is_magic() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0000);

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        p.push((reg(1), Op::Load(0), reg(5)));
        p.push((reg(2), Op::IntAnd, reg(1), Value::Const(0xf0, 1)));
        p.push((reg(3), Op::IntEqual, reg(2), Value::Const(0xa0, 1)));
        let block = lifter_block(p, 0x100, 0x110, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(5, 0x5800_0000);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));
        e.run_block(&block, &mut env);

        let recorded = recorded.borrow();
        assert_eq!(recorded.magics.len(), 1, "& 0xf0 == 0xa0 is a protocol check");
        assert_eq!(recorded.magics[0].1, 0xa0);
    }

    /// The other side of the mask rule: masking a *checksum* does not turn it into
    /// a protocol constant.  `tmp = crc & 0xFFFF; tmp == 0x4321` is the final check
    /// of a checksum, and labelling those bytes magic would hand the mutator a
    /// constant it cannot successfully rewrite.
    #[test]
    fn masked_checksum_comparison_is_checksum() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0000);

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        p.push((reg(1), Op::Load(0), reg(5)));
        p.push((reg(2), Op::IntXor, reg(2), reg(1)));            // crc accumulates
        p.push((reg(3), Op::IntAnd, reg(2), Value::Const(0xffff, 4)));
        p.push((reg(4), Op::IntEqual, reg(3), Value::Const(0x4321, 4)));
        let block = lifter_block(p, 0x100, 0x110, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(5, 0x5800_0000);
        env.regs.insert(2, 0);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));
        e.run_block(&block, &mut env);

        let recorded = recorded.borrow();
        assert!(
            recorded.magics.is_empty(),
            "masking a checksum does not make it a protocol constant"
        );
        assert_eq!(recorded.checksums.len(), 1);
    }

    /// Guard for the constant form: `cmp r, #imm` lifts to a subtraction whose
    /// "other" operand *is* the base, so the input-vs-input test must not be
    /// applied to it.  If it were, every real magic comparison would be
    /// downgraded to checksum and magic detection would silently vanish.
    #[test]
    fn constant_form_comparison_stays_magic() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0000);

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        p.push((reg(1), Op::Load(0), reg(5)));
        p.push((reg(2), Op::IntSub, reg(1), Value::Const(0xaa, 4)));
        p.push((reg(3), Op::IntEqual, reg(2), Value::Const(0, 4)));
        let block = lifter_block(p, 0x100, 0x110, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(5, 0x5800_0000);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));
        e.run_block(&block, &mut env);

        let recorded = recorded.borrow();
        let source = AccessContext::new(0x100, 0x5800_0000);
        assert_eq!(recorded.magics, vec![(vec![source], 0xaa, 0.85)]);
        assert!(recorded.checksums.is_empty(), "a constant comparison is not a checksum");
    }

    /// A checksum accumulated inside a loop is compared after the loop exits, in a
    /// different group.  The "this value is mixed" marker has to survive that
    /// boundary, or the final comparison is misread as a magic constant.
    #[test]
    fn cross_block_checksum_comparison_is_not_magic() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0000);

        // Group 1: the loop body mixes into a register accumulator.
        let mut p1 = pcode::Block::new();
        p1.push(marker(0x100));
        p1.push((reg(1), Op::Load(0), reg(5)));
        p1.push((reg(2), Op::IntXor, reg(2), reg(1)));
        let b1 = lifter_block(p1, 0x100, 0x108, BlockExit::invalid());

        // Group 2: after the loop, compare the accumulator against a constant.
        let mut p2 = pcode::Block::new();
        p2.push(marker(0x200));
        p2.push((reg(3), Op::IntSub, reg(2), Value::Const(0x1234, 4)));
        p2.push((reg(4), Op::IntEqual, reg(3), Value::Const(0, 4)));
        let b2 = lifter_block(p2, 0x200, 0x208, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(5, 0x5800_0000);
        env.regs.insert(2, 0);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));
        e.run_block(&b1, &mut env);
        e.run_block(&b2, &mut env);

        let recorded = recorded.borrow();
        assert!(
            recorded.magics.is_empty(),
            "a cross-block checksum comparison must not become magic"
        );
        assert_eq!(recorded.checksums.len(), 1);
    }

    /// The mirror image of the test above: a register that once held a checksum is
    /// *not* still a checksum after being reloaded with fresh input.
    ///
    /// Registers are reused constantly at -O2, and a CRC accumulator and a
    /// separator comparison often share the same low register number; a marker
    /// that never expires would silently delete magic evidence for every later
    /// comparison on that register.
    #[test]
    fn register_reuse_clears_mixed() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0000);
        e.pin_site(0x200, 0x5800_0004);

        // Block 1: r2 accumulates a checksum.
        let mut p1 = pcode::Block::new();
        p1.push(marker(0x100));
        p1.push((reg(1), Op::Load(0), reg(5)));
        p1.push((reg(2), Op::IntXor, reg(2), reg(1)));
        let b1 = lifter_block(p1, 0x100, 0x108, BlockExit::invalid());

        // Block 2: the same register is reloaded with fresh input.
        let mut p2 = pcode::Block::new();
        p2.push(marker(0x200));
        p2.push((reg(2), Op::Load(0), reg(7)));
        let b2 = lifter_block(p2, 0x200, 0x208, BlockExit::invalid());

        // Block 3: compare it against the '\r' separator.
        let mut p3 = pcode::Block::new();
        p3.push(marker(0x300));
        p3.push((reg(3), Op::IntSub, reg(2), Value::Const(0x0d, 4)));
        p3.push((reg(4), Op::IntEqual, reg(3), Value::Const(0, 4)));
        let b3 = lifter_block(p3, 0x300, 0x308, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(5, 0x5800_0000);
        env.regs.insert(7, 0x5800_0004);
        env.regs.insert(2, 0);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));
        e.run_block(&b1, &mut env);
        e.run_block(&b2, &mut env);
        e.run_block(&b3, &mut env);

        let recorded = recorded.borrow();
        let fresh = AccessContext::new(0x200, 0x5800_0004);
        assert_eq!(
            recorded.magics,
            vec![(vec![fresh], 0x0d, 0.85)],
            "a reloaded register is fresh input, not a checksum"
        );
        assert!(recorded.checksums.is_empty());
    }

    /// `count` is the number of reads of the target site over the whole pass, not
    /// the number that happened while the gate was in effect.
    ///
    /// In a bottom-tested loop (`body; subs; bne top`) the first read precedes the
    /// branch that establishes the gate, so the gated count is one lower than the
    /// length field's value.  Reporting the gated count would make the
    /// value-equals-count check fail systematically on exactly the loop shape the
    /// dump shows.
    #[test]
    fn loop_bound_count_is_the_site_total() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0008);
        e.pin_site(0x200, 0x5800_0000);

        // The length field is read once, before the loop.
        let mut p0 = pcode::Block::new();
        p0.push(marker(0x100));
        p0.push((reg(1), Op::Load(0), reg(9)));
        let b0 = lifter_block(p0, 0x100, 0x108, BlockExit::invalid());

        let mut p1 = pcode::Block::new();
        p1.push(marker(0x200));
        p1.push((reg(5), Op::Load(0), reg(8)));
        let b1 = lifter_block(p1, 0x200, 0x208, BlockExit::invalid());

        let mut p2 = pcode::Block::new();
        p2.push(marker(0x300));
        p2.push((reg(2), Op::IntLess, reg(3), reg(1)));
        let exit2 = BlockExit::Branch {
            cond: Value::Var(reg(2)),
            target: Target::External(Value::Const(0x200, 4)),
            fallthrough: Target::External(Value::Const(0x400, 4)),
        };
        let b2 = lifter_block(p2, 0x300, 0x308, exit2);

        let mut env = MockEnv::new();
        env.regs.insert(9, 0x5800_0008);
        env.regs.insert(8, 0x5800_0000);
        // Concrete inputs for the branch: counter 0 < length 5 keeps the loop
        // going, so the gate is established from the first branch onwards.
        env.regs.insert(3, 0);
        env.regs.insert(1, 5);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));

        e.run_block(&b0, &mut env);
        for _ in 0..3 {
            e.run_block(&b1, &mut env);
            e.run_block(&b2, &mut env);
        }
        e.emit_observations();

        let source = AccessContext::site(0x100, 0x5800_0008);
        let target = AccessContext::site(0x200, 0x5800_0000);
        assert_eq!(
            recorded.borrow().loop_bounds,
            vec![(source, vec![1], target, 3, 2)],
            "count is the site total (3); only 2 reads happened after the gate existed"
        );
    }

    /// Once the loop exits, the gate must stop bounding reads: otherwise every
    /// later read of every stream is labelled payload.
    #[test]
    fn loop_exit_ends_the_bound() {
        let mut e = engine();
        e.pin_site(0x200, 0x5800_0008);
        e.pin_site(0x300, 0x5800_0000);
        e.pin_site(0x400, 0x5800_000c);

        let mut p1 = pcode::Block::new();
        p1.push(marker(0x200));
        p1.push((reg(1), Op::Load(0), reg(9)));            // bound value
        p1.push((reg(2), Op::IntLess, reg(3), reg(1)));    // counter < bound
        let exit1 = BlockExit::Branch {
            cond: Value::Var(reg(2)),
            target: Target::External(Value::Const(0x200, 4)),   // loop back
            fallthrough: Target::External(Value::Const(0x400, 4)),
        };
        let b1 = lifter_block(p1, 0x200, 0x208, exit1);

        let mut p2 = pcode::Block::new();
        p2.push(marker(0x300));
        p2.push((reg(5), Op::Load(0), reg(8)));
        let b2 = lifter_block(p2, 0x300, 0x304, BlockExit::invalid());

        let mut p3 = pcode::Block::new();
        p3.push(marker(0x400));
        p3.push((reg(6), Op::Load(0), reg(10)));
        let b3 = lifter_block(p3, 0x400, 0x404, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(9, 0x5800_0008);
        env.regs.insert(8, 0x5800_0000);
        env.regs.insert(10, 0x5800_000c);
        env.mem.insert(0x5800_0008, 5);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));

        // Iteration 1: the loop continues, so the read of the second stream is
        // bounded.  Iteration 2: the loop exits, so nothing after it is bounded.
        env.regs.insert(3, 0);
        e.run_block(&b1, &mut env);
        e.run_block(&b2, &mut env);
        env.regs.insert(3, 5);
        e.run_block(&b1, &mut env);
        e.run_block(&b2, &mut env);
        e.run_block(&b3, &mut env);
        e.emit_observations();

        // The observation is keyed by sites, so the probe keys are sites too.
        let source = AccessContext::site(0x200, 0x5800_0008);
        let bounded = AccessContext::site(0x300, 0x5800_0000);
        let after = AccessContext::site(0x400, 0x5800_000c);
        let observations = &e.ctrl_obs;
        assert_eq!(
            observations.get(&(source, bounded)).map(|obs| obs.bounded_reads.len()),
            Some(1),
            "only the in-loop read is bounded"
        );
        assert!(
            observations.get(&(source, after)).is_none(),
            "a read after the loop exit must not be bounded"
        );
    }

    /// `run_block` must not panic when an `InstructionMarker` has a non-constant
    /// first input.  This can occur with synthetic ARM/Thumb blocks where the
    /// SLEIGH lifter or a rewriter emits an InstructionMarker with a `VarNode`
    /// address.
    #[test]
    fn run_block_does_not_panic_on_non_const_instruction_marker() {
        let mut e = engine();

        let malformed_marker = pcode::Instruction {
            op: pcode::Op::InstructionMarker,
            inputs: pcode::Inputs::new(
                pcode::Value::Var(VarNode::new(5, 8)),
                pcode::Value::Const(4, 8),
            ),
            output: VarNode::NONE,
        };
        let mut p = pcode::Block::new();
        p.instructions.push(malformed_marker);
        let block = lifter_block(p, 0xdead_0000, 0xdead_0004, BlockExit::invalid());

        // Must not panic; cur_pc should stay at block.start (0xdead_0000).
        e.run_block(&block, &mut MockEnv::new());
    }

    /// A call (PcodeOp) clears argument-register taint so it cannot leak across
    /// opaque calls.
    #[test]
    fn call_kills_argument_register_taint() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0000);

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        p.push((reg(1), Op::Load(0), reg(5)));
        p.push(marker(0x104));
        p.push((VarNode::NONE, Op::PcodeOp(0), reg(1)));
        let block = lifter_block(p, 0x100, 0x108, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(5, 0x5800_0000);
        e.run_block(&block, &mut env);

        assert!(e.shadow.reg_tag(1).is_clean(), "r1 taint must be killed after an opaque call");
    }

    /// Build a 3-sub-block group mirroring an internal-branch instruction (e.g. an
    /// ARM IT-block / divide zero-check):
    ///   L0 (entry, marker 0x100): r10 = LOAD[mmioA];  r2 = (r10 == 5)
    ///       Branch r2 -> L1 (taken)  else -> L2
    ///   L1 (marker 0x104):  r7 = r4 + r10;  r1 = LOAD[r7]   (B's MMIO read)
    ///   L2 (marker 0x108):  (fallthrough, empty)
    fn it_block_group(a_val: u64) -> (Vec<Block>, MockEnv) {
        let mmio_a = 0x5800_0000u64;
        let b_base = 0x2000_0000u64; // regular RAM, not MMIO

        let mut p0 = pcode::Block::new();
        p0.push(marker(0x100));
        p0.push((reg(10), Op::Load(0), reg(5)));
        p0.push((reg(2), Op::IntEqual, reg(10), Value::Const(5, 4)));
        let b0 = lifter_block(
            p0,
            0x100,
            0x104,
            BlockExit::Branch {
                cond: Value::Var(reg(2)),
                target: Target::Internal(1),
                fallthrough: Target::Internal(2),
            },
        );

        let mut p1 = pcode::Block::new();
        p1.push(marker(0x104));
        p1.push((reg(7), Op::IntAdd, reg(4), reg(10)));
        p1.push((reg(1), Op::Load(0), reg(7)));
        let b1 = lifter_block(p1, 0x104, 0x108, BlockExit::Jump { target: Target::Internal(2) });

        let mut p2 = pcode::Block::new();
        p2.push(marker(0x108));
        let b2 = lifter_block(p2, 0x108, 0x10c, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(5, mmio_a);
        env.regs.insert(4, b_base);
        env.mem.insert(mmio_a, a_val);
        (vec![b0, b1, b2], env)
    }

    /// `run_group` must interpret a non-entry sub-block reached by a taken
    /// internal branch, so a read site living there is actually observed.
    #[test]
    fn group_replay_observes_taken_internal_subblock() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0000);
        e.pin_site(0x104, 0x5800_0004);

        let (blocks, mut env) = it_block_group(5);
        e.run_group(&blocks, 0, &mut env);

        assert_eq!(e.read_count(AccessContext::new(0x104, 0x5800_0004)), 1);
    }

    /// `run_group` must NOT interpret the untaken sub-block.
    #[test]
    fn group_replay_skips_untaken_internal_subblock() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0000);
        e.pin_site(0x104, 0x5800_0004);

        let (blocks, mut env) = it_block_group(7);
        e.run_group(&blocks, 0, &mut env);

        assert_eq!(
            e.read_count(AccessContext::new(0x104, 0x5800_0004)),
            0,
            "untaken sub-block must not be interpreted"
        );
    }

    /// Acceptance 1: instrumentation and real analysis coexist in one block.
    ///
    /// The block contains the coverage bitmap update exactly as the p-code dump
    /// shows it (`mem.2[0x0:8]` read-modify-write, i.e. a *clean* store into the
    /// trace space) *and* a real MMIO load that is compared against a constant.
    /// The instrumentation must neither wipe guest taint that shares the shadow
    /// key nor produce any observation, while the real taint must survive and
    /// its constant must be recovered.
    #[test]
    fn instrumentation_cannot_touch_guest_taint() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0000);

        // Pre-existing taint at guest address 0: the address the coverage
        // bitmap write would collide with if spaces were not distinguished.
        let probe = AccessContext::new(0x50, 0x5800_0004);
        let guest_tag = e.shadow.source_tag(probe);
        e.shadow.set_mem_tag_range(0, 8, guest_tag.clone());

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        // $U2:1 = mem.2[0x0:8]; $U2:1 = $U2:1 | 0x1; mem.2[0x0:8] = $U2:1
        p.push((reg(20), Op::Load(2), Value::Const(0, 8)));
        p.push((reg(20), Op::IntOr, reg(20), Value::Const(1, 1)));
        p.push((VarNode::NONE, Op::Store(2), Value::Const(0, 8), reg(20)));
        // counter = counter - 1; if (counter == 0) hook.1()
        p.push((reg(21), Op::IntSub, reg(21), Value::Const(1, 8)));
        p.push((reg(22), Op::IntEqual, reg(21), Value::Const(0, 8)));
        p.push((VarNode::NONE, Op::HookIf(1), reg(22)));
        // The real work: read the input register and compare it against 0x6.
        p.push((reg(1), Op::Load(0), reg(5)));
        p.push((reg(2), Op::IntSub, reg(1), Value::Const(6, 4)));
        p.push((reg(3), Op::IntEqual, reg(2), Value::Const(0, 4)));

        let block = lifter_block(p, 0x100, 0x110, BlockExit::invalid());
        let mut env = MockEnv::new();
        env.regs.insert(5, 0x5800_0000);
        env.regs.insert(21, 5);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));
        e.run_block(&block, &mut env);
        e.flush_pending_source_loads();

        // The instrumentation was skipped, not interpreted.
        assert!(e.counters().skipped_space_ops >= 2, "space filter must trip");
        // ...and it did not wipe the guest taint that shares shadow key 0.
        assert_eq!(e.shadow.mem_tag_range(0, 8), guest_tag, "instrumentation wiped guest taint");

        let recorded = recorded.borrow();
        // No observation came from the instrumentation: the only fragment and the
        // only magic belong to the real read site.
        let source = AccessContext::new(0x100, 0x5800_0000);
        assert_eq!(recorded.loads, vec![(source, 4)]);
        assert_eq!(recorded.magics, vec![(vec![source], 6, 0.85)]);
        assert!(recorded.loop_bounds.is_empty());
    }

    /// Acceptance 2: the real `cmp` shape.
    ///
    /// `cmp r3, #6` lifts to `tmp = r3 - 6; ZR = tmp == 0`; the constant lives in
    /// the subtraction.  When the other operand is a variable loaded from RAM
    /// (the literal-pool form ARM uses for constants it cannot encode), the value
    /// is only known concretely, so the result is reported with lower confidence.
    #[test]
    fn real_cmp_shape_yields_the_compared_constant() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0000);

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        p.push((reg(1), Op::Load(0), reg(5)));
        p.push(marker(0x104));
        p.push((reg(7), Op::Load(0), Value::Const(0x1060, 4)));
        p.push(marker(0x108));
        p.push((reg(2), Op::IntSub, reg(1), reg(7)));
        p.push((reg(3), Op::IntEqual, reg(2), Value::Const(0, 4)));

        let block = lifter_block(p, 0x100, 0x110, BlockExit::invalid());
        let mut env = MockEnv::new();
        env.regs.insert(5, 0x5800_0000);
        env.mem.insert(0x1060, 0xa5);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));
        e.run_block(&block, &mut env);
        e.flush_pending_source_loads();

        let source = AccessContext::new(0x100, 0x5800_0000);
        assert_eq!(
            recorded.borrow().magics,
            vec![(vec![source], 0xa5, 0.6)],
            "literal-pool comparison must report the concrete value at reduced confidence"
        );
    }

    /// A flag update whose operand is clean must not be mistaken for a comparison
    /// against zero.  This is the gate that removes the ~2800 `== 0` forms the
    /// dump contains, without any instrumentation-specific blacklist.
    #[test]
    fn clean_flag_update_reports_no_magic() {
        let mut e = engine();

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        p.push((reg(1), Op::Copy, Value::Const(0x11, 4)));
        p.push((reg(2), Op::IntSub, reg(1), Value::Const(6, 4)));
        p.push((reg(3), Op::IntEqual, reg(2), Value::Const(0, 4)));
        let block = lifter_block(p, 0x100, 0x110, BlockExit::invalid());

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));
        e.run_block(&block, &mut MockEnv::new());

        assert!(recorded.borrow().magics.is_empty());
    }

    /// A bit test (`value & 1 == 1`) is not a protocol constant.
    #[test]
    fn bit_test_is_not_a_magic_value() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0000);

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        p.push((reg(1), Op::Load(0), reg(5)));
        p.push((reg(2), Op::IntAnd, reg(1), Value::Const(1, 4)));
        p.push((reg(3), Op::IntEqual, reg(2), Value::Const(1, 4)));
        let block = lifter_block(p, 0x100, 0x110, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(5, 0x5800_0000);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));
        e.run_block(&block, &mut env);

        assert!(recorded.borrow().magics.is_empty(), "a mask test is not a magic constant");
    }

    /// A table lookup indexed by input must carry the address's provenance,
    /// otherwise the chain (CRC-style table) dies at the load.
    #[test]
    fn tainted_table_index_propagates_through_the_load() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0000);

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        p.push((reg(1), Op::Load(0), reg(5)));            // r1 = input byte
        p.push(marker(0x104));
        p.push((reg(6), Op::Copy, Value::Const(0x2000, 4))); // table base
        p.push((reg(7), Op::IntAdd, reg(6), reg(1)));     // entry = base + input
        p.push(marker(0x108));
        p.push((reg(8), Op::Load(0), reg(7)));            // r8 = table[input]
        let block = lifter_block(p, 0x100, 0x110, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(5, 0x5800_0000);
        env.mem.insert(0x2000, 0x5a);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));
        e.run_block(&block, &mut env);

        assert_eq!(recorded.borrow().table_loads, 1, "the indexed load must be seen as a table load");
        assert!(!e.taint_in(Value::Var(reg(8))).is_clean(), "table result must stay tainted");
        assert_eq!(e.counters().table_loads, 1);
    }

    /// A tainted value that was mixed before being stored is checksum-like, not a
    /// payload byte.
    #[test]
    fn mixed_store_is_checksum_not_payload() {
        let mut e = engine();
        e.pin_site(0x100, 0x5800_0000);

        let mut p = pcode::Block::new();
        p.push(marker(0x100));
        p.push((reg(1), Op::Load(0), reg(5)));
        p.push((reg(2), Op::Copy, Value::Const(0, 4)));
        p.push((reg(2), Op::IntXor, reg(2), reg(1)));     // mix
        p.push((VarNode::NONE, Op::Store(0), reg(6), reg(2)));
        let block = lifter_block(p, 0x100, 0x110, BlockExit::invalid());

        let mut env = MockEnv::new();
        env.regs.insert(5, 0x5800_0000);
        env.regs.insert(6, 0x2000_0000);

        let (observer, recorded) = Recorder::new();
        e.set_observer(Box::new(observer));
        e.run_block(&block, &mut env);

        let recorded = recorded.borrow();
        assert_eq!(recorded.checksums.len(), 1, "a mixed store is checksum evidence");
        assert!(recorded.stores.is_empty(), "and must not also be reported as a sink");
    }
}
