//! Role map: the stable data contract between the analysis pass and mutation.
//!
//! The role map answers one question, per stream offset range:
//!
//! > What does the firmware *do* with these input bytes?
//!
//! It is produced by the dynamic taint pass ([`crate::phase_b`]) and is
//! deliberately kept separate from the block/section structure
//! ([`DataBlock`]) so the mutation side can be written against a contract that
//! does not change when the analysis internals do.
//!
//! The analysis runs entirely out of band (replay), never in the fuzzing loop.

use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, HashSet, VecDeque},
    env,
    rc::Rc,
    sync::{Arc, Mutex},
};

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use crate::{
    debugging::trace::IoTracer,
    input::StreamKey,
    phase_b::{MagicEvidence, PhaseBObserver},
    taint::AccessContext,
};

/// How the firmware consumes a range of input bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Compared against a constant that selects a protocol path.
    Magic,
    /// Bounds a later copy or read loop.
    Length,
    /// Bulk data copied into a buffer.
    Payload,
    /// Accumulated into a verification value.
    Checksum,
    /// Stored into a control/configuration register.
    Config,
    /// Consumed but not observable through a classified sink.
    Propagated,
}

/// One entry of the role map.
///
/// This is the frozen contract: `stream` and `offset_range` locate the bytes,
/// `role` says how they are consumed, `confidence` says how much the analysis
/// trusts that, and `discriminants` carries the equality-gated values the
/// firmware compared those bytes against (empty when the role is not gated on
/// a specific value).
///
/// `stream_level` marks the cases where the discriminant is a property of the
/// whole stream rather than of this position (see `StreamConstraint`): the byte
/// is subject to a delimiter test, so a mutator must not treat the constant as
/// the value this position has to hold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoleEntry {
    pub stream: StreamKey,
    pub offset_range: (u32, u32),
    pub role: Role,
    pub confidence: f32,
    pub discriminants: Vec<u64>,
    pub stream_level: bool,
}

impl RoleEntry {
    pub fn new(
        stream: StreamKey,
        offset_range: (u32, u32),
        role: Role,
        confidence: f32,
    ) -> Self {
        Self {
            stream,
            offset_range,
            role,
            confidence,
            discriminants: Vec::new(),
            stream_level: false,
        }
    }

    pub fn len(&self) -> u32 {
        self.offset_range.1.saturating_sub(self.offset_range.0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&self, offset: u32) -> bool {
        self.offset_range.0 <= offset && offset < self.offset_range.1
    }

    fn overlaps(&self, other: &RoleEntry) -> bool {
        self.stream == other.stream
            && self.offset_range.0 < other.offset_range.1
            && other.offset_range.0 < self.offset_range.1
    }
}

/// The complete role map: what the mutation side consumes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RoleMap {
    pub entries: Vec<RoleEntry>,
}

impl RoleMap {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Insert an entry, merging with an existing entry for the same role and
    /// overlapping range instead of producing duplicates.  Merging takes the
    /// strongest confidence and unions the discriminants.
    pub fn insert(&mut self, entry: RoleEntry) {
        if entry.is_empty() {
            return;
        }
        for existing in &mut self.entries {
            if existing.role != entry.role || !existing.overlaps(&entry) {
                continue;
            }
            existing.offset_range.0 = existing.offset_range.0.min(entry.offset_range.0);
            existing.offset_range.1 = existing.offset_range.1.max(entry.offset_range.1);
            existing.confidence = existing.confidence.max(entry.confidence);
            for value in entry.discriminants {
                if !existing.discriminants.contains(&value) {
                    existing.discriminants.push(value);
                }
            }
            existing.discriminants.sort_unstable();
            return;
        }
        self.entries.push(entry);
    }

    pub fn entries_for_stream(&self, stream: StreamKey) -> impl Iterator<Item = &RoleEntry> {
        self.entries.iter().filter(move |entry| entry.stream == stream)
    }

    pub fn streams(&self) -> Vec<StreamKey> {
        let mut streams: Vec<StreamKey> =
            self.entries.iter().map(|entry| entry.stream).collect();
        streams.sort_unstable();
        streams.dedup();
        streams
    }

    /// The role covering `offset`, preferring the highest-confidence match.
    ///
    /// Not called yet: this is the query the mutation stage will use to look up a
    /// byte's contract, so the dead-code lint is silenced here rather than
    /// project-wide (a genuinely dead helper must stay visible).
    #[allow(dead_code)]
    pub fn role_at(&self, stream: StreamKey, offset: u32) -> Option<&RoleEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.stream == stream && entry.contains(offset))
            .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
    }

    /// Derive the structural section view: contiguous runs of consumed offsets
    /// per stream.  Roles stay in the map, so a block only describes *where* the
    /// firmware reads, not *how*.
    pub fn blocks(&self) -> Vec<DataBlock> {
        let mut blocks = Vec::new();
        for stream in self.streams() {
            let mut ranges: Vec<(u32, u32)> = self
                .entries_for_stream(stream)
                .map(|entry| entry.offset_range)
                .collect();
            ranges.sort_unstable();

            let mut merged: Vec<(u32, u32)> = Vec::new();
            for range in ranges {
                // Small gaps are still one logical section: UART/FIFO loops read
                // a contiguous payload in small strides.
                let extend = match merged.last() {
                    Some(last) => range.0 <= last.1.saturating_add(8),
                    None => false,
                };
                if extend {
                    let last = merged.last_mut().expect("checked above");
                    last.1 = last.1.max(range.1);
                }
                else {
                    merged.push(range);
                }
            }

            for range in merged {
                blocks.push(DataBlock {
                    id: blocks.len() as u64,
                    stream,
                    offset_range: range,
                });
            }
        }
        blocks
    }

}

/// A contiguous section of a stream that the firmware consumes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataBlock {
    pub id: u64,
    pub stream: StreamKey,
    pub offset_range: (u32, u32),
}

/// One MMIO read observed by the tracer pipeline.
#[derive(Debug, Clone, Serialize)]
pub struct ReadRecord {
    pub seq: u64,
    pub site_pc: u64,
    pub stream: StreamKey,
    pub input_offset: u32,
    pub size: u8,
    pub value: u64,
    pub icount: u64,
    pub active_irq: u16,
}

impl ReadRecord {
    fn end(&self) -> u32 {
        self.input_offset.saturating_add(self.size as u32)
    }
}

/// Ordered record of every MMIO read, keyed by read site.
///
/// This is the provenance backbone: the dynamic pass identifies *which* read
/// site supplied a value, and this ledger turns that into the exact input byte
/// range it consumed.  Matching is positional, never value-based, so equal byte
/// values can no longer be confused with each other.
#[derive(Clone)]
pub struct ReadLedger {
    // `Arc<Mutex<_>>` (not `Rc<RefCell<_>>`) because the ledger is handed to an
    // `IoTracerAny`, which requires `Send + Sync`.
    per_site: Arc<Mutex<HashMap<(u64, StreamKey), VecDeque<ReadRecord>>>>,
    order: Arc<Mutex<Vec<ReadRecord>>>,
    next_seq: Arc<Mutex<u64>>,
    limit: usize,
}

impl Default for ReadLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl ReadLedger {
    pub fn new() -> Self {
        let limit =
            env::var("TAINT_READ_LIMIT").ok().and_then(|x| x.parse().ok()).unwrap_or(2_000_000);
        Self {
            per_site: Arc::new(Mutex::new(HashMap::new())),
            order: Arc::new(Mutex::new(Vec::new())),
            next_seq: Arc::new(Mutex::new(0)),
            limit,
        }
    }

    fn record(
        &self,
        site_pc: u64,
        stream: StreamKey,
        input_offset: u32,
        value: &[u8],
        context: icicle_cortexm::mmio::ReadContext,
    ) {
        if self.order.lock().expect("read ledger poisoned").len() >= self.limit {
            return;
        }
        let seq = {
            let mut next_seq = self.next_seq.lock().expect("read ledger poisoned");
            let seq = *next_seq;
            *next_seq += 1;
            seq
        };
        let record = ReadRecord {
            seq,
            site_pc,
            stream,
            input_offset,
            size: value.len().min(u8::MAX as usize) as u8,
            value: encode_value(value),
            // Kept because aligning a parent execution against a mutated one needs a
            // total order over reads, and the input offset alone is not one (two
            // sites share a stream, and one site is read repeatedly).
            icount: context.icount,
            active_irq: context.active_irq,
        };

        self
            .per_site
            .lock()
            .expect("read ledger poisoned")
            .entry((site_pc, stream))
            .or_default()
            .push_back(record.clone());
        self.order.lock().expect("read ledger poisoned").push(record);
    }

    pub fn records(&self) -> Vec<ReadRecord> {
        self.order.lock().expect("read ledger poisoned").clone()
    }

    pub fn len(&self) -> usize {
        self.order.lock().expect("read ledger poisoned").len()
    }

    /// Group the observed sites by PC.
    fn streams_by_pc(&self) -> HashMap<u64, HashSet<StreamKey>> {
        let per_site = self.per_site.lock().expect("read ledger poisoned");
        let mut by_pc: HashMap<u64, HashSet<StreamKey>> = HashMap::new();
        for (site_pc, stream) in per_site.keys() {
            by_pc.entry(*site_pc).or_default().insert(*stream);
        }
        by_pc
    }

    /// Read sites that served more than one stream.
    ///
    /// Kept for the audit trail only.  Every binding in this module is keyed by
    /// `(pc, stream)`, so a PC that serves several peripherals is split into one
    /// pseudo-site per stream and nothing is lost; a non-empty list is a property
    /// of the firmware (a shared `read_byte()` helper, or a register sweep over a
    /// bank of ports), not a limitation of the pass.
    pub fn multiplexed_read_sites(&self) -> Vec<u64> {
        let mut pcs: Vec<u64> = self
            .streams_by_pc()
            .into_iter()
            .filter(|(_, streams)| streams.len() > 1)
            .map(|(pc, _)| pc)
            .collect();
        pcs.sort_unstable();
        pcs
    }

    /// PC -> every stream it was seen reading.
    ///
    /// All of them, including those of a multiplexed PC: that is what the engine
    /// resolves per read, and what makes each `(pc, stream)` its own pseudo-site
    /// with its own occurrence sequence.  Sorted, so the map does not depend on
    /// hash iteration order.
    pub fn read_sites(&self) -> HashMap<u64, Vec<StreamKey>> {
        let mut sites = HashMap::new();
        for (pc, streams) in self.streams_by_pc() {
            let mut streams: Vec<StreamKey> = streams.into_iter().collect();
            streams.sort_unstable();
            sites.insert(pc, streams);
        }
        sites
    }

    pub fn site_count(&self) -> usize {
        self.per_site.lock().expect("read ledger poisoned").len()
    }

    /// Drop every recorded read.  Used between the discovery pass and the
    /// analysis pass so positional matching only ever sees one execution.
    pub fn clear(&self) {
        self.per_site.lock().expect("read ledger poisoned").clear();
        self.order.lock().expect("read ledger poisoned").clear();
        *self.next_seq.lock().expect("read ledger poisoned") = 0;
    }

    /// Positionally take the next read for `site`, skipping records that the
    /// pass already consumed.  Sites are read strictly forward, so a record
    /// before the expected offset is a stale leftover from a partially observed
    /// block and can be dropped.
    fn take_next(&self, site_pc: u64, stream: StreamKey, expected: u32) -> Option<ReadRecord> {
        let mut per_site = self.per_site.lock().expect("read ledger poisoned");
        let queue = per_site.get_mut(&(site_pc, stream))?;
        while let Some(front) = queue.front() {
            if front.input_offset < expected {
                queue.pop_front();
            }
            else {
                break;
            }
        }
        queue.pop_front()
    }

    /// Reads that were recorded but never consumed by a fragment binding.
    ///
    /// A non-zero value on its own is expected: reads after the last interpreted
    /// group are never bound.  It only indicates a misalignment when the pass also
    /// stopped early, because then the site's queue has been shifted.
    pub fn pending(&self) -> usize {
        self.per_site
            .lock()
            .expect("read ledger poisoned")
            .values()
            .map(|queue| queue.len())
            .sum()
    }

    /// Split the unclaimed reads into the ones whose site the pass could not
    /// represent and the ones left over by interpretation stopping early.
    ///
    /// The first kind is an audit signal rather than a limitation: every
    /// `(pc, stream)` the discovery pass saw is representable, so a read whose
    /// pair is missing from `read_sites` means the two passes disagreed about the
    /// site (a non-deterministic address, or a site found only in the second
    /// pass).  Only the second kind means the interpreter and the CPU took
    /// different paths through the program.
    pub fn pending_split(&self, read_sites: &HashMap<u64, Vec<StreamKey>>) -> (usize, usize) {
        let per_site = self.per_site.lock().expect("read ledger poisoned");
        let mut unrepresented = 0;
        let mut leftovers = 0;
        for ((pc, stream), queue) in per_site.iter() {
            let represented =
                read_sites.get(pc).map_or(false, |streams| streams.contains(stream));
            if represented {
                leftovers += queue.len();
            }
            else {
                unrepresented += queue.len();
            }
        }
        (unrepresented, leftovers)
    }

    /// Total input bytes delivered per stream, as the model consumed them.
    ///
    /// This is the budget split.  The analysis is what tells protocol bytes apart
    /// from the bytes the firmware only needed in order to read its own device
    /// state: on the target, one stream carried the console text and five carried
    /// a polled status register or a clock/GPIO configuration value.
    pub fn bytes_by_stream(&self) -> HashMap<StreamKey, u64> {
        let order = self.order.lock().expect("read ledger poisoned");
        let mut bytes: HashMap<StreamKey, u64> = HashMap::new();
        for record in order.iter() {
            *bytes.entry(record.stream).or_insert(0) += record.size as u64;
        }
        bytes
    }
}

fn encode_value(bytes: &[u8]) -> u64 {
    let mut encoded = 0_u64;
    for (index, byte) in bytes.iter().take(8).enumerate() {
        encoded |= (*byte as u64) << (index * 8);
    }
    encoded
}

/// Feeds the ledger from the existing MultiStream tracer pipeline.
#[derive(Clone)]
pub struct ReadLedgerTracer {
    ledger: ReadLedger,
}

impl ReadLedgerTracer {
    pub fn new(ledger: ReadLedger) -> Self {
        Self { ledger }
    }
}

impl IoTracer for ReadLedgerTracer {
    fn read(
        &mut self,
        addr: StreamKey,
        input_offset: u32,
        value: &[u8],
        context: icicle_cortexm::mmio::ReadContext,
    ) {
        self.ledger.record(context.pc, addr, input_offset, value, context);
    }

    fn snapshot(&self) -> Box<dyn std::any::Any> {
        Box::new(self.clone())
    }

    fn restore(&mut self, _snapshot: &Box<dyn std::any::Any>) {}
}

/// A fragment of input that a specific read site consumed.
#[derive(Debug, Clone, Serialize)]
pub struct SourceFragment {
    pub site_pc: u64,
    pub stream: StreamKey,
    pub offset_range: (u32, u32),
    pub value: u64,
    pub icount: u64,
}

/// Sink roles the classifier can attach to a store.
///
/// The default table is empty: roles are either supplied per target through
/// `TAINT_SINK_MAP` or inferred from the relation graph.  Hard-coding one
/// firmware's addresses into the analysis would make the result meaningless for
/// every other target.
#[derive(Debug, Clone, Default)]
pub struct SinkClassifier {
    by_pc: HashMap<u64, Role>,
}

impl SinkClassifier {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build the classifier from `TAINT_SINK_MAP`, a comma-separated list of
    /// `0x<pc>=<role>` entries, e.g. `0x81d24=payload,0x825a8=payload`.
    pub fn from_env() -> Self {
        let mut classifier = Self::new();
        let Ok(spec) = env::var("TAINT_SINK_MAP") else {
            return classifier;
        };
        for entry in spec.split(',') {
            let entry = entry.trim();
            if entry.is_empty() {
                continue;
            }
            let Some((address, role)) = entry.split_once('=') else {
                continue;
            };
            let Ok(pc) = parse_u64(address) else {
                continue;
            };
            if let Some(role) = parse_role(role.trim()) {
                classifier.insert(pc, role);
            }
        }
        classifier
    }

    pub fn insert(&mut self, pc: u64, role: Role) {
        self.by_pc.insert(pc, role);
    }

    pub fn classify(&self, pc: u64) -> Option<Role> {
        self.by_pc.get(&pc).copied()
    }
}

/// Share of a stream's bytes a constant must be reported from before it counts as
/// a delimiter class rather than as a positional field.
const STREAM_LEVEL_COVERAGE: f32 = 0.8;

/// ... and the least number of positions that can support it.
const MIN_STREAM_LEVEL_POSITIONS: usize = 3;

/// Confidence below which a discriminant is evidence only.
///
/// The engine demotes a comparison it knows is not a value the input should take
/// (an address, or a boundary of a value derived from the input).  Such an event
/// stays in the raw table, but listing it among the values a byte is *expected* to
/// hold would hand the mutator the boundary itself.
const ASSERTED_DISCRIMINANT_CONFIDENCE: f32 = 0.5;

fn parse_role(text: &str) -> Option<Role> {
    Some(match text.to_ascii_lowercase().as_str() {
        "magic" => Role::Magic,
        "length" => Role::Length,
        "payload" | "memcpy" | "strcpy" | "ring_buffer" => Role::Payload,
        "checksum" => Role::Checksum,
        "config" => Role::Config,
        "propagated" => Role::Propagated,
        _ => return None,
    })
}

fn parse_u64(value: &str) -> Result<u64, std::num::ParseIntError> {
    let value = value.trim();
    if let Some(hex) = value.strip_prefix("0x").or_else(|| value.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16)
    }
    else {
        value.parse()
    }
}

/// Summary of one analysis pass, for logs and reports.
#[derive(Debug, Clone, Default, Serialize)]
pub struct AnalysisReport {
    pub mmio_reads: usize,
    pub read_sites: usize,
    pub source_fragments: usize,
    pub role_entries: usize,
    pub data_blocks: usize,
    pub tainted_stores: usize,
    pub unbound_loads: usize,
    pub magic_compares: usize,
    pub loop_bounds: usize,
    pub table_loads: usize,
    pub checksum_events: usize,
    pub bounded_reads_marked: usize,
    /// Reads that belong to read sites the analysis deliberately dropped (a PC
    /// that served more than one stream cannot be represented at all).  Expected
    /// and harmless, but it must not be confused with the counter below.
    pub dropped_site_reads: usize,
    /// Reads the interpreter never claimed that do *not* belong to a dropped site.
    /// This is the actual misalignment signal for the pass.
    pub early_stop_leftovers: usize,
    /// Streams whose bytes were consumed as data (some read on them reached a sink).
    pub data_streams: usize,
    /// Streams the firmware only read in order to see its own device state.
    pub device_state_streams: usize,
    /// Input bytes that went to data streams, ...
    pub protocol_bytes: u64,
    /// ... and input bytes that went to device-state streams.  The ratio is the
    /// budget argument: the analysis is what makes it measurable per run.
    pub device_state_bytes: u64,
    pub passes: usize,
}

/// One loop bound observed by the dynamic pass: `source`'s value gated a loop
/// that read `target` `count` times.
///
/// `source` and `target` are read *sites* (`occ == 0`): the relation is between
/// sites, not between individual reads.  `source_occs` names the occurrences that
/// actually opened a gate, which is what the length role resolves to input bytes
/// with.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LoopBound {
    pub source: AccessContext,
    pub source_occs: Vec<u32>,
    pub target: AccessContext,
    /// Reads of the target over the whole pass.  This is the quantity that can be
    /// compared with the length field's own value.
    pub count: u64,
    /// Reads that happened while the gate was in effect (used to mark payload).
    /// Never greater than `count`: a bound cannot cover more reads than the site
    /// ever had, and `emit_observations` asserts it.
    pub gated_count: u64,
}

/// One equality-gated constant, with the read site it is expected from and where
/// the comparison happened.
///
/// Kept as raw evidence rather than collapsed into the roles: a suspicious
/// discriminant has to be traceable back to the instruction that stated it, which
/// is impossible without the comparison PC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct MagicSite {
    pub source: AccessContext,
    pub value: u64,
    /// How many read sites this single comparison drew on.  More than one is the
    /// fingerprint of an unmodelled checksum, so it is kept rather than collapsed.
    pub width: u32,
    pub compare_pc: u64,
    /// Whether the compared value's taint came from the register file's shadow
    /// rather than from a definition in the interpreted group.
    pub from_fallback: bool,
}

/// A constant the firmware tests against *every* byte of a stream.
///
/// This is a terminator or escape class (the target's `\r`, `\n` and `0x7f`), not
/// a positional field: the positions are free and the value is what matters when
/// it appears.  Reporting it as a magic *field* would have the mutator write the
/// delimiter into every byte of the stream, so the two are separated.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct StreamConstraint {
    pub stream: StreamKey,
    pub value: u64,
    /// Fraction of the stream's bytes this constant was reported from.
    pub coverage: f32,
    /// How many input positions reported it.
    pub positions: u32,
}

/// The full analysis artefact.
#[derive(Debug, Clone, Serialize)]
pub struct RoleMapOutput {
    pub role_map: RoleMap,
    pub blocks: Vec<DataBlock>,
    /// Raw loop-bound evidence, kept alongside the roles it produced so the
    /// inference can be audited without re-running the analysis.
    pub loop_bounds: Vec<LoopBound>,
    /// Raw magic evidence: every equality-gated constant, with the read site it
    /// was expected from and the comparison that stated it.
    pub magic_sites: Vec<MagicSite>,
    /// The constants that are stream-level constraints rather than fields.
    pub stream_constraints: Vec<StreamConstraint>,
    /// Streams whose bytes carry no protocol role: the firmware read them to see
    /// its own device state (a polled status register, a clock or GPIO
    /// configuration value).  Supplying them costs input budget and buys nothing.
    pub device_state_streams: Vec<StreamKey>,
    /// Loads whose *address* derived from input, with the instruction that loaded
    /// through it.  This is the evidence an `index`/`offset` role derives from.
    pub table_load_sites: Vec<(u64, Vec<AccessContext>)>,
    pub report: AnalysisReport,
}

/// Turns dynamic-pass events into role-map entries.
///
/// `on_source_load` binds a read site to the exact input bytes it consumed;
/// `on_tainted_store` receives the *precise* provenance of a store, which is the
/// point where the role classifier can be applied without any value guessing.
#[derive(Clone)]
pub struct RoleCollector {
    ledger: ReadLedger,
    classifier: SinkClassifier,
    state: Rc<RefCell<RoleCollectorState>>,
}

#[derive(Default)]
struct RoleCollectorState {
    /// Keyed by `(pc, stream, occ)`: one entry per read *occurrence*, so a packet
    /// collected byte-by-byte keeps every byte's identity instead of only the
    /// most recent one.
    fragments: HashMap<(u64, StreamKey, u32), SourceFragment>,
    /// Next expected input offset per read site, so positional matching cannot
    /// silently re-use bytes that were already attributed.
    expected_offset: HashMap<(u64, StreamKey), u32>,
    role_map: RoleMap,
    tainted_stores: usize,
    unbound_loads: usize,
    magic_compares: usize,
    loop_bounds: Vec<LoopBound>,
    /// Every magic value seen, with the read site it was expected from and the
    /// comparison that stated it.
    magic_sites: Vec<MagicSite>,
    /// Constants tested against a whole stream rather than against a position.
    stream_constraints: Vec<StreamConstraint>,
    /// Tainted-address loads, with the loading instruction.
    table_load_sites: Vec<(u64, Vec<AccessContext>)>,
    checksum_events: usize,
    table_loads: usize,
    bounded_reads_marked: usize,
    /// Number of source-load notifications received, i.e. loads the engine
    /// interpreted.  Compared against the engine's own total this exposes any
    /// drift between the two counters, which is the signature of interpretation
    /// and execution having taken different paths.
    loads_observed: usize,
    /// Loads whose interpreted width disagreed with the width the interpreter
    /// mode actually recorded.  Purely diagnostic: the recorded read is still
    /// authoritative, so the binding is kept.
    size_mismatches: usize,
    /// Read sites the discovery pass found, for the report's audit split.
    read_sites: HashMap<u64, Vec<StreamKey>>,
    /// `(pc, stream)` pairs whose reads were consumed as *data*: a magic value the
    /// firmware asserted, a length bound, a checksum chain, a tainted address, or
    /// a copy into guest memory.  A stream with none of these is device state.
    sink_sites: BTreeSet<(u64, StreamKey)>,
    /// `(pc, stream)` pairs whose value the firmware wrote back into a peripheral
    /// register -- the read-modify-write of a status register, which is device
    /// state and never protocol data.
    device_write_sites: BTreeSet<(u64, StreamKey)>,
}

impl RoleCollector {
    pub fn new(ledger: ReadLedger, classifier: SinkClassifier) -> Self {
        Self { ledger, classifier, state: Rc::new(RefCell::new(RoleCollectorState::default())) }
    }

    #[cfg(test)]
    pub fn role_map(&self) -> RoleMap {
        self.state.borrow().role_map.clone()
    }

    #[cfg(test)]
    pub fn tainted_stores(&self) -> usize {
        self.state.borrow().tainted_stores
    }

    #[cfg(test)]
    pub fn unbound_loads(&self) -> usize {
        self.state.borrow().unbound_loads
    }

    pub fn size_mismatches(&self) -> usize {
        self.state.borrow().size_mismatches
    }

    pub fn loads_observed(&self) -> usize {
        self.state.borrow().loads_observed
    }

    /// Hand the pass the read sites the discovery pass found.
    ///
    /// The report splits the unclaimed reads by whether the pass knew about the
    /// site at all, so it needs the same map the engine resolves reads with.
    pub fn set_read_sites(&mut self, read_sites: HashMap<u64, Vec<StreamKey>>) {
        self.state.borrow_mut().read_sites = read_sites;
    }

    /// Drop everything the previous pass produced.
    ///
    /// Only the ledger survives (it is the record of what the emulator did, and
    /// the driver clears it itself).  Everything else -- the role map, the loop
    /// bounds, the magic evidence and every counter -- belongs to one pass.  A
    /// later stage runs several passes in one process (a parent input and its
    /// mutations), and leaving them to accumulate would hand the comparison a
    /// mixture instead of two documents.
    pub fn reset_pass(&mut self) {
        let mut state = self.state.borrow_mut();
        state.fragments.clear();
        state.expected_offset.clear();
        state.role_map = RoleMap::default();
        state.loop_bounds.clear();
        state.magic_sites.clear();
        state.stream_constraints.clear();
        state.table_load_sites.clear();
        state.tainted_stores = 0;
        state.unbound_loads = 0;
        state.magic_compares = 0;
        state.checksum_events = 0;
        state.table_loads = 0;
        state.bounded_reads_marked = 0;
        state.loads_observed = 0;
        state.size_mismatches = 0;
    }

    pub fn output(&self, passes: usize) -> RoleMapOutput {
        let state = self.state.borrow();
        let mut role_map = state.role_map.clone();
        // Which streams carry protocol data and which are device state.  A stream
        // counts as data if some read on it reached a consumption sink; everything
        // else is the firmware reading its own device state, which the fuzzer must
        // still supply but which carries no protocol role.
        let data_streams: BTreeSet<StreamKey> =
            state.sink_sites.iter().map(|(_, stream)| *stream).collect();
        let bytes_by_stream = self.ledger.bytes_by_stream();
        let mut device_state_streams: Vec<StreamKey> = Vec::new();
        let (mut protocol_bytes, mut device_state_bytes) = (0_u64, 0_u64);
        for (stream, bytes) in bytes_by_stream.iter() {
            if data_streams.contains(stream) {
                protocol_bytes += *bytes;
            }
            else {
                device_state_bytes += *bytes;
                device_state_streams.push(*stream);
            }
        }
        device_state_streams.sort_unstable();
        // Reads that belong to a dropped site are not a misalignment signal: the
        // site could not be represented at all, so nothing was ever going to claim
        // them.  Only what is left over is.
        let (dropped_site_reads, early_stop_leftovers) =
            self.ledger.pending_split(&state.read_sites);
        let counters = AnalysisReport {
            mmio_reads: self.ledger.len(),
            read_sites: self.ledger.site_count(),
            source_fragments: state.fragments.len(),
            tainted_stores: state.tainted_stores,
            unbound_loads: state.unbound_loads,
            magic_compares: state.magic_compares,
            loop_bounds: state.loop_bounds.len(),
            table_loads: state.table_loads,
            checksum_events: state.checksum_events,
            bounded_reads_marked: state.bounded_reads_marked,
            dropped_site_reads,
            early_stop_leftovers,
            data_streams: data_streams.len(),
            device_state_streams: device_state_streams.len(),
            protocol_bytes,
            device_state_bytes,
            role_entries: 0,
            data_blocks: 0,
            passes,
        };
        let mut loop_bounds = state.loop_bounds.clone();
        let mut magic_sites = state.magic_sites.clone();
        let mut table_load_sites = state.table_load_sites.clone();
        // Canonical order.  This document is compared byte-for-byte by the aligned
        // parent/mutant harness, so two identical analyses must produce identical
        // bytes; a per-process hash seed must not leak into the file.  The merge
        // rules themselves are already commutative, so this is about ordering only.
        loop_bounds.sort_by_key(|bound| (bound.source, bound.target));
        // Keys must be total (no ties): a stable sort would keep the original hash
        // order for equal keys, and those bytes are compared across runs too.
        magic_sites.sort_unstable();
        // The same comparison can be reached twice within one group (a loop inside
        // the group, or an intra-group re-entry).  The table is a set of
        // (site, constant, comparison), not a count, so identical rows collapse.
        magic_sites.dedup();
        table_load_sites.sort_unstable();
        table_load_sites.dedup();
        // Cross-check each length role against the field's own value: a field whose
        // value equals the number of reads it gated is confirmed, one that does not
        // is probably a counter or an interrupt artefact.
        let mut length_confidence: BTreeMap<(StreamKey, (u32, u32)), f32> = BTreeMap::new();
        for bound in &state.loop_bounds {
            // Every occurrence that opened a gate is checked: the bound is a
            // relation between two sites, and any of the occurrences may carry it.
            for occ in &bound.source_occs {
                let key = (bound.source.pc, bound.source.addr, *occ);
                let Some(fragment) = state.fragments.get(&key) else { continue };
                // A demoted entry is expected to be a terminator loop that survived
                // the admission rule: its "length" is really the data byte that ends
                // the string, so its value does not equal the read count.  Such
                // entries are the negative samples of a confirmation-rate statistic
                // rather than a bug, and no payload span is derived from them.
                let confidence = if fragment.value == bound.count { 0.85 } else { 0.5 };
                // One field can legitimately gate two loops, and then only one of the
                // two may be confirmed.  Take the pessimistic side and take it
                // commutatively: assigning per bound would make the final
                // confidence depend on hash iteration order.
                length_confidence
                    .entry((bound.source.addr, fragment.offset_range))
                    .and_modify(|current| *current = current.min(confidence))
                    .or_insert(confidence);
            }
        }
        // Relation-driven payload: a length field bounds the bytes that follow it
        // in the same stream.  The per-read marking cannot cover this by itself --
        // in a bottom-tested loop the first iteration's read happens before the
        // gate exists -- and deriving the span is also the first time a *relation*
        // (not just a role) produces marks.
        let mut derived_payload = Vec::new();
        for bound in &state.loop_bounds {
            // Only within one stream: the formula assumes the payload follows the
            // length field in the input, which is not true across streams.
            if bound.target.addr != bound.source.addr {
                continue;
            }
            let width = target_width(&state, bound.target);
            if width == 0 {
                continue;
            }
            for occ in &bound.source_occs {
                let key = (bound.source.pc, bound.source.addr, *occ);
                let Some(fragment) = state.fragments.get(&key) else { continue };
                // Only a *confirmed* length derives a span.  `count` is the number
                // of reads of the target site, so if other paths share that site the
                // count is inflated and the derived interval would paint unrelated
                // bytes; value == count is exactly the confirmation signal.
                if fragment.value != bound.count {
                    continue;
                }
                let start = fragment.offset_range.1;
                let end = start.saturating_add((bound.count * width as u64) as u32);
                if end > start {
                    derived_payload.push((bound.source.addr, (start, end), 0.85));
                }
            }
        }
        // How far each stream reaches, for the stream-level classification below: a
        // delimiter is tested against (nearly) all of a stream's bytes, while a
        // positional field is tested against the one position that carries it.
        let mut stream_bytes: BTreeMap<StreamKey, u32> = BTreeMap::new();
        for ((_, stream, _), fragment) in &state.fragments {
            let entry = stream_bytes.entry(*stream).or_insert(0);
            *entry = (*entry).max(fragment.offset_range.1);
        }
        drop(state);

        // A dispatching comparison chain (`cmp #0xe100` ... `cmp #0x1c200`) shows
        // up as several discriminants on the same bytes, which is much stronger
        // evidence of a protocol constant than a single comparison.
        for entry in &mut role_map.entries {
            if entry.role == Role::Magic && entry.discriminants.len() >= 2 {
                entry.confidence = entry.confidence.max(0.85);
            }
        }
        for ((stream, offset_range), confidence) in length_confidence {
            for entry in &mut role_map.entries {
                if entry.role == Role::Length
                    && entry.stream == stream
                    && entry.offset_range == offset_range
                {
                    entry.confidence = confidence;
                }
            }
        }
        for (stream, offset_range, confidence) in derived_payload {
            role_map.insert(RoleEntry::new(stream, offset_range, Role::Payload, confidence));
        }
        upgrade_consumed_payload(&mut role_map);

        // A constant the firmware tests against *every* byte of a stream is a
        // delimiter, not a positional field.  Coverage is measured over the bytes of
        // the stream the analysis actually saw: on the target the console
        // terminator fires on every byte, while a command-name letter fires on the
        // one position that carries it.
        let mut per_value: BTreeMap<(StreamKey, u64), BTreeSet<u32>> = BTreeMap::new();
        for entry in &role_map.entries {
            if entry.role != Role::Magic {
                continue;
            }
            for value in &entry.discriminants {
                per_value
                    .entry((entry.stream, *value))
                    .or_default()
                    .insert(entry.offset_range.0);
            }
        }
        let mut stream_constraints = Vec::new();
        let mut stream_level: BTreeSet<(StreamKey, u64)> = BTreeSet::new();
        for ((stream, value), positions) in per_value {
            let Some(&bytes) = stream_bytes.get(&stream) else { continue };
            if bytes == 0 || positions.len() < MIN_STREAM_LEVEL_POSITIONS {
                continue;
            }
            let coverage = positions.len() as f32 / bytes as f32;
            if coverage < STREAM_LEVEL_COVERAGE {
                continue;
            }
            stream_level.insert((stream, value));
            stream_constraints.push(StreamConstraint {
                stream,
                value,
                coverage,
                positions: positions.len() as u32,
            });
        }
        for entry in &mut role_map.entries {
            if entry.role == Role::Magic
                && entry
                    .discriminants
                    .iter()
                    .any(|value| stream_level.contains(&(entry.stream, *value)))
            {
                entry.stream_level = true;
            }
        }

        // A magic role with nothing asserted about it is not a role: every
        // comparison that produced it was demoted (a pointer, a boundary of a
        // derived value, or a flag), so there is no claim left to make.  The raw
        // events stay in `magic_sites` for the audit trail.
        role_map.entries.retain(|entry| {
            entry.role != Role::Magic
                || entry.confidence >= ASSERTED_DISCRIMINANT_CONFIDENCE
                || !entry.discriminants.is_empty()
        });

        role_map
            .entries
            .sort_by_key(|entry| {
                (entry.stream, entry.offset_range.0, entry.offset_range.1, entry.role)
            });
        let blocks = role_map.blocks();
        let report = AnalysisReport {
            role_entries: role_map.len(),
            data_blocks: blocks.len(),
            ..counters
        };
        RoleMapOutput {
            role_map,
            blocks,
            loop_bounds,
            magic_sites,
            stream_constraints,
            device_state_streams,
            table_load_sites,
            report,
        }
    }

    fn bind_fragment(&self, context: AccessContext, size: u8) {
        let key = (context.pc, context.addr, context.occ);
        let mut state = self.state.borrow_mut();
        let site = (context.pc, context.addr);
        let expected = state.expected_offset.get(&site).copied().unwrap_or(0);
        let Some(record) = self.ledger.take_next(context.pc, context.addr, expected) else {
            state.unbound_loads += 1;
            return;
        };
        if record.size != size {
            state.size_mismatches += 1;
        }
        state.expected_offset.insert(site, record.end());
        state.fragments.insert(
            key,
            SourceFragment {
                site_pc: context.pc,
                stream: context.addr,
                offset_range: (record.input_offset, record.end()),
                value: record.value,
                icount: record.icount,
            },
        );
    }

    /// Turn provenance into roles: resolve each source read site to the input
    /// bytes it supplied, and record `role` for those bytes.
    ///
    /// Resolution is positional (via the ledger), so two sources that happen to
    /// hold equal byte values can never be conflated, and a source with no bound
    /// fragment simply contributes nothing rather than a guess.
    fn emit_role(
        &self,
        sources: &[AccessContext],
        role: Role,
        confidence: f32,
        discriminants: &[u64],
    ) {
        let mut state = self.state.borrow_mut();
        let mut bound: Vec<(StreamKey, (u32, u32))> = Vec::new();
        for source in sources {
            if let Some(fragment) =
                state.fragments.get(&(source.pc, source.addr, source.occ))
            {
                bound.push((source.addr, fragment.offset_range));
            }
        }
        for (stream, offset_range) in bound {
            let mut entry = RoleEntry::new(stream, offset_range, role, confidence);
            entry.discriminants = discriminants.to_vec();
            state.role_map.insert(entry);
        }
    }

    /// Note that these reads were consumed as data rather than as device state.
    fn note_sinks(&self, sources: &[AccessContext]) {
        let mut state = self.state.borrow_mut();
        for source in sources {
            state.sink_sites.insert((source.pc, source.addr));
        }
    }
}

impl PhaseBObserver for RoleCollector {
    fn on_source_load(&mut self, context: AccessContext, size: u8) {
        self.state.borrow_mut().loads_observed += 1;
        self.bind_fragment(context, size);
    }

    fn on_tainted_store(
        &mut self,
        sources: &[AccessContext],
        pc: u64,
        _addr: u64,
        _size: u8,
        _value: u64,
    ) {
        self.state.borrow_mut().tainted_stores += 1;
        // A copy into guest memory is the clearest use of the bytes as data.
        self.note_sinks(sources);
        let role = self.classifier.classify(pc).unwrap_or(Role::Propagated);
        let confidence = if role == Role::Propagated { 0.4 } else { 0.9 };
        self.emit_role(sources, role, confidence, &[]);
    }

    fn on_device_write(&mut self, sources: &[AccessContext], _pc: u64, _addr: u64) {
        // A write back into a peripheral register is the firmware driving its own
        // device.  It gives these reads no role, and it is recorded as evidence
        // that the stream is device state rather than protocol data.
        let mut state = self.state.borrow_mut();
        for source in sources {
            state.device_write_sites.insert((source.pc, source.addr));
        }
    }

    fn on_magic_compare(
        &mut self,
        sources: &[AccessContext],
        value: u64,
        confidence: f32,
        evidence: MagicEvidence,
    ) {
        // Only an asserted comparison says the firmware consumed these bytes as
        // data; a demoted one is evidence, not consumption.
        if confidence >= ASSERTED_DISCRIMINANT_CONFIDENCE {
            self.note_sinks(sources);
        }
        {
            let mut state = self.state.borrow_mut();
            state.magic_compares += 1;
            let width = sources.len() as u32;
            for source in sources {
                state.magic_sites.push(MagicSite {
                    source: *source,
                    value,
                    width,
                    compare_pc: evidence.compare_pc,
                    from_fallback: evidence.from_fallback,
                });
            }
        }
        // The compared constant is what the firmware expects to read, so it is a
        // discriminant of these bytes -- captured at the comparison, with no
        // branch or gating bookkeeping in between.
        let discriminants: &[u64] = if confidence < ASSERTED_DISCRIMINANT_CONFIDENCE {
            &[]
        }
        else {
            &[value]
        };
        self.emit_role(sources, Role::Magic, confidence, discriminants);
    }

    fn on_loop_bound(
        &mut self,
        source: AccessContext,
        source_occs: &[u32],
        target: AccessContext,
        count: u64,
        gated_count: u64,
    ) {
        self.state.borrow_mut().loop_bounds.push(LoopBound {
            source,
            source_occs: source_occs.to_vec(),
            target,
            count,
            gated_count,
        });
        // The source's value decided how many times the target was consumed, so
        // it is a length field.  The role attaches to every occurrence that opened
        // a gate, which is what resolves the relation to concrete input bytes.
        //
        // Only the source is consumption evidence: the target is *bounded* by the
        // length, and on this target that included the status register the gate
        // was polled through, which is not data.
        self.note_sinks(&[source]);
        for occ in source_occs {
            let read = AccessContext::at(source.pc, source.addr, *occ);
            self.emit_role(&[read], Role::Length, 0.7, &[]);
        }
    }

    fn on_bounded_read(&mut self, read: AccessContext) {
        // A byte read while a length field was in effect is part of the payload
        // that length paid for.
        self.state.borrow_mut().bounded_reads_marked += 1;
        self.emit_role(&[read], Role::Payload, 0.8, &[]);
    }

    fn on_table_load(&mut self, addr_sources: &[AccessContext], pc: u64) {
        // The value read out of the table is clean, so there is nothing to label on
        // the value; the event is checksum evidence (see the store path in the
        // engine), and its provenance is what an index/offset role needs.  Using a
        // byte as an address is also the clearest sign it is data.
        self.note_sinks(addr_sources);
        let mut state = self.state.borrow_mut();
        state.table_loads += 1;
        state.table_load_sites.push((pc, addr_sources.to_vec()));
    }

    fn on_checksum(&mut self, sources: &[AccessContext], confidence: f32) {
        self.state.borrow_mut().checksum_events += 1;
        self.note_sinks(sources);
        self.emit_role(sources, Role::Checksum, confidence, &[]);
    }
}

pub fn save_role_map(path: &std::path::Path, output: &RoleMapOutput) -> anyhow::Result<()> {
    std::fs::write(path, serde_json::to_vec_pretty(output)?)?;
    Ok(())
}

/// Width, in bytes, of one read of `target`.
///
/// Prefers the first occurrence, falling back to the lowest occurrence that was
/// bound: the width of a given load instruction is constant, and `size_mismatches`
/// reports it if that ever stops being true.
fn target_width(state: &RoleCollectorState, target: AccessContext) -> u32 {
    let mut best: Option<(u32, u32)> = None;
    for ((pc, addr, occ), fragment) in &state.fragments {
        if *pc != target.pc || *addr != target.addr {
            continue;
        }
        let width = fragment.offset_range.1.saturating_sub(fragment.offset_range.0);
        match best {
            Some((best_occ, _)) if best_occ <= *occ => {}
            _ => best = Some((*occ, width)),
        }
    }
    best.map(|(_, width)| width).unwrap_or(0)
}

/// Promote a long run of unclassified-but-consumed bytes to payload.
///
/// A byte copied out of a buffer without ever being compared or used as a bound
/// still only ever ends up labelled `Propagated`; when several such bytes sit
/// next to each other they are, in aggregate, the payload.
fn upgrade_consumed_payload(map: &mut RoleMap) {
    let mut additions = Vec::new();
    let mut covered = Vec::new();
    for stream in map.streams() {
        let mut runs: Vec<(u32, u32)> = Vec::new();
        for entry in map
            .entries
            .iter()
            .filter(|entry| entry.stream == stream && entry.role == Role::Propagated)
        {
            // Small gaps are still one run: a loop reads in strides.
            let extend = match runs.last() {
                Some(last) => entry.offset_range.0 <= last.1.saturating_add(8),
                None => false,
            };
            if extend {
                let last = runs.last_mut().expect("checked above");
                last.1 = last.1.max(entry.offset_range.1);
            }
            else {
                runs.push(entry.offset_range);
            }
        }
        for run in runs {
            if run.1.saturating_sub(run.0) > 4 {
                covered.push((stream, run));
                additions.push(RoleEntry::new(stream, run, Role::Payload, 0.8));
            }
        }
    }
    if additions.is_empty() {
        return;
    }
    map.entries.retain(|entry| {
        !(entry.role == Role::Propagated
            && covered.iter().any(|(stream, run)| {
                *stream == entry.stream
                    && entry.offset_range.0 >= run.0
                    && entry.offset_range.1 <= run.1
            }))
    });
    for entry in additions {
        map.insert(entry);
    }
}

pub fn save_reads(path: &std::path::Path, reads: &[ReadRecord]) -> anyhow::Result<()> {
    use std::io::Write;

    let mut writer = std::io::BufWriter::new(std::fs::File::create(path)?);
    for record in reads {
        writeln!(writer, "{}", serde_json::to_string(record)?)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_entries_merge_overlapping_same_role_ranges() {
        let mut map = RoleMap::default();
        map.insert(RoleEntry::new(0x4000, (0, 4), Role::Payload, 0.5));
        map.insert(RoleEntry::new(0x4000, (2, 8), Role::Payload, 0.9));
        assert_eq!(map.len(), 1);
        assert_eq!(map.entries[0].offset_range, (0, 8));
        assert_eq!(map.entries[0].confidence, 0.9);
    }

    #[test]
    fn different_roles_on_one_stream_stay_separate() {
        let mut map = RoleMap::default();
        map.insert(RoleEntry::new(0x4000, (0, 1), Role::Magic, 0.9));
        map.insert(RoleEntry::new(0x4000, (1, 4), Role::Length, 0.9));
        map.insert(RoleEntry::new(0x4000, (4, 20), Role::Payload, 0.9));
        assert_eq!(map.len(), 3);
        assert_eq!(map.role_at(0x4000, 0).unwrap().role, Role::Magic);
        assert_eq!(map.role_at(0x4000, 2).unwrap().role, Role::Length);
        assert_eq!(map.role_at(0x4000, 10).unwrap().role, Role::Payload);
    }

    #[test]
    fn blocks_are_derived_from_contiguous_role_ranges() {
        let mut map = RoleMap::default();
        map.insert(RoleEntry::new(0x4000, (0, 1), Role::Magic, 0.9));
        map.insert(RoleEntry::new(0x4000, (1, 4), Role::Length, 0.9));
        map.insert(RoleEntry::new(0x4000, (40, 44), Role::Payload, 0.9));

        let blocks = map.blocks();
        assert_eq!(blocks.len(), 2, "a wide gap starts a new block: {blocks:?}");
        assert_eq!(blocks[0].offset_range, (0, 4));
        assert_eq!(blocks[1].offset_range, (40, 44));
    }

    #[test]
    fn role_map_round_trips_through_json() {
        let mut map = RoleMap::default();
        map.insert(RoleEntry {
            stream: 0x4000,
            offset_range: (0, 4),
            role: Role::Magic,
            confidence: 0.9,
            discriminants: vec![3, 7],
            stream_level: false,
        });
        // Exercise the mechanism the mutation side actually reads: the encoding
        // that `save_role_map` writes.
        let json = serde_json::to_string(&map).unwrap();
        assert_eq!(serde_json::from_str::<RoleMap>(&json).unwrap(), map);
    }

    #[test]
    fn ledger_matches_reads_positionally_not_by_value() {
        let ledger = ReadLedger::new();
        // Two identical values at the same site: they must not be confused.
        ledger.record(0x100, 0x4000, 0, &[0x41], Default::default());
        ledger.record(0x100, 0x4000, 1, &[0x41], Default::default());

        let first = ledger.take_next(0x100, 0x4000, 0).expect("first read");
        assert_eq!(first.input_offset, 0);
        let second = ledger.take_next(0x100, 0x4000, 1).expect("second read");
        assert_eq!(second.input_offset, 1);
        assert!(ledger.take_next(0x100, 0x4000, 2).is_none());
    }

    #[test]
    fn ledger_drops_stale_records_before_the_expected_offset() {
        let ledger = ReadLedger::new();
        ledger.record(0x100, 0x4000, 0, &[1], Default::default());
        ledger.record(0x100, 0x4000, 1, &[2], Default::default());
        ledger.record(0x100, 0x4000, 7, &[3], Default::default());

        // A pass that starts at offset 7 must skip the two stale records.
        let record = ledger.take_next(0x100, 0x4000, 7).expect("record at 7");
        assert_eq!(record.input_offset, 7);
    }

    #[test]
    fn classifier_maps_configured_sinks_to_roles() {
        let mut classifier = SinkClassifier::new();
        classifier.insert(0x81d24, Role::Payload);
        assert_eq!(classifier.classify(0x81d24), Some(Role::Payload));
        assert_eq!(classifier.classify(0xdead), None);
        assert_eq!(parse_role("length"), Some(Role::Length));
        assert_eq!(parse_role("bogus"), None);
    }

    #[test]
    fn collector_binds_fragment_and_classifies_tainted_store() {
        let ledger = ReadLedger::new();
        ledger.record(0x100, 0x4000, 0, &[0x41, 0x42], Default::default());

        let mut classifier = SinkClassifier::new();
        classifier.insert(0x200, Role::Payload);

        let mut collector = RoleCollector::new(ledger, classifier);
        let source = AccessContext::new(0x100, 0x4000);
        collector.on_source_load(source, 2);
        collector.on_tainted_store(&[source], 0x200, 0x2000_0000, 2, 0x4241);

        let map = collector.role_map();
        assert_eq!(map.len(), 1);
        let entry = &map.entries[0];
        assert_eq!(entry.stream, 0x4000);
        assert_eq!(entry.offset_range, (0, 2));
        assert_eq!(entry.role, Role::Payload);
        assert_eq!(collector.tainted_stores(), 1);
        assert_eq!(collector.unbound_loads(), 0);
    }

    #[test]
    fn collector_reports_unbound_loads_instead_of_guessing() {
        let ledger = ReadLedger::new();
        let mut collector = RoleCollector::new(ledger, SinkClassifier::new());
        let source = AccessContext::new(0x100, 0x4000);
        collector.on_source_load(source, 2);
        collector.on_tainted_store(&[source], 0x200, 0x2000_0000, 2, 0);

        assert!(
            collector.role_map().entries.is_empty(),
            "no fragment means no role entry"
        );
        assert_eq!(collector.unbound_loads(), 1);
    }

    #[test]
    fn magic_compare_becomes_a_magic_role_with_its_discriminant() {
        let ledger = ReadLedger::new();
        ledger.record(0x100, 0x4000, 0, &[0x41], Default::default());

        let mut collector = RoleCollector::new(ledger, SinkClassifier::new());
        let source = AccessContext::new(0x100, 0x4000);
        collector.on_source_load(source, 1);
        collector.on_magic_compare(&[source], 0x41, 0.85, MagicEvidence::at(0x200));

        let map = collector.role_map();
        assert_eq!(map.len(), 1);
        assert_eq!(map.entries[0].role, Role::Magic);
        assert_eq!(map.entries[0].discriminants, vec![0x41]);
        assert_eq!(map.entries[0].offset_range, (0, 1));
    }

    #[test]
    fn repeated_magic_compares_accumulate_discriminants() {
        let ledger = ReadLedger::new();
        ledger.record(0x100, 0x4000, 0, &[0x41], Default::default());

        let mut collector = RoleCollector::new(ledger, SinkClassifier::new());
        let source = AccessContext::new(0x100, 0x4000);
        collector.on_source_load(source, 1);
        collector.on_magic_compare(&[source], 0x41, 0.85, MagicEvidence::at(0x200));
        collector.on_magic_compare(&[source], 0x42, 0.85, MagicEvidence::at(0x204));

        let map = collector.role_map();
        assert_eq!(map.len(), 1, "one range, one role, two discriminants");
        assert_eq!(map.entries[0].discriminants, vec![0x41, 0x42]);
    }

    #[test]
    fn loop_bound_becomes_a_length_role_and_keeps_its_evidence() {
        let ledger = ReadLedger::new();
        ledger.record(0x200, 0x5800_0008, 0, &[4], Default::default());

        let mut collector = RoleCollector::new(ledger, SinkClassifier::new());
        let read = AccessContext::new(0x200, 0x5800_0008);
        let source = AccessContext::site(0x200, 0x5800_0008);
        let target = AccessContext::site(0x300, 0x5800_0000);
        collector.on_source_load(read, 1);
        collector.on_loop_bound(source, &[1], target, 4, 4);

        let output = collector.output(1);
        assert_eq!(output.role_map.len(), 1);
        assert_eq!(output.role_map.entries[0].role, Role::Length);
        assert_eq!(output.role_map.entries[0].offset_range, (0, 1));
        assert_eq!(output.report.loop_bounds, 1);
        assert_eq!(
            output.loop_bounds,
            vec![LoopBound {
                source,
                source_occs: vec![1],
                target,
                count: 4,
                gated_count: 4,
            }]
        );
    }

    #[test]
    fn sinked_and_gated_bytes_on_one_stream_keep_separate_roles() {
        let ledger = ReadLedger::new();
        ledger.record(0x100, 0x4000, 0, &[0x41], Default::default());
        ledger.record(0x100, 0x4000, 1, &[9, 0, 0, 0], Default::default());

        let mut classifier = SinkClassifier::new();
        classifier.insert(0x200, Role::Payload);

        let mut collector = RoleCollector::new(ledger, classifier);
        let source = AccessContext::new(0x100, 0x4000);
        collector.on_source_load(source, 1);
        collector.on_magic_compare(&[source], 0x41, 0.85, MagicEvidence::at(0x200));
        collector.on_source_load(source, 4);
        collector.on_tainted_store(&[source], 0x200, 0x2000_0000, 4, 9);

        let map = collector.role_map();
        assert_eq!(map.len(), 2, "magic byte and payload bytes are different ranges");
        assert_eq!(map.role_at(0x4000, 0).unwrap().role, Role::Magic);
        assert_eq!(map.role_at(0x4000, 2).unwrap().role, Role::Payload);
    }

    /// The value-equals-count cross-check, which is what separates a real length
    /// field from a counter or an interrupt artefact.
    #[test]
    fn length_role_is_confirmed_when_the_value_matches_the_read_count() {
        let ledger = ReadLedger::new();
        // The length field reads 3, and the loop it gates reads the target 3 times.
        ledger.record(0x200, 0x5800_0008, 0, &[3], Default::default());

        let mut collector = RoleCollector::new(ledger, SinkClassifier::new());
        let read = AccessContext::new(0x200, 0x5800_0008);
        let source = AccessContext::site(0x200, 0x5800_0008);
        let target = AccessContext::site(0x300, 0x5800_0000);
        collector.on_source_load(read, 1);
        collector.on_loop_bound(source, &[1], target, 3, 2);

        let output = collector.output(1);
        assert_eq!(output.role_map.entries[0].role, Role::Length);
        assert_eq!(output.role_map.entries[0].confidence, 0.85);
        assert_eq!(output.loop_bounds[0].gated_count, 2, "evidence keeps both counts");
    }

    #[test]
    fn length_role_is_demoted_when_the_value_disagrees_with_the_count() {
        let ledger = ReadLedger::new();
        // The field reads 9 but only 3 reads happened: not a length.
        ledger.record(0x200, 0x5800_0008, 0, &[9], Default::default());

        let mut collector = RoleCollector::new(ledger, SinkClassifier::new());
        let read = AccessContext::new(0x200, 0x5800_0008);
        let source = AccessContext::site(0x200, 0x5800_0008);
        let target = AccessContext::site(0x300, 0x5800_0000);
        collector.on_source_load(read, 1);
        collector.on_loop_bound(source, &[1], target, 3, 3);

        let output = collector.output(1);
        assert_eq!(output.role_map.entries[0].role, Role::Length);
        assert_eq!(output.role_map.entries[0].confidence, 0.5, "unconfirmed, not discarded");
    }

    /// Build a single-stream `[LEN][PAYLOAD]` shape: the length field reads 3 and
    /// the payload site is read three times, one byte each.
    fn length_then_payload(value: u8, stream: StreamKey) -> RoleCollector {
        let ledger = ReadLedger::new();
        ledger.record(0x200, stream, 0, &[value], Default::default());
        ledger.record(0x300, stream, 1, &[0x11], Default::default());
        ledger.record(0x300, stream, 2, &[0x22], Default::default());
        ledger.record(0x300, stream, 3, &[0x33], Default::default());

        let mut collector = RoleCollector::new(ledger, SinkClassifier::new());
        let source = AccessContext::site(0x200, stream);
        collector.on_source_load(AccessContext::new(0x200, stream), 1);
        for occ in 1..=3 {
            collector.on_source_load(AccessContext::at(0x300, stream, occ), 1);
        }
        collector.on_loop_bound(
            source,
            &[1],
            AccessContext::site(0x300, stream),
            3,
            2,
        );
        collector
    }

    /// The derived span is what covers the byte the per-read marking cannot: in a
    /// bottom-tested loop the first payload byte is read before the gate exists.
    #[test]
    fn derived_payload_covers_the_ungated_first_byte() {
        let stream = 0x5800_0008;
        let output = length_then_payload(3, stream).output(1);

        // The length field itself.
        assert_eq!(output.role_map.role_at(stream, 0).unwrap().role, Role::Length);
        assert_eq!(output.role_map.role_at(stream, 0).unwrap().confidence, 0.85);
        // The first payload byte, which only the derived span can reach.
        let first = output.role_map.role_at(stream, 1).expect("first payload byte");
        assert_eq!(first.role, Role::Payload);
        assert_eq!(first.confidence, 0.85);
        assert_eq!(
            first.offset_range,
            (1, 4),
            "the span is length_end .. length_end + count * width"
        );
    }

    /// An unconfirmed length must not derive a span: the count may include reads
    /// that belong to another path, and painting bytes from it would be invention.
    #[test]
    fn derived_payload_requires_a_confirmed_length() {
        let stream = 0x5800_0008;
        // The field says 9 but only 3 reads happened, so it is not a length.
        let output = length_then_payload(9, stream).output(1);

        assert_eq!(output.role_map.role_at(stream, 0).unwrap().confidence, 0.5);
        let first = output.role_map.role_at(stream, 1);
        assert!(
            first.map_or(true, |entry| entry.role != Role::Payload),
            "no span may be derived from an unconfirmed length"
        );
    }

    /// The span formula assumes the payload follows the length field *in the same
    /// stream*; across streams it would paint unrelated bytes.
    #[test]
    fn derived_payload_stays_within_one_stream() {
        let length_stream = 0x5800_0008;
        let payload_stream = 0x5800_000c;
        let mut collector = length_then_payload(3, length_stream);
        // Re-point the bound at a target on another stream.
        collector.state.borrow_mut().loop_bounds.clear();
        collector.on_loop_bound(
            AccessContext::site(0x200, length_stream),
            &[1],
            AccessContext::site(0x300, payload_stream),
            3,
            2,
        );
        let output = collector.output(1);

        assert!(
            output.role_map.role_at(length_stream, 1).is_none(),
            "no bytes may be painted on the length field's stream"
        );
    }

    /// A PC that read two streams keeps both: they are separate pseudo-sites, not
    /// a site that has to be dropped.
    ///
    /// This is the shared-`read_byte()` shape.  Dropping the site would cost both
    /// streams every role they have; splitting keeps both, which is why the
    /// analysis no longer drops anything it can key by `(pc, stream)`.
    #[test]
    fn multiplexed_read_sites_are_split_not_dropped() {
        let ledger = ReadLedger::new();
        ledger.record(0x100, 0x4000, 0, &[1], Default::default());
        // Same instruction, different peripheral: an indirect load with a new base.
        ledger.record(0x100, 0x4004, 0, &[2], Default::default());
        ledger.record(0x200, 0x4000, 0, &[3], Default::default());

        assert_eq!(ledger.multiplexed_read_sites(), vec![0x100], "still named for the audit");
        let sites = ledger.read_sites();
        assert_eq!(
            sites.get(&0x100).map(|streams| streams.as_slice()),
            Some(&[0x4000u64, 0x4004u64][..]),
            "both streams must survive"
        );
        assert_eq!(sites.get(&0x200).map(|streams| streams.as_slice()), Some(&[0x4000u64][..]));
    }

    /// A field gating two loops is only as confirmed as its weakest link, and the
    /// verdict must not depend on the order the two bounds were observed in.
    #[test]
    fn shared_length_is_confirmed_only_if_every_bound_is() {
        let stream = 0x5800_0008;

        let build = |confirmed_first: bool| {
            let ledger = ReadLedger::new();
            ledger.record(0x200, stream, 0, &[3], Default::default());
            let mut collector = RoleCollector::new(ledger, SinkClassifier::new());
            let source = AccessContext::site(0x200, stream);
            collector.on_source_load(AccessContext::new(0x200, stream), 1);

            // 3 reads matches the field's value of 3; 5 reads does not.
            let confirmed = (source, AccessContext::site(0x300, stream), 3_u64, 3_u64);
            let unconfirmed = (source, AccessContext::site(0x400, stream), 5_u64, 5_u64);
            let order = if confirmed_first {
                [confirmed, unconfirmed]
            }
            else {
                [unconfirmed, confirmed]
            };
            for (source, target, count, gated) in order {
                collector.on_loop_bound(source, &[1], target, count, gated);
            }
            collector
        };

        let first = build(true).output(1);
        let second = build(false).output(1);

        assert_eq!(
            first.role_map.role_at(stream, 0).unwrap().confidence,
            0.5,
            "one unconfirmed bound is enough to make the whole field unconfirmed"
        );
        assert_eq!(first.role_map, second.role_map, "order must not change the verdict");
        assert_eq!(first.loop_bounds, second.loop_bounds, "evidence is ordered canonically");
    }

    /// A constant tested against *every* byte of a stream is a delimiter, not a
    /// positional field, and the two must not be reported the same way.
    #[test]
    fn stream_level_delimiter_is_separated_from_positional_magic() {
        let delim_stream = 0x5800_0000;
        let field_stream = 0x5800_0004;
        let ledger = ReadLedger::new();
        for offset in 0..4u32 {
            ledger.record(0x200, delim_stream, offset, &[0x0d], Default::default());
        }
        ledger.record(0x300, field_stream, 0, &[0x70], Default::default());

        let mut collector = RoleCollector::new(ledger, SinkClassifier::new());
        for occ in 1..=4u32 {
            let read = AccessContext::at(0x200, delim_stream, occ);
            collector.on_source_load(read, 1);
            // Every byte of the stream is tested against the line terminator.
            collector.on_magic_compare(&[read], 0x0d, 0.85, MagicEvidence::at(0x310));
        }
        let field = AccessContext::new(0x300, field_stream);
        collector.on_source_load(field, 1);
        // One position carries a command-name letter: that one *is* a field.
        collector.on_magic_compare(&[field], 0x70, 0.85, MagicEvidence::at(0x314));

        let output = collector.output(1);
        assert_eq!(output.stream_constraints.len(), 1, "only the delimiter qualifies");
        let constraint = output.stream_constraints[0];
        assert_eq!(constraint.stream, delim_stream);
        assert_eq!(constraint.value, 0x0d);
        assert_eq!(constraint.positions, 4);
        assert!(constraint.coverage > 0.99, "every byte: {constraint:?}");

        assert!(
            output.role_map.role_at(delim_stream, 0).unwrap().stream_level,
            "the terminator is a constraint on the stream"
        );
        assert!(
            !output.role_map.role_at(field_stream, 0).unwrap().stream_level,
            "a single position is a field, not a delimiter"
        );
    }

    /// The magic evidence carries the comparison that stated the constant, and an
    /// event reached twice inside one group is one piece of evidence.
    #[test]
    fn magic_evidence_records_the_comparison_and_drops_duplicates() {
        let ledger = ReadLedger::new();
        ledger.record(0x100, 0x4000, 0, &[0x41], Default::default());

        let mut collector = RoleCollector::new(ledger, SinkClassifier::new());
        let source = AccessContext::new(0x100, 0x4000);
        collector.on_source_load(source, 1);
        collector.on_magic_compare(&[source], 0x41, 0.85, MagicEvidence::at(0x200));
        collector.on_magic_compare(&[source], 0x41, 0.85, MagicEvidence::at(0x200));
        collector.on_magic_compare(
            &[source],
            0x41,
            0.85,
            MagicEvidence { compare_pc: 0x208, from_fallback: true },
        );

        let output = collector.output(1);
        assert_eq!(output.magic_sites.len(), 2, "{:?}", output.magic_sites);
        assert_eq!(output.magic_sites[0].compare_pc, 0x200);
        assert_eq!(output.magic_sites[0].source, source);
        assert!(!output.magic_sites[0].from_fallback);
        assert_eq!(output.magic_sites[1].compare_pc, 0x208);
        assert!(output.magic_sites[1].from_fallback);
    }

    /// Reads at a site the pass did not know about are an audit signal; the ones
    /// left over from an early stop are the misalignment signal.
    #[test]
    fn pending_reads_are_split_by_whether_the_site_was_known() {
        let ledger = ReadLedger::new();
        // A multiplexed site: both pairs are known, so both are representable.
        ledger.record(0x100, 0x4000, 0, &[1, 2], Default::default());
        ledger.record(0x100, 0x4004, 0, &[3, 4], Default::default());
        // A read at a site the discovery pass never reported.
        ledger.record(0x200, 0x4000, 0, &[5, 6], Default::default());

        let mut known = HashMap::new();
        known.insert(0x100u64, vec![0x4000u64, 0x4004u64]);
        assert_eq!(ledger.pending_split(&known), (1, 2));
        assert_eq!(ledger.pending(), 3);
    }

    /// A load whose address is tainted is the evidence an index/offset role needs,
    /// so its provenance is kept alongside the count.
    #[test]
    fn table_load_provenance_is_kept() {
        let ledger = ReadLedger::new();
        let mut collector = RoleCollector::new(ledger, SinkClassifier::new());
        let index = AccessContext::new(0x100, 0x4000);
        collector.on_table_load(&[index], 0x200);
        collector.on_table_load(&[index], 0x200);

        let output = collector.output(1);
        assert_eq!(output.table_load_sites, vec![(0x200, vec![index])], "deduplicated");
        assert_eq!(output.report.table_loads, 2, "but the event count is kept");
    }
}
