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
    collections::VecDeque,
    env,
    rc::Rc,
    sync::{Arc, Mutex},
};

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use crate::{
    debugging::trace::IoTracer,
    input::StreamKey,
    phase_b::PhaseBObserver,
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

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Magic => "magic",
            Self::Length => "length",
            Self::Payload => "payload",
            Self::Checksum => "checksum",
            Self::Config => "config",
            Self::Propagated => "propagated",
        }
    }
}

/// One entry of the role map.
///
/// This is the frozen contract: `stream` and `offset_range` locate the bytes,
/// `role` says how they are consumed, `confidence` says how much the analysis
/// trusts that, and `discriminants` carries the equality-gated values the
/// firmware compared those bytes against (empty when the role is not gated on
/// a specific value).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoleEntry {
    pub stream: StreamKey,
    pub offset_range: (u32, u32),
    pub role: Role,
    pub confidence: f32,
    pub discriminants: Vec<u64>,
}

impl RoleEntry {
    pub fn new(
        stream: StreamKey,
        offset_range: (u32, u32),
        role: Role,
        confidence: f32,
    ) -> Self {
        Self { stream, offset_range, role, confidence, discriminants: Vec::new() }
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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
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

    pub fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn from_json(text: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(text)?)
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

    fn record(&self, site_pc: u64, stream: StreamKey, input_offset: u32, value: &[u8]) {
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
            icount: 0,
            active_irq: 0,
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

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// PC -> stream map for every site seen so far.
    pub fn read_sites(&self) -> HashMap<u64, StreamKey> {
        let mut sites = HashMap::new();
        for (site_pc, stream) in self.per_site.lock().expect("read ledger poisoned").keys() {
            sites.insert(*site_pc, *stream);
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
        self.ledger.record(context.pc, addr, input_offset, value);
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
        let mut classifier = Self::default();
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
                classifier.by_pc.insert(pc, role);
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
    pub passes: usize,
}

/// One loop bound observed by the dynamic pass: `source`'s value gated a loop
/// that read `target` `count` times.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct LoopBound {
    pub source: AccessContext,
    pub target: AccessContext,
    pub count: u64,
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
    /// was expected from.
    pub magic_sites: Vec<(AccessContext, u64)>,
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
    /// Every magic value seen, with the read site it was expected from.
    magic_sites: Vec<(AccessContext, u64)>,
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
}

impl RoleCollector {
    pub fn new(ledger: ReadLedger, classifier: SinkClassifier) -> Self {
        Self { ledger, classifier, state: Rc::new(RefCell::new(RoleCollectorState::default())) }
    }

    pub fn role_map(&self) -> RoleMap {
        self.state.borrow().role_map.clone()
    }

    pub fn tainted_stores(&self) -> usize {
        self.state.borrow().tainted_stores
    }

    pub fn unbound_loads(&self) -> usize {
        self.state.borrow().unbound_loads
    }

    pub fn size_mismatches(&self) -> usize {
        self.state.borrow().size_mismatches
    }

    pub fn loads_observed(&self) -> usize {
        self.state.borrow().loads_observed
    }

    /// Fragment bound to a read site's most recent execution.
    pub fn fragment_for(&self, context: AccessContext) -> Option<SourceFragment> {
        self.state
            .borrow()
            .fragments
            .get(&(context.pc, context.addr, context.occ))
            .cloned()
    }

    /// Reset per-pass observations, keeping the ledger (which is the record of
    /// what the emulator actually did).
    pub fn reset_pass(&mut self) {
        let mut state = self.state.borrow_mut();
        state.fragments.clear();
        state.expected_offset.clear();
    }

    pub fn output(&self, passes: usize) -> RoleMapOutput {
        let state = self.state.borrow();
        let mut role_map = state.role_map.clone();
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
            role_entries: 0,
            data_blocks: 0,
            passes,
        };
        let loop_bounds = state.loop_bounds.clone();
        let magic_sites = state.magic_sites.clone();
        // Cross-check each length role against the field's own value: a field whose
        // value equals the number of reads it gated is confirmed, one that does not
        // is probably a counter or an interrupt artefact.
        let mut length_confidence = Vec::new();
        for bound in &state.loop_bounds {
            let key = (bound.source.pc, bound.source.addr, bound.source.occ);
            let Some(fragment) = state.fragments.get(&key) else { continue };
            let confidence = if fragment.value == bound.count { 0.85 } else { 0.5 };
            length_confidence.push((bound.source.addr, fragment.offset_range, confidence));
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
        for (stream, offset_range, confidence) in length_confidence {
            for entry in &mut role_map.entries {
                if entry.role == Role::Length
                    && entry.stream == stream
                    && entry.offset_range == offset_range
                {
                    entry.confidence = confidence;
                }
            }
        }
        upgrade_consumed_payload(&mut role_map);

        let blocks = role_map.blocks();
        let report = AnalysisReport {
            role_entries: role_map.len(),
            data_blocks: blocks.len(),
            ..counters
        };
        RoleMapOutput { role_map, blocks, loop_bounds, magic_sites, report }
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
        let role = self.classifier.classify(pc).unwrap_or(Role::Propagated);
        let confidence = if role == Role::Propagated { 0.4 } else { 0.9 };
        self.emit_role(sources, role, confidence, &[]);
    }

    fn on_magic_compare(&mut self, sources: &[AccessContext], value: u64, confidence: f32) {
        {
            let mut state = self.state.borrow_mut();
            state.magic_compares += 1;
            for source in sources {
                state.magic_sites.push((*source, value));
            }
        }
        // The compared constant is what the firmware expects to read, so it is a
        // discriminant of these bytes -- captured at the comparison, with no
        // branch or gating bookkeeping in between.
        self.emit_role(sources, Role::Magic, confidence, &[value]);
    }

    fn on_loop_bound(&mut self, source: AccessContext, target: AccessContext, count: u64) {
        self.state.borrow_mut().loop_bounds.push(LoopBound { source, target, count });
        // The source's value decided how many times the target was consumed, so
        // it is a length field.
        self.emit_role(&[source], Role::Length, 0.7, &[]);
    }

    fn on_bounded_read(&mut self, target: AccessContext, occ: u32) {
        // Bytes read while a length field was in effect are the payload that
        // length paid for.
        let read = AccessContext::at(target.pc, target.addr, occ);
        self.state.borrow_mut().bounded_reads_marked += 1;
        self.emit_role(&[read], Role::Payload, 0.8, &[]);
    }

    fn on_table_load(&mut self, _addr_sources: &[AccessContext]) {
        // The value read out of the table is clean, so there is nothing to label
        // here; the event is checksum evidence (see the store path in the engine)
        // and a number worth reporting.
        self.state.borrow_mut().table_loads += 1;
    }

    fn on_checksum(&mut self, sources: &[AccessContext], confidence: f32) {
        self.state.borrow_mut().checksum_events += 1;
        self.emit_role(sources, Role::Checksum, confidence, &[]);
    }
}

pub fn save_role_map(path: &std::path::Path, output: &RoleMapOutput) -> anyhow::Result<()> {
    std::fs::write(path, serde_json::to_vec_pretty(output)?)?;
    Ok(())
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
        let mut map = RoleMap::new();
        map.insert(RoleEntry::new(0x4000, (0, 4), Role::Payload, 0.5));
        map.insert(RoleEntry::new(0x4000, (2, 8), Role::Payload, 0.9));
        assert_eq!(map.len(), 1);
        assert_eq!(map.entries[0].offset_range, (0, 8));
        assert_eq!(map.entries[0].confidence, 0.9);
    }

    #[test]
    fn different_roles_on_one_stream_stay_separate() {
        let mut map = RoleMap::new();
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
        let mut map = RoleMap::new();
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
        let mut map = RoleMap::new();
        map.insert(RoleEntry {
            stream: 0x4000,
            offset_range: (0, 4),
            role: Role::Magic,
            confidence: 0.9,
            discriminants: vec![3, 7],
        });
        let json = map.to_json().unwrap();
        assert_eq!(RoleMap::from_json(&json).unwrap(), map);
    }

    #[test]
    fn ledger_matches_reads_positionally_not_by_value() {
        let ledger = ReadLedger::new();
        // Two identical values at the same site: they must not be confused.
        ledger.record(0x100, 0x4000, 0, &[0x41]);
        ledger.record(0x100, 0x4000, 1, &[0x41]);

        let first = ledger.take_next(0x100, 0x4000, 0).expect("first read");
        assert_eq!(first.input_offset, 0);
        let second = ledger.take_next(0x100, 0x4000, 1).expect("second read");
        assert_eq!(second.input_offset, 1);
        assert!(ledger.take_next(0x100, 0x4000, 2).is_none());
    }

    #[test]
    fn ledger_drops_stale_records_before_the_expected_offset() {
        let ledger = ReadLedger::new();
        ledger.record(0x100, 0x4000, 0, &[1]);
        ledger.record(0x100, 0x4000, 1, &[2]);
        ledger.record(0x100, 0x4000, 7, &[3]);

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
        ledger.record(0x100, 0x4000, 0, &[0x41, 0x42]);

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

        assert!(collector.role_map().is_empty(), "no fragment means no role entry");
        assert_eq!(collector.unbound_loads(), 1);
    }

    #[test]
    fn magic_compare_becomes_a_magic_role_with_its_discriminant() {
        let ledger = ReadLedger::new();
        ledger.record(0x100, 0x4000, 0, &[0x41]);

        let mut collector = RoleCollector::new(ledger, SinkClassifier::new());
        let source = AccessContext::new(0x100, 0x4000);
        collector.on_source_load(source, 1);
        collector.on_magic_compare(&[source], 0x41, 0.85);

        let map = collector.role_map();
        assert_eq!(map.len(), 1);
        assert_eq!(map.entries[0].role, Role::Magic);
        assert_eq!(map.entries[0].discriminants, vec![0x41]);
        assert_eq!(map.entries[0].offset_range, (0, 1));
    }

    #[test]
    fn repeated_magic_compares_accumulate_discriminants() {
        let ledger = ReadLedger::new();
        ledger.record(0x100, 0x4000, 0, &[0x41]);

        let mut collector = RoleCollector::new(ledger, SinkClassifier::new());
        let source = AccessContext::new(0x100, 0x4000);
        collector.on_source_load(source, 1);
        collector.on_magic_compare(&[source], 0x41, 0.85);
        collector.on_magic_compare(&[source], 0x42, 0.85);

        let map = collector.role_map();
        assert_eq!(map.len(), 1, "one range, one role, two discriminants");
        assert_eq!(map.entries[0].discriminants, vec![0x41, 0x42]);
    }

    #[test]
    fn loop_bound_becomes_a_length_role_and_keeps_its_evidence() {
        let ledger = ReadLedger::new();
        ledger.record(0x200, 0x5800_0008, 0, &[4]);

        let mut collector = RoleCollector::new(ledger, SinkClassifier::new());
        let source = AccessContext::new(0x200, 0x5800_0008);
        let target = AccessContext::new(0x300, 0x5800_0000);
        collector.on_source_load(source, 1);
        collector.on_loop_bound(source, target, 4);

        let output = collector.output(1);
        assert_eq!(output.role_map.len(), 1);
        assert_eq!(output.role_map.entries[0].role, Role::Length);
        assert_eq!(output.role_map.entries[0].offset_range, (0, 1));
        assert_eq!(output.report.loop_bounds, 1);
        assert_eq!(output.loop_bounds, vec![LoopBound { source, target, count: 4 }]);
    }

    #[test]
    fn sinked_and_gated_bytes_on_one_stream_keep_separate_roles() {
        let ledger = ReadLedger::new();
        ledger.record(0x100, 0x4000, 0, &[0x41]);
        ledger.record(0x100, 0x4000, 1, &[9, 0, 0, 0]);

        let mut classifier = SinkClassifier::new();
        classifier.insert(0x200, Role::Payload);

        let mut collector = RoleCollector::new(ledger, classifier);
        let source = AccessContext::new(0x100, 0x4000);
        collector.on_source_load(source, 1);
        collector.on_magic_compare(&[source], 0x41, 0.85);
        collector.on_source_load(source, 4);
        collector.on_tainted_store(&[source], 0x200, 0x2000_0000, 4, 9);

        let map = collector.role_map();
        assert_eq!(map.len(), 2, "magic byte and payload bytes are different ranges");
        assert_eq!(map.role_at(0x4000, 0).unwrap().role, Role::Magic);
        assert_eq!(map.role_at(0x4000, 2).unwrap().role, Role::Payload);
    }
}
