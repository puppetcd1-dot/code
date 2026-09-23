//! The consumer side of the role map: what the analysis could justify about each
//! byte of the input, arranged so that a mutator can ask one question at a time.
//!
//! This module is deliberately *only* a consumer.  It reads `role_map.json` -- the
//! frozen contract, whose `ANALYSIS_SCHEMA` says which revision produced it -- and
//! expands it into a per-`(stream, offset)` decision table:
//!
//! | question | source | decision layer |
//! |---|---|---|
//! | `class` / `accept_stream` | `stream_profiles.class` | which stream to spend budget on |
//! | `guard` / `restore_protected` | `gate_constraints` with `state == confirmed` | 1: fix the bit |
//! | `pick_magic` | asserted `magic` entries | 2: semantic (refill) |
//! | `pick_length` | `length` entries | 2: semantic (boundary + joint supply) |
//! | `payload_span` | the confirmed-length formula | 2: joint supply |
//!
//! Nothing here re-derives anything.  The filtering rules the analysis documented
//! are *applied* here, because this is the first consumer that acts on them:
//! a `stream_level` entry is a stream-level constraint and never a position; a
//! value listed in `stream_constraints` is not a positional constant (a delimiter
//! is not a field); a comparison that was demoted never appears in `discriminants`
//! in the first place, and the loader asserts that again rather than trusting it.
//!
//! When there is no contract (no `role_map.json`, an unreadable one, or
//! `TAINT_CONTRACT=0`), every answer is the permissive one and the mutator behaves
//! exactly like the baseline havoc -- which is what makes "havoc" an ablation arm
//! of the same binary rather than a different build.

use std::path::Path;

use hashbrown::HashMap;
use rand::seq::SliceRandom;
use rand::Rng;

use crate::input::StreamKey;

/// What a stream is, from `stream_profiles`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// A data channel: several positions carry constants, or a delimiter class.
    Channel,
    /// A register (or a stream nothing was asserted about, but something stored):
    /// the firmware reading itself.
    Register,
    /// Evidence did not reach a verdict.
    Undecided,
}

impl Class {
    /// How much of the mutation budget a stream of this class should get.
    ///
    /// Multiplicative on the base distribution rather than a replacement for it:
    /// the base already encodes which streams reached new code (colorization), and
    /// the analysis only adds "this one is the device talking to itself".  On the
    /// target that is the 91.1% of input bytes that device-state polls consume.
    fn factor(self) -> f64 {
        match self {
            Class::Channel => 1.0,
            Class::Undecided => 0.6,
            Class::Register => 0.25,
        }
    }

    /// The name the report uses, which is the one the analysis writes.
    pub fn name(self) -> &'static str {
        match self {
            Class::Channel => "channel",
            Class::Register => "register",
            Class::Undecided => "undecided",
        }
    }
}

/// One position-carrying assertion about a stream.
#[derive(Debug, Clone)]
pub enum Field {
    /// A comparison states these constants at this position (`role == magic`).
    Magic { values: Vec<u64>, confidence: f32, stream_level: bool },
    /// The field's value decided how many times another site was read
    /// (`role == length`); `confirmed` means its value equalled that count.
    Length { confirmed: bool },
}

/// What one stream's bytes mean, as far as the analysis could say.
#[derive(Debug, Clone, Default)]
pub struct StreamContract {
    class: Option<Class>,
    /// `(start, end, field)` for every asserted entry, in offset order.
    fields: Vec<(u32, u32, Field)>,
    /// `(start, end, mask)` for every *confirmed* gate: the bytes the tested read
    /// consumed, and the bit of its value the firmware branches on (the reason the
    /// bytes are pinned -- see `guard`).
    gates: Vec<(u32, u32, u64)>,
    payload: Vec<(u32, u32)>,
}

impl StreamContract {
    /// How many positions of this stream a confirmed gate pins.
    ///
    /// Counted as the union over the gate spans, capped at the stream's length: a
    /// gate's span is the read site's bytes, and two gates on one site state the
    /// same span.
    fn pinned_positions(&self, len: usize) -> usize {
        let mut pinned = 0;
        for (start, end, _) in &self.gates {
            let lo = (*start as usize).min(len);
            let hi = (*end as usize).min(len);
            pinned += hi.saturating_sub(lo);
        }
        pinned.min(len)
    }
}

/// A read of one position: the byte range, and the constant a comparison expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Refill {
    pub offset: u32,
    pub value: u64,
    pub width: u32,
    pub confidence: u32,
}

/// A length field, as the mutator needs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LengthField {
    pub start: u32,
    pub width: u32,
    pub value: u64,
    pub confirmed: bool,
}

/// Which decision-tree layer produced a mutation.
///
/// Recorded in the corpus metadata next to the value, so the ablation arms can be
/// read back from the inputs that were produced: "this byte was not guessed" has to
/// be checkable after the fact, not asserted in prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ActionKind {
    /// Layer 2: a constant a comparison expects, written at its position.
    MagicRefill,
    /// Layer 2: a boundary value for a length field.
    LengthBoundary,
    /// Layer 2: the same, with enough bytes supplied behind it to be read.
    LengthJoint,
}

/// An evidence-driven action, ready to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Magic(Refill),
    Length(LengthField),
}

/// The decision table for one run.
#[derive(Debug, Clone, Default)]
pub struct Contract {
    streams: HashMap<StreamKey, StreamContract>,
    /// The stream-level constraint values per stream, by value: the filter the
    /// analysis documented for consumers (`stream_level` is the coarse hint, this
    /// table is the authority).
    constraints: HashMap<StreamKey, Vec<u64>>,
    /// Whether a confirmed length also supplies the bytes behind it.
    ///
    /// `TAINT_JOINT=0` turns that half off, which is the third ablation arm: roles
    /// without the relation (`havoc` is a role map that is absent, this is one
    /// applied without the joint supply).
    joint: bool,
}

impl Contract {
    /// A contract that says nothing: every question gets the permissive answer.
    pub fn empty() -> Self {
        Self { joint: true, ..Self::default() }
    }

    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }

    /// How many streams the table holds.
    ///
    /// Read by the run's startup line, so "the contract loaded" and "it loaded as
    /// nothing" are distinguishable without a log level.
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    /// Read the decision table out of a `role_map.json`.
    ///
    /// A missing or unreadable file is not an error: it means "run the baseline",
    /// which has to stay reachable without editing the build.
    pub fn load(path: &Path) -> Self {
        let Ok(text) = std::fs::read_to_string(path) else {
            warn_once_about_missing(path);
            return Self::empty();
        };
        match Self::parse(&text) {
            Ok(mut contract) => {
                contract.joint = joint_from_env();
                tracing::info!(
                    "contract: {} stream(s) from {}",
                    contract.streams.len(),
                    path.display()
                );
                contract
            }
            Err(err) => {
                tracing::warn!("contract at {} is unusable ({}): baseline havoc", path.display(), err);
                Self::empty()
            }
        }
    }

    /// Parse the fields the mutator consumes, and nothing else.
    ///
    /// The mirror structs below are the contract's *use*: a field the analysis adds
    /// is invisible here until a consumer needs it, and a field this module reads
    /// that the analysis stops writing fails the load instead of silently emptying
    /// the table.
    pub fn parse(text: &str) -> Result<Self, String> {
        let raw: RawOutput = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let mut contract = Contract::empty();
        for constraint in raw.stream_constraints {
            contract.constraints.entry(constraint.stream).or_default().push(constraint.value);
        }
        for profile in raw.stream_profiles {
            let class = match profile.class.as_str() {
                "channel" => Class::Channel,
                "register" => Class::Register,
                _ => Class::Undecided,
            };
            contract.streams.entry(profile.stream).or_default().class = Some(class);
        }
        // The budget's verdict overrides the profile where the profile has none: a
        // stream the analysis did not judge and did not see stored is device state by
        // the byte budget, and mutating it is the budget this stage exists to move
        // away from.
        for stream in raw.device_state_streams {
            let entry = contract.streams.entry(stream).or_default();
            // `Undecided` counts as "the profile has no verdict": it assigns that
            // class to every stream it saw, so keying on `None` alone would make this
            // override dead code.
            if matches!(entry.class, None | Some(Class::Undecided)) {
                entry.class = Some(Class::Register);
            }
        }
        // Layer 1: only a *confirmed* gate fixes a bit.  A candidate is reported
        // (and a human may look at it) and an unobserved one is noise; neither may
        // constrain the mutator, or the analysis' open questions become behaviour.
        for gate in raw.gate_constraints {
            if gate.state != "confirmed" || gate.mask == 0 {
                continue;
            }
            contract
                .streams
                .entry(gate.stream)
                .or_default()
                .gates
                .push((gate.offset_range.0, gate.offset_range.1, gate.mask));
        }
        for entry in raw.role_map.entries {
            let stream = contract.streams.entry(entry.stream).or_default();
            let (start, end) = entry.offset_range;
            if end <= start {
                continue;
            }
            match entry.role.as_str() {
                "magic" => {
                    // Asserted only: the analysis prunes entries with no assertion,
                    // and the loader does not trust that it always will.
                    if entry.discriminants.is_empty() || entry.confidence < 0.5 {
                        continue;
                    }
                    stream.fields.push((
                        start,
                        end,
                        Field::Magic {
                            values: entry.discriminants,
                            confidence: entry.confidence,
                            stream_level: entry.stream_level,
                        },
                    ));
                }
                "length" => {
                    let confirmed = entry.confidence >= 0.8;
                    stream.fields.push((start, end, Field::Length { confirmed }));
                }
                "payload" => stream.payload.push((start, end)),
                _ => {}
            }
        }
        for stream in contract.streams.values_mut() {
            stream.fields.sort_by_key(|(start, _, _)| *start);
        }
        Ok(contract)
    }

    /// Whether to spend this round's budget on `key`.
    ///
    /// The share of the stream the mutator may actually change, as a factor on the
    /// base distribution: what it *is* (`class`), times how much of it is not pinned
    /// by a confirmed gate.
    ///
    /// The second half is not a refinement but the measurement: a confirmed gate's
    /// span is the read site's own bytes, so on the target a 979-byte status-register
    /// region is entirely pinned -- every byte mutation there is put back, and the
    /// budget spent on it is spent undoing itself.  A stream with nothing left to
    /// change gets a factor of zero and is not drawn at all.
    pub fn stream_factor(&self, key: StreamKey, len: usize) -> f64 {
        let Some(stream) = self.streams.get(&key) else { return 1.0 };
        let Some(class) = stream.class else { return 1.0 };
        let pinned = stream.pinned_positions(len);
        if len == 0 {
            return 0.0;
        }
        let free = (len - pinned.min(len)) as f64 / len as f64;
        class.factor() * free
    }

    pub fn class(&self, key: StreamKey) -> Class {
        self.streams.get(&key).and_then(|s| s.class).unwrap_or(Class::Undecided)
    }

    /// Whether the analysis put this stream in the device-state budget.
    ///
    /// This is the class the *weighting* uses, and it is a different question from
    /// the profile's class: on the target sixteen of the seventeen device streams are
    /// `undecided` in the profile (nothing was asserted about them and nothing stored
    /// them), while the budget already knows they are not protocol -- they are the
    /// 91.1% of input bytes the firmware spends on itself.
    pub fn is_device(&self, key: StreamKey) -> bool {
        self.class(key) == Class::Register
    }

    /// How many bytes of this stream the analysis marked as payload.
    ///
    /// Reported per stream so the mutation share can be read next to the byte
    /// budget: the point of the down-weighting is that the two agree.
    pub fn payload_bytes(&self, key: StreamKey) -> u64 {
        self.streams
            .get(&key)
            .map(|stream| {
                stream
                    .payload
                    .iter()
                    .map(|(start, end)| u64::from(end.saturating_sub(*start)))
                    .sum()
            })
            .unwrap_or(0)
    }

    /// How many positions of this stream a confirmed gate pins.
    ///
    /// Reported next to the stream's length: the two together are what says whether
    /// the budget spent on it could change anything.
    pub fn pinned_bytes(&self, key: StreamKey, len: usize) -> usize {
        self.streams.get(&key).map(|stream| stream.pinned_positions(len)).unwrap_or(0)
    }

    /// `(start, end, mask)` for every confirmed gate of this stream.
    ///
    /// The mask is the bit of the *read value* the firmware branches on; the span is
    /// the input the read consumed.  Both are reported, because they answer different
    /// questions: the span is what gets pinned, the mask is why.
    pub fn protected_spans(&self, key: StreamKey) -> Vec<(u32, u32, u64)> {
        self.streams.get(&key).map(|stream| stream.gates.clone()).unwrap_or_default()
    }

    /// The protected positions of a stream, with the values they must keep.
    ///
    /// The span is the read site's own bytes, so it can be long: the target's RXNE
    /// gate covers the 589 bytes the status register was polled from, and pinning
    /// them is the decision (the polling loop then behaves the same way every time).
    /// The bound only guards against a pathological range.
    ///
    /// A whole byte, not the masked bits: the mask is a bit of the value the *model*
    /// returned for that read, and which input byte carries that bit is the model's
    /// business rather than the analysis'.  Pinning the bytes the read consumed is
    /// the conservative reading, and the mask stays in the record as the reason.
    pub fn guard(&self, key: StreamKey, bytes: &[u8]) -> Vec<(u32, u8)> {
        let Some(stream) = self.streams.get(&key) else { return Vec::new() };
        let mut guard = Vec::new();
        for (start, end, _) in &stream.gates {
            for offset in *start..*end {
                if guard.len() >= MAX_GUARD {
                    return guard;
                }
                let Some(byte) = bytes.get(offset as usize) else { break };
                guard.push((offset, *byte));
            }
        }
        guard
    }

    /// Put back the bytes a confirmed gate fixes, whatever the mutation did.
    ///
    /// Whole bytes: see `guard` for why the bit-level version would be a claim the
    /// analysis cannot support.
    pub fn restore_protected(bytes: &mut [u8], guard: &[(u32, u8)]) -> usize {
        let mut restored = 0;
        for (offset, before) in guard {
            let Some(byte) = bytes.get_mut(*offset as usize) else { continue };
            if byte != before {
                *byte = *before;
                restored += 1;
            }
        }
        restored
    }

    /// A position to refill with a constant the firmware compares against.
    ///
    /// The evidence is in the entry: which position, and which of the asserted
    /// constants.  Two documented filters are applied here rather than left to the
    /// caller -- a `stream_level` entry states a property of the whole stream (a
    /// terminator is not a field), and a value the stream's constraint table lists
    /// is not a positional constant even when it shares a position with one.
    pub fn pick_magic<R: Rng>(&self, rng: &mut R, key: StreamKey, len: usize) -> Option<Refill> {
        let stream = self.streams.get(&key)?;
        let constraints = self.constraints.get(&key).map(|v| v.as_slice()).unwrap_or(&[]);
        let candidates: Vec<&(u32, u32, Field)> = stream
            .fields
            .iter()
            .filter(|(start, _, field)| match field {
                Field::Magic { values, stream_level, .. } => {
                    !*stream_level && !values.is_empty() && (*start as usize) < len
                }
                Field::Length { .. } => false,
            })
            .collect();
        let (start, end, field) = *candidates.choose(rng)?;
        let Field::Magic { values, confidence, .. } = field else { return None };
        let positional: Vec<u64> = values
            .iter()
            .copied()
            .filter(|value| !constraints.contains(value))
            .collect();
        let value = *positional.choose(rng)?;
        Some(Refill {
            offset: *start,
            value,
            width: (*end - *start).max(1),
            confidence: (*confidence * 100.0) as u32,
        })
    }

    /// A length field of this stream, with the value it currently holds.
    ///
    /// The value is read out of the bytes rather than remembered from the analysis:
    /// the analysis states *which* position is a length field and whether that was
    /// confirmed, and the mutator needs the value it is about to change.
    pub fn pick_length<R: Rng>(
        &self,
        rng: &mut R,
        key: StreamKey,
        bytes: &[u8],
    ) -> Option<LengthField> {
        let stream = self.streams.get(&key)?;
        let candidates: Vec<&(u32, u32, Field)> = stream
            .fields
            .iter()
            .filter(|(_start, end, field)| {
                matches!(field, Field::Length { .. }) && (*end as usize) <= bytes.len()
            })
            .collect();
        let (start, end, field) = *candidates.choose(rng)?;
        let Field::Length { confirmed } = field else { return None };
        let width = (*end - *start).max(1);
        Some(LengthField {
            start: *start,
            width,
            value: read_value(bytes, *start, width),
            confirmed: *confirmed,
        })
    }

    /// A boundary value for a length field: what firmware parsers get wrong.
    ///
    /// Blind havoc hits `value + 1` with negligible probability, and the failure it
    /// looks for (an off-by-one in the reader) lives exactly there.
    pub fn length_boundary<R: Rng>(&self, rng: &mut R, value: u64, width: u32) -> u64 {
        let saturation = if width >= 8 { u64::MAX } else { (1u64 << (8 * width)) - 1 };
        let candidates = [
            0,
            1,
            value.saturating_sub(1),
            value,
            value.saturating_add(1),
            saturation,
        ];
        *candidates.choose(rng).unwrap_or(&value)
    }

    /// Apply an evidence-driven action.
    ///
    /// Returns `(kind, offset, value)` for the record, or `None` when the bytes
    /// cannot take the action (an empty stream, a position past the end) -- in which
    /// case the caller falls back to a random mutation rather than silently doing
    /// nothing and counting it as an improvement.
    ///
    /// The joint half of the length action only runs for a *confirmed* field: the
    /// analysis refuses to derive a span from an unconfirmed one (its value does not
    /// match the reads it gated, so the interval would be painted over unrelated
    /// bytes), and this side inherits the same rule instead of inventing a second
    /// one.  "The length says 8 and three bytes follow" is not a test of the parser,
    /// it is a test of the bounds check the mutator did not mean to write.
    pub fn apply<R: Rng>(&self, rng: &mut R, action: Action, bytes: &mut Vec<u8>) -> Option<(ActionKind, u32, u64)> {
        match action {
            Action::Magic(refill) => {
                let written = write_value(bytes, refill.offset, refill.width, refill.value);
                written.then_some((ActionKind::MagicRefill, refill.offset, refill.value))
            }
            Action::Length(field) => {
                let value = self.length_boundary(rng, field.value, field.width);
                if value == field.value || !write_value(bytes, field.start, field.width, value) {
                    return None;
                }
                let kind = if field.confirmed && self.joint {
                    let start = field.start.saturating_add(field.width);
                    if let Some((_, end)) = payload_span(start, value, field.width) {
                        if end as usize > bytes.len() {
                            while bytes.len() < end as usize {
                                bytes.push(rng.gen());
                            }
                        }
                    }
                    ActionKind::LengthJoint
                }
                else {
                    ActionKind::LengthBoundary
                };
                Some((kind, field.start, value))
            }
        }
    }
}

/// `TAINT_JOINT=0` disables the joint half of the length action.
fn joint_from_env() -> bool {
    std::env::var("TAINT_JOINT").map(|value| value != "0").unwrap_or(true)
}

/// Say it once per process, not once per havoc round.
///
/// A plain run has no role map and this is asked for every input, so the message
/// cannot be per-round -- but a *silent* fallback to baseline havoc is the one
/// failure mode of this wiring that no artifact would show.
fn warn_once_about_missing(path: &Path) {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        tracing::warn!(
            "no role map at {}: running baseline havoc (point TAINT_CONTRACT at one, or \
             run the analysis in this workdir)",
            path.display()
        );
    });
}

/// Upper bound on the positions one stream's guard restores.  A confirmed gate can
/// cover a whole read site (the target's is 589 bytes), which is intended, so this
/// is a guard against a pathological range rather than a policy.
const MAX_GUARD: usize = 4096;

/// The bytes a confirmed length field pays for.
///
/// One formula, two consumers: the analysis marks the span (`derived_payload`) and
/// the mutator keeps enough bytes for it to be read.  Saturating, not a bare cast:
/// `count * width` wrapped into a *small* interval would silently mark the wrong
/// bytes -- the same class of landmine as a one-way bit flip or a splice that was
/// never written back.
pub fn payload_span(start: u32, count: u64, width: u32) -> Option<(u32, u32)> {
    if width == 0 {
        return None;
    }
    let span = count.saturating_mul(u64::from(width)).min(u64::from(u32::MAX));
    let end = start.saturating_add(span as u32);
    (end > start).then_some((start, end))
}

/// Read a little-endian value of `width` bytes, clamped to the buffer.
pub fn read_value(bytes: &[u8], start: u32, width: u32) -> u64 {
    let mut value = 0u64;
    for index in 0..width.min(8) {
        let Some(byte) = bytes.get(start as usize + index as usize) else { break };
        value |= u64::from(*byte) << (8 * index);
    }
    value
}

/// Write a little-endian value of `width` bytes, clamped to the buffer.
pub fn write_value(bytes: &mut [u8], start: u32, width: u32, value: u64) -> bool {
    let mut written = false;
    for index in 0..width.min(8) {
        let Some(byte) = bytes.get_mut(start as usize + index as usize) else { break };
        *byte = (value >> (8 * index)) as u8;
        written = true;
    }
    written
}

#[derive(serde::Deserialize)]
struct RawOutput {
    role_map: RawRoleMap,
    #[serde(default)]
    stream_profiles: Vec<RawProfile>,
    #[serde(default)]
    gate_constraints: Vec<RawGate>,
    #[serde(default)]
    stream_constraints: Vec<RawConstraint>,
    /// The budget's device-state class: streams whose bytes are not protocol.
    #[serde(default)]
    device_state_streams: Vec<u64>,
}

#[derive(serde::Deserialize)]
struct RawRoleMap {
    entries: Vec<RawEntry>,
}

#[derive(serde::Deserialize)]
struct RawEntry {
    stream: u64,
    offset_range: (u32, u32),
    role: String,
    #[serde(default)]
    confidence: f32,
    #[serde(default)]
    discriminants: Vec<u64>,
    #[serde(default)]
    stream_level: bool,
}

#[derive(serde::Deserialize)]
struct RawProfile {
    stream: u64,
    class: String,
}

#[derive(serde::Deserialize)]
struct RawGate {
    stream: u64,
    offset_range: (u32, u32),
    mask: u64,
    state: String,
}

#[derive(serde::Deserialize)]
struct RawConstraint {
    stream: u64,
    value: u64,
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use super::*;

    /// One stream with the three shapes the refill has to tell apart, one stream
    /// the profile calls a register, and three gates in three states.
    const FIXTURE: &str = r#"{
      "role_map": {"entries": [
        {"stream": 16384, "offset_range": [0, 1], "role": "magic",
         "confidence": 0.85, "discriminants": [13], "stream_level": true},
        {"stream": 16384, "offset_range": [1, 2], "role": "magic",
         "confidence": 0.85, "discriminants": [13, 108], "stream_level": false},
        {"stream": 16384, "offset_range": [2, 3], "role": "magic",
         "confidence": 0.85, "discriminants": [112], "stream_level": false},
        {"stream": 16384, "offset_range": [3, 4], "role": "length",
         "confidence": 0.85, "discriminants": [], "stream_level": false},
        {"stream": 16384, "offset_range": [4, 8], "role": "payload",
         "confidence": 0.8, "discriminants": [], "stream_level": false},
        {"stream": 16388, "offset_range": [0, 1], "role": "payload",
         "confidence": 0.8, "discriminants": [], "stream_level": false}
      ]},
      "stream_profiles": [
        {"stream": 16384, "class": "channel"},
        {"stream": 16388, "class": "register"}
      ],
      "gate_constraints": [
        {"source_pc": 256, "stream": 16388, "offset_range": [0, 4], "mask": 32,
         "branch_pc": 258, "confirmed": 9, "weak": 0, "unobserved": 0,
         "state": "confirmed", "confidence": 0.8},
        {"source_pc": 256, "stream": 16388, "offset_range": [4, 8], "mask": 128,
         "branch_pc": 260, "confirmed": 0, "weak": 3, "unobserved": 0,
         "state": "candidate", "confidence": 0.5},
        {"source_pc": 256, "stream": 16388, "offset_range": [8, 12], "mask": 64,
         "branch_pc": 262, "confirmed": 0, "weak": 0, "unobserved": 4,
         "state": "unobserved", "confidence": 0.5}
      ],
      "stream_constraints": [
        {"stream": 16384, "value": 13, "coverage": 1.0, "positions": 132}
      ]
    }"#;

    fn fixture() -> Contract {
        Contract::parse(FIXTURE).expect("fixture parses")
    }

    /// A refill must not write a delimiter, and must not treat a stream-level entry
    /// as a position: the mixed entry keeps its command letter and loses its
    /// terminator, and the entry whose only value is the terminator is skipped
    /// entirely.
    #[test]
    fn refill_skips_stream_level_entries_and_constraint_values() {
        let contract = fixture();
        let mut rng = StdRng::seed_from_u64(7);
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..200 {
            let refill = contract.pick_magic(&mut rng, 16384, 132).expect("a refill");
            assert_ne!(refill.offset, 0, "the stream-level entry is not a position");
            assert_ne!(refill.value, 13, "a constraint value is not a constant here");
            seen.insert((refill.offset, refill.value));
        }
        assert!(seen.contains(&(1, 108)), "the command letter of the mixed entry: {seen:?}");
        assert!(seen.contains(&(2, 112)), "the positional constant: {seen:?}");
        assert_eq!(seen.len(), 2, "nothing else is offered");
    }

    /// Only a confirmed gate fixes anything, and the unit it fixes is the byte the
    /// read consumed -- the mask says which bit of the read *value* the firmware
    /// branches on, which is the reason, not the unit.
    #[test]
    fn only_a_confirmed_gate_protects_bytes() {
        let contract = fixture();
        assert_eq!(contract.protected_spans(16388), vec![(0, 4, 0x20)]);

        let mut bytes = vec![0xabu8; 12];
        let guard = contract.guard(16388, &bytes);
        assert_eq!(guard.len(), 4, "the confirmed gate's span, and nothing else: {guard:?}");
        bytes[1] = 0x5a;
        bytes[7] = 0x5a;
        assert_eq!(Contract::restore_protected(&mut bytes, &guard), 1);
        assert_eq!(bytes[1], 0xab, "a byte the read consumed goes back");
        assert_eq!(bytes[7], 0x5a, "a candidate's span is not behaviour");
    }

    /// The span formula is the one the analysis marks spans with, and it saturates
    /// instead of wrapping into a small interval.
    #[test]
    fn payload_span_is_the_analysis_formula() {
        assert_eq!(payload_span(4, 3, 1), Some((4, 7)));
        assert_eq!(payload_span(0, u64::MAX, 4), Some((0, u32::MAX)));
        assert_eq!(payload_span(10, 1, 0), None, "a zero width states nothing");
        assert_eq!(payload_span(0, 0, 2), None, "an empty span is not a span");
    }

    /// The class decides the share of the budget, and it never silences a stream
    /// completely: a register still gets mutated, just less often.
    #[test]
    fn classes_order_the_mutation_budget() {
        let contract = fixture();
        let mut rng = StdRng::seed_from_u64(11);
        let (mut channel, mut register) = (0, 0);
        for _ in 0..1000 {
            channel += usize::from(rng.gen_bool(contract.stream_factor(16384, 12)));
            register += usize::from(rng.gen_bool(contract.stream_factor(16388, 12)));
        }
        assert!(channel > register * 2, "channel {channel} vs register {register}");
        // The register carries the confirmed gate over its first four bytes, so two
        // thirds of it are still free to change -- and no more than a quarter of the
        // draws are spent there.
        assert!(register > 50 && register < 250, "register share: {register}");
        // Nothing free means nothing spent: the reason the factor exists is a stream
        // whose every byte a confirmed gate pins, where each mutation is undone.
        assert_eq!(contract.stream_factor(16388, 4), 0.0, "fully pinned");
        assert_eq!(contract.pinned_bytes(16388, 12), 4);
        assert_eq!(contract.class(16384), Class::Channel);
        assert_eq!(contract.class(16388), Class::Register);
        assert_eq!(contract.class(0xdead), Class::Undecided);
        assert_eq!(contract.payload_bytes(16384), 4);
    }

    /// A length field's value comes from the bytes, and the boundary set is the one
    /// that finds off-by-one readers.
    #[test]
    fn length_field_reads_its_value_and_offers_boundaries() {
        let contract = fixture();
        let mut rng = StdRng::seed_from_u64(13);
        let bytes = [0u8, 0, 0, 5, 0, 0, 0, 0];
        let field = contract.pick_length(&mut rng, 16384, &bytes).expect("a length field");
        assert_eq!((field.start, field.width, field.value, field.confirmed), (3, 1, 5, true));

        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..200 {
            seen.insert(contract.length_boundary(&mut rng, field.value, field.width));
        }
        assert_eq!(seen, [0, 1, 4, 5, 6, 255].into_iter().collect());
    }

    /// No contract, an unreadable one and a truncated one all mean "baseline": the
    /// ablation arm has to be reachable without editing the build.
    #[test]
    fn a_missing_or_broken_contract_is_the_baseline() {
        let missing = Contract::load(Path::new("this-file-does-not-exist.json"));
        assert!(missing.is_empty());
        let mut rng = StdRng::seed_from_u64(3);
        assert_eq!(missing.stream_factor(16388, 12), 1.0, "nothing is down-weighted");
        assert!(missing.pick_magic(&mut rng, 16388, 8).is_none());
        assert!(missing.guard(16388, &[0u8; 8]).is_empty());
        assert!(Contract::parse("{ not json").is_err());
    }
}
