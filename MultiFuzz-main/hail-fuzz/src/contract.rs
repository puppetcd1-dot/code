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

/// The confidence at or above which the analysis *asserted* a field, i.e. the
/// threshold `parse` turns into behaviour.
///
/// Named because a fixture that hard-codes the number can drift away from it and
/// then the test passes for the wrong reason -- which is exactly what happened to
/// `has_length`: its fixture used 0.85 while the threshold lived in a literal, so the
/// assertion and the mechanism were not testing the same thing.
pub const CONFIRMED_F: f32 = 0.8;

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
    /// `(start, end, mask, expected)` for every *confirmed* gate: the bytes the tested
    /// read consumed, the bit of its value the firmware branches on (the reason the
    /// bytes are pinned -- see `guard`), and the state that read was observed in.
    ///
    /// The last element is what turns pinning into asserting: a lineage that has never
    /// taken this branch can be *put* into the state the firmware reached it in,
    /// instead of only being prevented from leaving it.
    gates: Vec<(u32, u32, u64, Option<Expected>)>,
    payload: Vec<(u32, u32)>,
}

impl StreamContract {
    /// How many positions of this stream a confirmed gate pins.
    ///
    /// Counted as the union over the gate spans, capped at the stream's length: a
    /// gate's span is the read site's bytes, and two gates on one site state the
    /// same span.
    fn pinned_positions(&self, len: usize) -> usize {
        // The *union* of the gate spans, not the sum: two gates on one read site state
        // the same bytes, and adding them would count those bytes twice -- which would
        // report a stream as more pinned than it is long, and `stream_factor` would
        // discount it that much harder for a reason that is an artefact of bookkeeping.
        let mut spans: Vec<(usize, usize)> = self
            .gates
            .iter()
            .map(|(start, end, _, _)| {
                ((*start as usize).min(len), (*end as usize).min(len))
            })
            .filter(|(lo, hi)| hi > lo)
            .collect();
        spans.sort_unstable();
        let mut pinned = 0;
        let mut current: Option<(usize, usize)> = None;
        for (lo, hi) in spans {
            match current {
                Some((_, end)) if lo <= end => {
                    current = Some((current.expect("just matched").0, end.max(hi)));
                }
                Some((start, end)) => {
                    pinned += end.saturating_sub(start);
                    current = Some((lo, hi));
                }
                None => current = Some((lo, hi)),
            }
        }
        if let Some((start, end)) = current {
            pinned += end.saturating_sub(start);
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

/// The state a confirmed gate was entered in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Expected {
    /// What the tested read returned.
    pub value: u64,
    /// How many input bytes that read consumed.
    ///
    /// Never the gate's span: a span is the union over every sample of the site's
    /// input (589 bytes for the target's RXNE gate) while the value is one read, so
    /// writing the span would zero bytes the analysis never saw in this value.
    pub width: u8,
}

/// Which decision-tree layer produced a mutation.
///
/// Recorded in the corpus metadata next to the value, so the ablation arms can be
/// read back from the inputs that were produced: "this byte was not guessed" has to
/// be checkable after the fact, not asserted in prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionKind {
    /// Layer 2: a constant a comparison expects, written at its position.
    MagicRefill,
    /// Layer 2: a boundary value for a length field.
    LengthBoundary,
    /// Layer 2: the same, with enough bytes supplied behind it to be read.
    LengthJoint,
    /// Layer 1: a stream-level constraint value appended to the stream, so the
    /// obligation the analysis' `stream_constraints` table states -- "this stream
    /// carries this value" -- is the mutator's job and not only a filter.
    DelimPlace,
    /// Layer 1, the assert half: a confirmed gate's observed value written back, so a
    /// lineage that never took that branch can still enter the code behind it.
    GateAssert,
}

/// An evidence-driven action, ready to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Magic(Refill),
    Length(LengthField),
    /// Append a value the stream is known to carry (a delimiter, a separator).
    Delim(u64),
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
    /// Whether a stream's constraint values are supplied when they are missing.
    ///
    /// `TAINT_DELIM=0` turns it off, so "the delimiter is written on purpose" is an
    /// arm of its own rather than something the contract always does.
    delim: bool,
    /// Whether a confirmed gate's observed state is written back before mutating.
    ///
    /// `TAINT_ASSERT=0` leaves pinning in place and turns the assert half off: the
    /// difference between "this lineage cannot leave the state" and "it is put into
    /// it" is exactly what the arm is meant to measure.
    assert: bool,
}

impl Contract {
    /// A contract that says nothing: every question gets the permissive answer.
    pub fn empty() -> Self {
        Self { joint: true, delim: true, assert: true, ..Self::default() }
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
                contract.delim = delim_from_env();
                contract.assert = assert_from_env();
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
        // The artifact says which revision produced it, and a consumer that reads a
        // field the writer has stopped writing would otherwise run with a silently
        // emptier table -- the one failure mode no artifact of *this* run can show.
        let expected = crate::semantic_taint::ANALYSIS_SCHEMA;
        if raw.report.schema != expected {
            return Err(format!(
                "role_map schema {} != the consumer's {}",
                raw.report.schema, expected
            ));
        }
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
            let width = gate.offset_range.1.saturating_sub(gate.offset_range.0);
            let full = match width {
                0 => 0,
                8.. => u64::MAX,
                _ => (1u64 << (8 * width)) - 1,
            };
            if gate.mask == full {
                // A mask that covers the whole read is not a bit test: it says "this
                // byte equals a constant", which is a comparison the magic rule owns.
                // Seen on the target as memchr's `eor`-to-zero loop
                // (`0x8005138 eor r3,r3,r1; 0x800513c cbz`), whose 0xff would otherwise
                // pin every byte of the read on the strength of a comparison.
                continue;
            }
            contract
                .streams
                .entry(gate.stream)
                .or_default()
                .gates
                .push((
                    gate.offset_range.0,
                    gate.offset_range.1,
                    gate.mask,
                    gate.expected_value.map(|value| Expected {
                        value,
                        // A role map from before the width existed reports a value
                        // without one; a single byte is the conservative reading.
                        width: gate.expected_width.unwrap_or(1).max(1),
                    }),
                ));
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
                    let confirmed = entry.confidence >= CONFIRMED_F;
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
        self.streams
            .get(&key)
            .map(|stream| {
                stream.gates.iter().map(|(start, end, mask, _)| (*start, *end, *mask)).collect()
            })
            .unwrap_or_default()
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
        for (start, end, _, _) in &stream.gates {
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
            .filter(|(start, end, field)| match field {
                Field::Magic { values, stream_level, .. } => {
                    // The whole field has to fit: a partial write is not a refill, and
                    // counting it as one would put a write in the Evidence record that
                    // never happened.
                    let fits = (*start as usize).saturating_add((*end - *start) as usize) <= len;
                    // ... and it must not land inside a *same-stream* confirmed gate: the
                    // guard would undo it in the same round, so the action and its undo
                    // would be one event -- which is exactly the bookkeeping the record
                    // is supposed to rule out.  (A gate on another stream says nothing
                    // about this one.)
                    let outside_gates = !stream
                        .gates
                        .iter()
                        .any(|(guard_start, guard_end, _, _)| *start >= *guard_start && *start < *guard_end);
                    !*stream_level && !values.is_empty() && fits && outside_gates
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
    ///
    /// Only a *confirmed* field is offered.  The target's eight length roles are all
    /// unconfirmed (they are the terminator loop's residue: their value never equalled
    /// the reads they gated), and mutating those boundaries is work spent on an
    /// artefact -- a documented zero is worth more than 680 actions that cannot be
    /// attributed.  Unproven fields can come back as their own counted arm if a target
    /// ever makes them interesting.
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
                matches!(field, Field::Length { confirmed: true })
                    && (*end as usize) <= bytes.len()
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

    /// A constraint value to append to this stream, if it is a channel.
    ///
    /// `stream_constraints` states a property of the whole stream -- "a `\\r` appears
    /// somewhere in it" -- and the refill half of this module only knows how to *not*
    /// write those values as if they were positional constants.  Nothing was putting
    /// them in, which showed up as a corpus with 38 `\\n` and zero `\\r`: the firmware's
    /// line discipline was never given the byte it compares against.
    pub fn pick_delim<R: Rng>(&self, rng: &mut R, key: StreamKey) -> Option<u64> {
        if !self.has_delim(key) {
            return None;
        }
        let values = self.constraints.get(&key)?;
        values.choose(rng).copied()
    }

    /// Whether a delimiter can be supplied to this stream at all (see `pick_delim`).
    ///
    /// The mutation side asks this to weight its layers: a stream with no constraint
    /// table should not have a share of the semantic budget spent asking it for one.
    pub fn has_delim(&self, key: StreamKey) -> bool {
        self.delim
            && self.class(key) == Class::Channel
            && self.constraints.get(&key).map(|values| !values.is_empty()).unwrap_or(false)
    }

    /// Whether this stream has a position a constant is expected at.
    ///
    /// Read by the mutation side to *weight the layers against each other*: drawing
    /// uniformly between a refill and a length boundary throws half the semantic
    /// budget away on a stream that only has one of the two, and on the target (eight
    /// unconfirmed length fields, twelve asserted constants) that is not a corner case.
    pub fn has_magic(&self, key: StreamKey) -> bool {
        self.streams
            .get(&key)
            .map(|stream| stream.fields.iter().any(|(_, _, field)| matches!(field, Field::Magic { .. })))
            .unwrap_or(false)
    }

    /// Whether this stream has a *confirmed* length field (what `pick_length` offers).
    pub fn has_length(&self, key: StreamKey) -> bool {
        self.streams
            .get(&key)
            .map(|stream| {
                stream
                    .fields
                    .iter()
                    .any(|(_, _, field)| matches!(field, Field::Length { confirmed: true }))
            })
            .unwrap_or(false)
    }

    /// The confirmed gates of this stream that can be *asserted*, as
    /// `(start, end, mask, expected)`.
    ///
    /// A gate without an observed value can only be pinned: the analysis never saw
    /// the firmware enter it, so there is no state to replay.
    pub fn asserted_gates(&self, key: StreamKey) -> Vec<(u32, u32, u64, Expected)> {
        if !self.assert {
            return Vec::new();
        }
        self.streams
            .get(&key)
            .map(|stream| {
                stream
                    .gates
                    .iter()
                    .filter_map(|(start, end, mask, expected)| {
                        expected.map(|expected| (*start, *end, *mask, expected))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Every asserted gate, as `(stream, start, end, expected)`.
    ///
    /// The per-stream form is what the mutation loop needs; this is what the round's
    /// assert pass needs, because that pass is not about a chosen stream -- it is about
    /// the state the firmware was observed in, and a stream whose bytes are otherwise
    /// pinned is exactly the one that needs it.
    pub fn asserted_gates_all(&self) -> Vec<(StreamKey, u32, u32, Expected)> {
        if !self.assert {
            return Vec::new();
        }
        let mut all = Vec::new();
        for (key, stream) in &self.streams {
            for (start, end, _, expected) in &stream.gates {
                if let Some(expected) = expected {
                    all.push((*key, *start, *end, *expected));
                }
            }
        }
        all.sort_by_key(|(key, start, _, _)| (*key, *start));
        all
    }

    /// Every constant the analysis saw a comparison state, per stream, with the width
    /// of the field it was stated at.
    ///
    /// Injected into the fuzzer's *per-stream* dictionary: a single-byte entry composed
    /// with the corpus' own bytes is how a command name is assembled from letters the
    /// analysis only ever saw one at a time, and it is what keeps a random mutation on
    /// that stream inside the alphabet instead of knocking the field out of it.
    ///
    /// Per stream and not global: the letters are one channel's evidence, and a byte
    /// that is a command letter on the console is not a constant anywhere else.  The
    /// values carry `pick_magic`'s filters (asserted, not a stream constraint), so the
    /// dictionary states exactly what the contract is willing to write -- supplying a
    /// delimiter stays `DelimPlace`'s job, which keeps `TAINT_DELIM`'s attribution
    /// clean.
    pub fn magic_values(&self) -> Vec<(StreamKey, Vec<(u64, u8)>)> {
        let mut by_stream = Vec::new();
        for (key, stream) in &self.streams {
            let constraints = self.constraints.get(key).map(|v| v.as_slice()).unwrap_or(&[]);
            let mut values = Vec::new();
            for (start, end, field) in &stream.fields {
                if let Field::Magic { values: asserted, stream_level: false, .. } = field {
                    // A multi-byte constant goes into the dictionary as the whole field
                    // value, little-endian: the same order `write_value` uses, so the
                    // dictionary and the refill cannot disagree about a field's bytes.
                    let width = (end.saturating_sub(*start)).clamp(1, 8) as u8;
                    for value in asserted {
                        let entry = (*value, width);
                        if !constraints.contains(value) && !values.contains(&entry) {
                            values.push(entry);
                        }
                    }
                }
            }
            if !values.is_empty() {
                by_stream.push((*key, values));
            }
        }
        by_stream.sort_by_key(|(key, _)| *key);
        by_stream
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
                // Cap before writing, and cap the *field value* with it: a length the
                // stream cannot hold states a relation the payload does not satisfy
                // ("4 GiB declared, 1 KiB present"), and the extension below would try
                // to materialise it.
                let payload_start = field.start.saturating_add(field.width) as usize;
                // In *elements*, because the bytes it implies are `cap * width`: capping
                // the count at the room left would still allow `width` times the stream
                // to be materialised.
                let room = crate::config::MAX_STREAM_LEN.saturating_sub(payload_start);
                let cap = (room / field.width.max(1) as usize) as u64;
                let value = value.min(cap);
                if value == field.value || !write_value(bytes, field.start, field.width, value) {
                    return None;
                }
                let kind = if field.confirmed && self.joint {
                    if let Some((_, end)) = payload_span(payload_start as u32, value, field.width) {
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
            Action::Delim(value) => {
                // Appended, not written over a byte: the obligation is "this stream
                // carries the value", and overwriting would be a positional claim the
                // analysis explicitly did not make.  At the size cap the append
                // degrades to the last byte, which is the only place left.
                let limit = crate::config::MAX_STREAM_LEN;
                if bytes.len() >= limit {
                    let last = bytes.len() - 1;
                    bytes[last] = value as u8;
                    Some((ActionKind::DelimPlace, last as u32, value))
                }
                else {
                    bytes.push(value as u8);
                    Some((ActionKind::DelimPlace, (bytes.len() - 1) as u32, value))
                }
            }
        }
    }
}

/// `TAINT_JOINT=0` disables the joint half of the length action.
fn joint_from_env() -> bool {
    std::env::var("TAINT_JOINT").map(|value| value != "0").unwrap_or(true)
}

/// `TAINT_DELIM=0` disables supplying a stream's constraint values.
fn delim_from_env() -> bool {
    std::env::var("TAINT_DELIM").map(|value| value != "0").unwrap_or(true)
}

/// `TAINT_ASSERT=0` disables writing a confirmed gate's observed state back.
fn assert_from_env() -> bool {
    std::env::var("TAINT_ASSERT").map(|value| value != "0").unwrap_or(true)
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
    // All of the field or none of it: a partial write reported as success is how a
    // counter ends up recording a refill that never landed.
    let width = width.min(8) as usize;
    let Some(end) = (start as usize).checked_add(width) else { return false };
    if end > bytes.len() {
        return false;
    }
    for index in 0..width {
        bytes[start as usize + index] = (value >> (8 * index)) as u8;
    }
    true
}

#[derive(serde::Deserialize)]
struct RawOutput {
    report: RawReport,
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
struct RawReport {
    schema: u32,
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
    /// Absent in role maps from before this field existed, and `None` is also a real
    /// value (a gate whose samples never bound a fragment), so the two collapse to the
    /// same behaviour: pin only.
    #[serde(default)]
    expected_value: Option<u64>,
    /// How many input bytes the read that produced `expected_value` consumed.
    #[serde(default)]
    expected_width: Option<u8>,
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
         "confidence": 0.75, "discriminants": [], "stream_level": false},
        {"stream": 16388, "offset_range": [3, 4], "role": "length",
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
         "expected_value": 32, "expected_width": 4,
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
        parsed(FIXTURE)
    }

    /// A fixture with the schema this consumer expects.
    ///
    /// Every fixture goes through here, so the version check is not what the other tests
    /// are testing -- and so a fixture cannot silently keep passing after the schema moves.
    fn with_schema(body: &str) -> String {
        assert!(body.starts_with('{'), "fixtures are JSON objects");
        format!(
            "{{\"report\": {{\"schema\": {}}},{}",
            crate::semantic_taint::ANALYSIS_SCHEMA,
            &body[1..]
        )
    }

    fn parsed(body: &str) -> Contract {
        Contract::parse(&with_schema(body)).expect("fixture parses")
    }

    /// The three shapes that must *not* produce behaviour: an unconfirmed length
    /// field, a mask that covers the whole read, and a confirmed gate the analysis
    /// never saw do anything (so there is no value to assert).
    const NEGATIVE_FIXTURE: &str = r#"{
      "role_map": {"entries": [
        {"stream": 16384, "offset_range": [3, 4], "role": "length",
         "confidence": 0.5, "discriminants": [], "stream_level": false}
      ]},
      "stream_profiles": [{"stream": 16384, "class": "channel"}],
      "gate_constraints": [
        {"source_pc": 256, "stream": 16388, "offset_range": [0, 1], "mask": 255,
         "branch_pc": 258, "confirmed": 4, "weak": 0, "unobserved": 0,
         "expected_value": 255, "state": "confirmed", "confidence": 0.8},
        {"source_pc": 256, "stream": 16388, "offset_range": [2, 4], "mask": 32,
         "branch_pc": 258, "confirmed": 4, "weak": 0, "unobserved": 0,
         "state": "confirmed", "confidence": 0.8}
      ],
      "stream_constraints": []
    }"#;

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
        assert_eq!(
            contract.asserted_gates(16388),
            vec![(0, 4, 0x20, Expected { value: 32, width: 4 })],
            "a confirmed gate with an observed value can be asserted, not only pinned"
        );

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
        let field = contract.pick_length(&mut rng, 16388, &bytes).expect("a length field");
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

    /// The constraint table is an obligation for the mutator, not only a filter: the
    /// value the analysis says the stream carries is *supplied* when it is missing.
    #[test]
    fn a_delimiter_is_appended_and_only_from_the_constraint_table() {
        let contract = fixture();
        let mut rng = StdRng::seed_from_u64(5);
        assert_eq!(contract.pick_delim(&mut rng, 16384), Some(13), "the stream's constraint");
        assert_eq!(contract.pick_delim(&mut rng, 16388), None, "a register carries none");

        let mut bytes = b"p".to_vec();
        assert_eq!(
            contract.apply(&mut rng, Action::Delim(13), &mut bytes),
            Some((ActionKind::DelimPlace, 1, 13))
        );
        assert_eq!(bytes, b"p\r", "appended: the byte that was there is still there");
        assert_eq!(
            contract.magic_values(),
            vec![(16384, vec![(108, 1), (112, 1)])],
            "the positional constants, per stream: the dictionary they feed is per stream"
        );
    }

    /// `TAINT_DELIM=0` is an arm of its own, so the obligation has to be switchable.
    #[test]
    fn the_delimiter_obligation_can_be_switched_off() {
        let contract = Contract { delim: false, ..fixture() };
        let mut rng = StdRng::seed_from_u64(5);
        assert!(contract.pick_delim(&mut rng, 16384).is_none());
    }

    /// `TAINT_ASSERT=0` turns the assert half off and leaves the pin alone, so the arm
    /// measures "put into the state" and not "protected at all".
    #[test]
    fn the_assert_half_can_be_switched_off() {
        let contract = Contract { assert: false, ..fixture() };
        assert!(contract.asserted_gates(16388).is_empty());
        assert!(
            !contract.guard(16388, &[0u8; 8]).is_empty(),
            "the pin is unchanged: the two halves are separate"
        );
    }

    /// Three shapes that must stay silent: a length field the analysis could not
    /// confirm, a mask that covers the whole read, and a gate nothing was observed
    /// behind.
    #[test]
    fn unconfirmed_fields_full_width_masks_and_unseen_gates_do_nothing() {
        let contract = parsed(NEGATIVE_FIXTURE);
        let mut rng = StdRng::seed_from_u64(9);
        assert!(
            contract.pick_length(&mut rng, 16384, &[0, 0, 0, 5]).is_none(),
            "an unconfirmed length is not a length field yet"
        );
        assert_eq!(
            contract.protected_spans(16388),
            vec![(2, 4, 0x20)],
            "0xff covers the whole read: a comparison, not a bit test"
        );
        assert!(
            contract.asserted_gates(16388).is_empty(),
            "no observed value means nothing to assert"
        );
        assert!(contract.pick_delim(&mut rng, 16384).is_none(), "no constraint table");
    }

    /// Two gates on one read site state the same bytes: counting them twice would report
    /// a stream as more pinned than it is long, and `stream_factor` would discount it for
    /// an artefact of bookkeeping.
    #[test]
    fn overlapping_gate_spans_are_counted_once() {
        const OVERLAP: &str = r#"{
          "role_map": {"entries": []},
          "stream_profiles": [{"stream": 16388, "class": "register"}],
          "gate_constraints": [
            {"source_pc": 256, "stream": 16388, "offset_range": [0, 48], "mask": 1,
             "branch_pc": 258, "confirmed": 5, "weak": 0, "unobserved": 0,
             "state": "confirmed", "confidence": 0.8},
            {"source_pc": 256, "stream": 16388, "offset_range": [24, 64], "mask": 2,
             "branch_pc": 258, "confirmed": 5, "weak": 0, "unobserved": 0,
             "state": "confirmed", "confidence": 0.8}
          ],
          "stream_constraints": []
        }"#;
        let contract = parsed(OVERLAP);
        assert_eq!(contract.pinned_bytes(16388, 100), 64, "the union, not 48 + 40");
        assert_eq!(contract.pinned_bytes(16388, 40), 40, "capped at the stream's length");
        assert!(
            (contract.stream_factor(16388, 100) - 0.09).abs() < 1e-6,
            "0.25 class weight times the 36% of the stream that is free: got {}",
            contract.stream_factor(16388, 100)
        );
    }

    /// The layer mix follows what the stream has: half the semantic draws were being
    /// thrown away on a stream with no length field.
    #[test]
    fn the_semantic_layers_are_weighted_by_what_the_stream_has() {
        let contract = fixture();
        assert!(contract.has_magic(16384), "the console has asserted constants");
        assert!(
            !contract.has_length(16384),
            "its length entry is below the confirmed threshold, on purpose"
        );
        assert!(
            contract.has_length(16388),
            "and its twin above the threshold is confirmed, or the test is not testing the threshold"
        );
        assert!(!contract.has_magic(16388));
    }

    /// A contract written by another revision is refused rather than read as far as it
    /// goes: the fields this consumer needs may simply be missing.
    #[test]
    fn a_contract_from_another_schema_is_refused() {
        let stale = FIXTURE.replacen("{\"role_map\"", "{\"report\": {\"schema\": 8}, \"role_map\"", 1);
        assert!(Contract::parse(&stale).is_err(), "schema 8 must not reach the mutator");
        assert!(Contract::parse(&with_schema(FIXTURE)).is_ok());
    }

    /// A field that does not fit, and a field inside a same-stream confirmed gate: neither
    /// may be offered.  The first would be recorded as a write that never happened; the
    /// second would be undone by the guard in the same round.
    #[test]
    fn refills_that_cannot_land_are_not_offered() {
        const EDGES: &str = r#"{
          "role_map": {"entries": [
            {"stream": 16384, "offset_range": [0, 4], "role": "magic",
             "confidence": 0.85, "discriminants": [112], "stream_level": false},
            {"stream": 16386, "offset_range": [2, 6], "role": "magic",
             "confidence": 0.85, "discriminants": [112], "stream_level": false}
          ]},
          "stream_profiles": [{"stream": 16384, "class": "channel"}],
          "gate_constraints": [
            {"source_pc": 256, "stream": 16384, "offset_range": [0, 4], "mask": 32,
             "branch_pc": 258, "confirmed": 5, "weak": 0, "unobserved": 0,
             "expected_value": 32, "expected_width": 4,
             "state": "confirmed", "confidence": 0.8}
          ],
          "stream_constraints": []
        }"#;
        let contract = parsed(EDGES);
        let mut rng = StdRng::seed_from_u64(17);
        assert!(
            contract.pick_magic(&mut rng, 16386, 3).is_none(),
            "a four-byte field with three bytes left is not a refill"
        );
        assert!(
            contract.pick_magic(&mut rng, 16384, 10).is_none(),
            "a refill inside the stream's own gate would be undone in the same round"
        );
    }

    /// A partial write is not a write, so the counter must not record one.
    #[test]
    fn write_value_writes_all_of_the_field_or_none_of_it() {
        let mut bytes = vec![0u8; 4];
        assert!(write_value(&mut bytes, 0, 4, 0x1234_5678));
        assert_eq!(bytes, vec![0x78, 0x56, 0x34, 0x12], "little-endian, as the refill writes");
        let mut bytes = vec![0u8; 4];
        assert!(!write_value(&mut bytes, 2, 4, 0x1234_5678), "it does not fit");
        assert_eq!(bytes, vec![0, 0, 0, 0], "and nothing was written");
    }

    /// A length may not declare more than the stream can hold: the field's value and the
    /// payload that exists have to agree, or the relation is one the mutator invented.
    #[test]
    fn a_length_cannot_declare_more_than_the_stream_can_hold() {
        let contract = fixture();
        let mut rng = StdRng::seed_from_u64(21);
        let field = LengthField { start: 8, width: 8, value: 0, confirmed: true };
        let mut bytes = vec![0u8; 16];
        let cap = (crate::config::MAX_STREAM_LEN.saturating_sub(16) / 8) as u64;
        let mut applied = None;
        for _ in 0..400 {
            if let Some(applied) = contract.apply(&mut rng, Action::Length(field), &mut bytes) {
                if applied.2 == cap {
                    applied = Some(applied);
                    break;
                }
            }
        }
        assert_eq!(applied.map(|a| a.2), Some(cap), "the saturating boundary is drawn eventually");
        assert_eq!(bytes.len(), crate::config::MAX_STREAM_LEN, "the stream stops at the cap");
        assert_eq!(read_value(&bytes, 8, 8), cap, "the field states the payload that exists");
    }
}
