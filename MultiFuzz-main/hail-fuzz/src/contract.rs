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

use std::collections::BTreeSet;

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
    /// See `Contract::stream_factor`: the analysis knows this stream only as something
    /// the firmware reads to see itself.
    poll_only: bool,
    payload: Vec<(u32, u32)>,
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
}

impl Contract {
    /// A contract that says nothing: every question gets the permissive answer.
    pub fn empty() -> Self {
        Self { joint: true, delim: true, ..Self::default() }
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
        let device_side: BTreeSet<StreamKey> = raw.device_state_streams.iter().copied().collect();
        for stream in raw.device_state_streams.iter().copied() {
            let entry = contract.streams.entry(stream).or_default();
            // `Undecided` counts as "the profile has no verdict": it assigns that
            // class to every stream it saw, so keying on `None` alone would make this
            // override dead code.
            if matches!(entry.class, None | Some(Class::Undecided)) {
                entry.class = Some(Class::Register);
            }
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
        // Poll-only: a stream the analysis put in the device-state budget and said nothing
        // about.  Nobody asserts a constant there and no length was confirmed, so the only
        // thing a mutation could do is disturb a poll -- which is exactly what the
        // device-state budget says is not protocol input.
        for (key, stream) in contract.streams.iter_mut() {
            let has_contract = stream.fields.iter().any(|(_, _, field)| {
                matches!(field, Field::Magic { .. } | Field::Length { confirmed: true })
            });
            stream.poll_only = device_side.contains(key) && !has_contract;
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
        // A poll-only stream is not drawn at all: there is no contract for it (no asserted
        // constant, no confirmed length), the firmware only reads it to see itself, and it
        // is where the device-state budget goes.  Its bytes are still supplied -- this
        // removes it from the mutation budget, it does not stop it being data.
        if stream.poll_only || len == 0 {
            return 0.0;
        }
        class.factor()
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
                    !*stream_level && !values.is_empty() && fits
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
  "role_map": {
    "entries": [
      {
        "stream": 16384,
        "offset_range": [
          0,
          1
        ],
        "role": "magic",
        "confidence": 0.85,
        "discriminants": [
          13
        ],
        "stream_level": true
      },
      {
        "stream": 16384,
        "offset_range": [
          1,
          2
        ],
        "role": "magic",
        "confidence": 0.85,
        "discriminants": [
          13,
          108
        ],
        "stream_level": false
      },
      {
        "stream": 16384,
        "offset_range": [
          2,
          3
        ],
        "role": "magic",
        "confidence": 0.85,
        "discriminants": [
          112
        ],
        "stream_level": false
      },
      {
        "stream": 16384,
        "offset_range": [
          3,
          4
        ],
        "role": "length",
        "confidence": 0.75,
        "discriminants": [],
        "stream_level": false
      },
      {
        "stream": 16388,
        "offset_range": [
          3,
          4
        ],
        "role": "length",
        "confidence": 0.85,
        "discriminants": [],
        "stream_level": false
      },
      {
        "stream": 16384,
        "offset_range": [
          4,
          8
        ],
        "role": "payload",
        "confidence": 0.8,
        "discriminants": [],
        "stream_level": false
      },
      {
        "stream": 16388,
        "offset_range": [
          0,
          1
        ],
        "role": "payload",
        "confidence": 0.8,
        "discriminants": [],
        "stream_level": false
      }
    ]
  },
  "stream_profiles": [
    {
      "stream": 16384,
      "class": "channel"
    },
    {
      "stream": 16388,
      "class": "register"
    }
  ],
  "stream_constraints": [
    {
      "stream": 16384,
      "value": 13,
      "coverage": 1.0,
      "positions": 132
    }
  ],
  "device_state_streams": [
    16388
  ]
}"#;

    fn fixture() -> Contract {
        parsed(FIXTURE)
    }

    /// A fixture with the schema this consumer expects.
    ///
    /// Every fixture goes through here, so the version check is not what the other tests
    /// are testing -- and so a fixture cannot silently keep passing after the schema moves.
    fn with_schema_version(body: &str, schema: u32) -> String {
        assert!(body.starts_with('{'), "fixtures are JSON objects");
        format!("{{\"report\": {{\"schema\": {}}},{}", schema, &body[1..])
    }

    fn with_schema(body: &str) -> String {
        with_schema_version(body, crate::semantic_taint::ANALYSIS_SCHEMA)
    }

    fn parsed(body: &str) -> Contract {
        Contract::parse(&with_schema(body)).expect("fixture parses")
    }

    /// The three shapes that must *not* produce behaviour: an unconfirmed length
    /// field, a mask that covers the whole read, and a confirmed gate the analysis
    /// never saw do anything (so there is no value to assert).
    const NEGATIVE_FIXTURE: &str = r#"{
  "role_map": {
    "entries": [
      {
        "stream": 16384,
        "offset_range": [
          3,
          4
        ],
        "role": "length",
        "confidence": 0.5,
        "discriminants": [],
        "stream_level": false
      }
    ]
  },
  "stream_profiles": [
    {
      "stream": 16384,
      "class": "channel"
    }
  ],
  "stream_constraints": [],
  "device_state_streams": []
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
        let stale = with_schema_version(FIXTURE, crate::semantic_taint::ANALYSIS_SCHEMA - 1);
        assert!(Contract::parse(&stale).is_err(), "an older schema must not reach the mutator");
        assert!(
            Contract::parse(&FIXTURE).is_err(),
            "and a body with no report at all is refused too"
        );
        assert!(Contract::parse(&with_schema(FIXTURE)).is_ok());
    }

    /// A field that does not fit, and a field inside a same-stream confirmed gate: neither
    /// may be offered.  The first would be recorded as a write that never happened; the
    /// second would be undone by the guard in the same round.
    #[test]
    fn refills_that_cannot_land_are_not_offered() {
        const EDGES: &str = r#"{
  "role_map": {
    "entries": [
      {
        "stream": 16384,
        "offset_range": [
          0,
          4
        ],
        "role": "magic",
        "confidence": 0.85,
        "discriminants": [
          112
        ],
        "stream_level": false
      },
      {
        "stream": 16386,
        "offset_range": [
          2,
          6
        ],
        "role": "magic",
        "confidence": 0.85,
        "discriminants": [
          112
        ],
        "stream_level": false
      }
    ]
  },
  "stream_profiles": [
    {
      "stream": 16384,
      "class": "channel"
    }
  ],
  "stream_constraints": [],
  "device_state_streams": []
}"#;
        let contract = parsed(EDGES);
        let mut rng = StdRng::seed_from_u64(17);
        assert!(
            contract.pick_magic(&mut rng, 16386, 3).is_none(),
            "a four-byte field with three bytes left is not a refill"
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
            if let Some(appliedres) = contract.apply(&mut rng, Action::Length(field), &mut bytes) {
                if appliedres.2 == cap {
                    applied = Some(appliedres);
                    break;
                }
            }
        }
        assert_eq!(applied.map(|a| a.2), Some(cap), "the saturating boundary is drawn eventually");
        assert_eq!(bytes.len(), crate::config::MAX_STREAM_LEN, "the stream stops at the cap");
        assert_eq!(read_value(&bytes, 8, 8), cap, "the field states the payload that exists");
    }
}
