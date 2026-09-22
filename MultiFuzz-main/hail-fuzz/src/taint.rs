//! Shadow taint state for the dynamic P-code taint pass.
//!
//! Tags carry *provenance*, not bytes: a tag is a set of MMIO read sites
//! ([`AccessContext`]) whose values contributed bits to the current value.  This
//! is what removes the ambiguity of value matching: a tainted register points
//! at the exact read that produced it, and the read points at the exact input
//! offset range it consumed.

use hashbrown::HashMap;
use pcode::VarId;
use serde::Serialize;

use crate::input::StreamKey;

/// Identity of one MMIO read *occurrence*: the accessing instruction's PC, the
/// stream it read from, and which repeat of that read this was.
///
/// This is the provenance identity of the whole analysis: a [`TaintTag`] is a
/// set of these, and each one resolves back to the exact input bytes the read
/// consumed.  Nothing else about "how streams relate to each other" is needed to
/// infer a field's role.
///
/// `occ` matters because firmware commonly reads a whole packet one byte at a
/// time from a single data register: without it every byte of the packet would
/// share one identity, and a role discovered after the packet was collected
/// would be attributed to the last byte only.  With it, "the value compared
/// after collecting the packet" still traces back to the byte read first.
///
/// A loop therefore creates one [`TaintIndex`] entry per iteration.  That is
/// bounded by the loop trip count and is the price of exact attribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct AccessContext {
    pub pc: u64,
    pub addr: StreamKey,
    /// 1-based ordinal of this read among all reads of the same `(pc, addr)`.
    pub occ: u32,
}

impl AccessContext {
    /// Identity of the first read at a site.
    pub const fn new(pc: u64, addr: StreamKey) -> Self {
        Self { pc, addr, occ: 1 }
    }

    /// Identity of the `occ`-th read at a site.
    pub const fn at(pc: u64, addr: StreamKey, occ: u32) -> Self {
        Self { pc, addr, occ }
    }

    /// Identity of the *site* itself, with no occurrence attached (`occ == 0`).
    ///
    /// Loop gating is a property of the read site -- this site's value decides
    /// how much of *that* site is consumed -- and not of one occurrence: two
    /// nested back edges can reap the same source occurrence, and they describe
    /// the same loop.  The bound is therefore keyed by the sites, and the
    /// occurrences that actually opened it are carried alongside it
    /// (`LoopBound::source_occs` in `semantic_taint`).
    ///
    /// Occurrences are 1-based, so `occ == 0` can never collide with a real read.
    pub const fn site(pc: u64, addr: StreamKey) -> Self {
        Self { pc, addr, occ: 0 }
    }
}

/// Upper bound on tracked byte-level memory tags.  Taint is only an analysis
/// artefact, so evicting the oldest entries degrades recall rather than
/// correctness, and keeps a long replay from growing without bound.
const MAX_MEM_TAINT_BYTES: usize = 1 << 20;

/// Interns read sites into small integer ids so a tag stays compact even when it
/// is cloned once per tainted byte.
#[derive(Debug, Default, Clone)]
pub struct TaintIndex {
    contexts: Vec<AccessContext>,
    ids: HashMap<AccessContext, u32>,
}

impl TaintIndex {
    pub fn intern(&mut self, context: AccessContext) -> u32 {
        if let Some(id) = self.ids.get(&context) {
            return *id;
        }
        let id = self.contexts.len() as u32;
        self.contexts.push(context);
        self.ids.insert(context, id);
        id
    }

    pub fn resolve(&self, id: u32) -> Option<AccessContext> {
        self.contexts.get(id as usize).copied()
    }

    /// Resolve every read site a tag depends on.
    pub fn contexts_in_tag(&self, tag: &TaintTag) -> Vec<AccessContext> {
        tag.ids.iter().filter_map(|id| self.resolve(*id)).collect()
    }
}

/// A set of interned read sites that a value's bits derive from.
///
/// `ids` is kept sorted so union is a linear merge and equality is structural.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TaintTag {
    ids: Vec<u32>,
}

impl TaintTag {
    // Deliberately not SCREAMING_CASE: `TaintTag::Clean` is read far more often
    // than it is written, and the name is part of how the pass explains itself.
    #[allow(non_upper_case_globals)]
    pub const Clean: TaintTag = TaintTag { ids: Vec::new() };

    pub fn single(id: u32) -> Self {
        Self { ids: vec![id] }
    }

    pub fn is_clean(&self) -> bool {
        self.ids.is_empty()
    }

    /// Union of two provenance sets.
    pub fn union(&self, other: &TaintTag) -> TaintTag {
        if self.is_clean() {
            return other.clone();
        }
        if other.is_clean() {
            return self.clone();
        }

        let mut ids = Vec::with_capacity(self.ids.len() + other.ids.len());
        let (mut left, mut right) = (0, 0);
        while left < self.ids.len() && right < other.ids.len() {
            match self.ids[left].cmp(&other.ids[right]) {
                std::cmp::Ordering::Less => {
                    ids.push(self.ids[left]);
                    left += 1;
                }
                std::cmp::Ordering::Greater => {
                    ids.push(other.ids[right]);
                    right += 1;
                }
                std::cmp::Ordering::Equal => {
                    ids.push(self.ids[left]);
                    left += 1;
                    right += 1;
                }
            }
        }
        ids.extend_from_slice(&self.ids[left..]);
        ids.extend_from_slice(&other.ids[right..]);
        TaintTag { ids }
    }
}

/// Register, memory and provenance state for one taint pass.
#[derive(Debug, Clone)]
pub struct ShadowState {
    pub index: TaintIndex,
    regs: HashMap<VarId, TaintTag>,
    mem: HashMap<u32, TaintTag>,
    /// Argument registers clobbered by an opaque call.
    call_clobber: Vec<VarId>,
    evictions: u64,
}

impl Default for ShadowState {
    fn default() -> Self {
        Self::new()
    }
}

impl ShadowState {
    pub fn new() -> Self {
        Self {
            index: TaintIndex::default(),
            regs: HashMap::new(),
            mem: HashMap::new(),
            // Matches the ARM AAPCS argument registers; the driver replaces this
            // with the real VarIds resolved from the loaded SLEIGH spec.
            call_clobber: vec![0, 1, 2, 3],
            evictions: 0,
        }
    }

    /// Install the true argument-register ids resolved from the CPU's spec.
    pub fn set_call_clobber(&mut self, ids: impl IntoIterator<Item = VarId>) {
        self.call_clobber = ids.into_iter().collect();
    }

    pub fn reg_tag(&self, id: VarId) -> TaintTag {
        self.regs.get(&id).cloned().unwrap_or(TaintTag::Clean)
    }

    pub fn set_reg_tag(&mut self, id: VarId, tag: TaintTag) {
        if tag.is_clean() {
            self.regs.remove(&id);
        }
        else {
            self.regs.insert(id, tag);
        }
    }

    /// Provenance of a freshly observed MMIO read.
    pub fn source_tag(&mut self, context: AccessContext) -> TaintTag {
        TaintTag::single(self.index.intern(context))
    }

    pub fn mem_tag_range(&self, addr: u32, size: usize) -> TaintTag {
        let mut tag = TaintTag::Clean;
        for offset in 0..size.max(1) as u32 {
            if let Some(byte) = self.mem.get(&addr.wrapping_add(offset)) {
                tag = tag.union(byte);
            }
        }
        tag
    }

    /// Strong update: the written span takes exactly `tag`, so a store of clean
    /// data also *clears* whatever taint lived at those addresses.
    pub fn set_mem_tag_range(&mut self, addr: u32, size: usize, tag: TaintTag) {
        for offset in 0..size.max(1) as u32 {
            let address = addr.wrapping_add(offset);
            if tag.is_clean() {
                self.mem.remove(&address);
            }
            else {
                self.mem.insert(address, tag.clone());
            }
        }
        self.enforce_memory_bound();
    }

    /// An opaque call may clobber the argument registers, so drop their taint.
    pub fn kill_call_regs(&mut self) {
        for id in &self.call_clobber {
            self.regs.remove(id);
        }
    }

    #[cfg(test)]
    pub fn tracked_memory_bytes(&self) -> usize {
        self.mem.len()
    }

    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    fn enforce_memory_bound(&mut self) {
        if self.mem.len() <= MAX_MEM_TAINT_BYTES {
            return;
        }
        let excess = self.mem.len() - MAX_MEM_TAINT_BYTES;
        let victims: Vec<u32> = self.mem.keys().copied().take(excess).collect();
        for victim in victims {
            self.mem.remove(&victim);
            self.evictions += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(pc: u64, addr: u64) -> AccessContext {
        AccessContext::new(pc, addr)
    }

    #[test]
    fn tags_track_the_exact_read_site() {
        let mut shadow = ShadowState::new();
        let a = shadow.source_tag(ctx(0x100, 0x5800_0000));
        let b = shadow.source_tag(ctx(0x200, 0x5800_0004));
        assert!(!a.is_clean() && !b.is_clean());
        assert_ne!(a, b);

        let both = a.union(&b);
        assert_eq!(
            shadow.index.contexts_in_tag(&both),
            vec![ctx(0x100, 0x5800_0000), ctx(0x200, 0x5800_0004)]
        );
    }

    #[test]
    fn union_is_idempotent_and_ordered() {
        let mut shadow = ShadowState::new();
        let a = shadow.source_tag(ctx(0x100, 0x5800_0000));
        let b = shadow.source_tag(ctx(0x200, 0x5800_0004));
        assert_eq!(a.union(&a), a);
        assert_eq!(a.union(&b), b.union(&a));
        assert_eq!(TaintTag::Clean.union(&a), a);
    }

    #[test]
    fn store_strong_update_clears_previous_taint() {
        let mut shadow = ShadowState::new();
        let tag = shadow.source_tag(ctx(0x100, 0x5800_0000));
        shadow.set_mem_tag_range(0x2000_0000, 4, tag.clone());
        assert_eq!(shadow.mem_tag_range(0x2000_0000, 4), tag);

        shadow.set_mem_tag_range(0x2000_0000, 4, TaintTag::Clean);
        assert!(shadow.mem_tag_range(0x2000_0000, 4).is_clean());
        assert_eq!(shadow.tracked_memory_bytes(), 0);
    }

    #[test]
    fn opaque_call_kills_argument_registers() {
        let mut shadow = ShadowState::new();
        shadow.set_call_clobber([0, 1, 2, 3]);
        let tag = shadow.source_tag(ctx(0x100, 0x5800_0000));
        for id in 0..4 {
            shadow.set_reg_tag(id, tag.clone());
        }
        shadow.set_reg_tag(7, tag.clone());

        shadow.kill_call_regs();

        for id in 0..4 {
            assert!(shadow.reg_tag(id).is_clean(), "r{id} must be killed");
        }
        assert_eq!(shadow.reg_tag(7), tag);
    }

    #[test]
    fn partial_range_reads_union_only_the_requested_bytes() {
        let mut shadow = ShadowState::new();
        let low = shadow.source_tag(ctx(0x100, 0x5800_0000));
        let high = shadow.source_tag(ctx(0x200, 0x5800_0004));
        shadow.set_mem_tag_range(0x2000_1000, 1, low.clone());
        shadow.set_mem_tag_range(0x2000_1001, 1, high.clone());

        assert_eq!(shadow.mem_tag_range(0x2000_1000, 1), low);
        assert_eq!(shadow.mem_tag_range(0x2000_1001, 1), high);
        assert_eq!(shadow.mem_tag_range(0x2000_1000, 2), low.union(&high));
    }
}
