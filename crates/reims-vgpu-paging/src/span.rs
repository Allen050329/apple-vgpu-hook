//! Byte-span page arithmetic.

/// How many guest pages `[gva, gva+span)` touches, given `page_size`.
///
/// The `gva % page_size` term is the whole content: a span that starts
/// mid-page reaches one page further than its length alone implies. Callers
/// compare a walk's result against this to decide whether the *whole* span
/// resolved, and getting it wrong reads as "fully covered" for exactly the
/// windows that straddle a page boundary — which is most of them.
pub fn pages_spanned(gva: u64, span: u64, page_size: u64) -> u64 {
    ((gva % page_size) + span).div_ceil(page_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A span's page count is decided by where it *starts*, not only by how
    /// long it is.
    ///
    /// The device's rails compare a walk's page count against this to decide
    /// whether the whole span resolved. Drop the offset term and a window that
    /// straddles a page boundary — which is most of them, since a texture row
    /// rarely starts page-aligned — reports fully covered while missing its
    /// last page. The gather then hands the GPU a short buffer, which is a
    /// wrong frame.
    #[test]
    fn pages_spanned_counts_the_page_the_offset_pushes_a_span_into() {
        const PAGE: u64 = 4096;
        // Page-aligned: exactly what the length implies.
        assert_eq!(pages_spanned(0, PAGE, PAGE), 1);
        assert_eq!(pages_spanned(PAGE * 7, PAGE * 3, PAGE), 3);
        // Offset by one byte: the same length now reaches one page further.
        assert_eq!(pages_spanned(1, PAGE, PAGE), 2);
        assert_eq!(pages_spanned(PAGE * 7 + 1, PAGE * 3, PAGE), 4);
        // A span wholly inside one page stays at one, wherever it starts.
        assert_eq!(pages_spanned(PAGE - 1, 1, PAGE), 1);
        // …and one byte longer crosses.
        assert_eq!(pages_spanned(PAGE - 1, 2, PAGE), 2);
        // The arm64 pathway's 16 KiB pages take the same rule.
        assert_eq!(pages_spanned(16384 * 3 + 5, 16384, 16384), 2);
        // A zero span touches nothing.
        assert_eq!(pages_spanned(0, 0, PAGE), 0);
    }
}
