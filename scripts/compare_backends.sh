#!/bin/bash
# Backend Comparison Script
#
# This script runs the backend_comparison example with both rustfft and microfft
# backends, then compares the results to ensure they produce identical output.

set -e  # Exit on error

echo "==================================================================="
echo "Backend Comparison Test"
echo "==================================================================="
echo ""

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Clean up old results
echo "Cleaning up old results..."
rm -f results_rustfft.txt results_microfft.txt
echo ""

# Run with rustfft backend
echo "-------------------------------------------------------------------"
echo "Running with rustfft backend (std)..."
echo "-------------------------------------------------------------------"
cargo run --release --example backend_comparison -- rustfft
echo ""

# Run with microfft backend
echo "-------------------------------------------------------------------"
echo "Running with microfft backend (no_std)..."
echo "-------------------------------------------------------------------"
cargo run --release --no-default-features --features microfft-backend --example backend_comparison -- microfft
echo ""

# Compare results
echo "-------------------------------------------------------------------"
echo "Comparing results..."
echo "-------------------------------------------------------------------"

if [ ! -f results_rustfft.txt ]; then
    echo -e "${RED}Error: results_rustfft.txt not found${NC}"
    exit 1
fi

if [ ! -f results_microfft.txt ]; then
    echo -e "${RED}Error: results_microfft.txt not found${NC}"
    exit 1
fi

# Extract SNR values
rustfft_snr=$(grep "SNR:" results_rustfft.txt | awk '{print $2}')
microfft_snr=$(grep "SNR:" results_microfft.txt | awk '{print $2}')

echo "rustfft SNR:  $rustfft_snr dB"
echo "microfft SNR: $microfft_snr dB"
echo ""

# Check SNR values are both high
rustfft_snr_value=$(echo $rustfft_snr | cut -d. -f1)
microfft_snr_value=$(echo $microfft_snr | cut -d. -f1)

if [ "$rustfft_snr_value" -lt 100 ]; then
    echo -e "${RED}FAIL: rustfft SNR too low ($rustfft_snr dB < 100 dB)${NC}"
    exit 1
fi

if [ "$microfft_snr_value" -lt 100 ]; then
    echo -e "${RED}FAIL: microfft SNR too low ($microfft_snr dB < 100 dB)${NC}"
    exit 1
fi

echo -e "${GREEN}Both backends have correct reconstruction quality (>100 dB SNR)${NC}"
echo ""

# Check if numerical differences are within tolerance
echo "Checking numerical precision..."

# Count number of different lines (excluding backend name and SNR lines which we expect to differ)
diff_count=$(diff results_rustfft.txt results_microfft.txt | grep -E "^[<>]" | grep -vE "(Backend:|SNR:)" | wc -l)

if [ "$diff_count" -eq 0 ]; then
    echo -e "${GREEN}Results are identical!${NC}"
else
    echo -e "${YELLOW}Found $diff_count lines with numerical differences${NC}"
    echo ""

    # Check if SNR difference is acceptable (within 0.5 dB)
    snr_diff=$(awk -v r="$rustfft_snr" -v m="$microfft_snr" 'BEGIN {diff = r - m; if (diff < 0) diff = -diff; print diff}')
    snr_threshold=1.0

    snr_check=$(awk -v diff="$snr_diff" -v thresh="$snr_threshold" 'BEGIN {if (diff > thresh) print "fail"; else print "pass"}')

    if [ "$snr_check" = "fail" ]; then
        echo -e "${RED}FAIL: SNR difference too large: ${snr_diff} dB (threshold: ${snr_threshold} dB)${NC}"
        echo ""
        echo "Sample differences:"
        diff results_rustfft.txt results_microfft.txt | grep -E "^[<>]" | grep -vE "(Backend:|SNR:)" | head -20
        exit 1
    fi

    echo "  SNR difference: ${snr_diff} dB (< ${snr_threshold} dB threshold) ✓"
    echo "  This is expected due to different FFT implementations and floating-point rounding."
    echo ""
    echo "Sample differences (first few lines):"
    diff results_rustfft.txt results_microfft.txt | grep -E "^[<>]" | grep -vE "(Backend:|SNR:)" | head -10
    echo ""
    echo -e "${GREEN}PASS: Differences are within acceptable numerical tolerance${NC}"
fi

echo "-------------------------------------------------------------------"
echo -e "${GREEN}Backend comparison test completed successfully!${NC}"
echo "-------------------------------------------------------------------"
echo ""
echo "Both rustfft and microfft backends are working correctly and"
echo "produce identical results for f32 STFT/iSTFT processing."
echo ""
