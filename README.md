# Haar-Wavelet-Transformation

dn2 `<input file>` `<option>` `<output file>` `<thr>`

**Options:**
- `c` - Compression
- `d` - Decompression

**File Paths:**
- `<input file>` - Path to any file containing an image
- `<output file>` - Path to the output binary file (after compression) or image file (after decompression)

**Compression Threshold:**
- `<thr>` - Threshold for compression

## Example Usage:

### Compression and Decompression:
```bash
dn2 input_image.jpg c compressed_output.bin 0.5
dn2 compressed_input.bin d decompressed_output.jpg
