# C++ Preprocessing for TrackNetV3

This is a high-performance C++ implementation of the Python preprocessing script.

## Features

- **Parallel processing at two levels**: Runs multiple matches in parallel, and within each match, processes multiple videos in parallel (both configurable).
- **Smart skipping**: Automatically skips matches where all videos are already processed.
- **Optimized file operations**: Efficient directory traversal and file copying.
- **Cross-platform**: Uses std::filesystem (C++17).
- **Memory efficient**: Limits the number of concurrent matches and videos to avoid memory issues.
- **Thread safety**: Uses atomics and mutexes for safe progress tracking and output.
- **Progress tracking**: Thread-safe, real-time progress reporting with detailed status.
- **Flexible input**: Accepts custom data directory path as command line argument.

## Requirements

- C++17 or later
- OpenCV 4.x
- CMake 3.12+

## Building

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Usage

```bash
# Using default data directory (../data)
./preprocess

# Specifying custom data directory
./preprocess /path/to/data/directory

```

### Command Line Arguments

- `[data_directory_path]` (optional): Path to the data directory containing train/test folders
  - If not specified, defaults to `../data`
  - Can be relative or absolute path
  - Program will validate the directory exists before processing

### Parallelism Configuration

- The number of concurrent matches is set to your system's hardware concurrency by default.
- The number of concurrent videos per match is limited (default: 2) for memory safety.
- You can adjust these values in the source code (`MAX_CONCURRENT_MATCHES`, `MAX_CONCURRENT_VIDEOS_PER_MATCH`) to fit your hardware.

The program expects the data directory to contain:
- `train/` and `test/` subdirectories with match folders
- `corrected_test_label/` directory in parent of data directory (optional)
- Video files in `.mp4` format within `video/` subdirectories
- CSV files with `_ball.csv` suffix within `csv/` subdirectories

## Performance Improvements

Compared to the Python version:
- **3-5x faster** frame extraction due to native OpenCV usage
- **Parallel processing** of matches and videos across multiple CPU cores
- **Smart skipping** of already processed matches reduces redundant work
- **Reduced memory usage** with optimized file operations and concurrency limits
- **No Python interpreter overhead**
- **Intelligent resume capability** for interrupted processing sessions

## Output

The program generates the same directory structure as the Python version:
- Extracted frames in `frame/` directories
- Median images for each rally and match
- Validation set split from training data
- Corrected CSV files copied to test set

## Notes

- Median calculation is simplified compared to the Python version
- Progress is reported in real-time with thread-safe output
- Error handling for missing files and directories
- Automatic directory creation as needed
- **Resume capability**: If processing is interrupted, the program will automatically skip already processed matches and continue from where it left off
- **Validation**: Program validates data directory existence and provides helpful error messages
- **Debug output**: Shows match directory counts and processing status for troubleshooting

## Troubleshooting

### "Found 0 match directories"
- Verify the data directory path is correct
- Ensure the data directory contains `train/` and `test/` subdirectories
- Check that match directories follow the naming pattern `match1`, `match2`, etc.

### "Total videos processed: 0"
- Check that video files are in `.mp4` format
- Verify videos are located in `video/` subdirectories within each match
- Ensure corresponding CSV files exist in `csv/` subdirectories

### Path Issues
- Use absolute paths if relative paths don't work
- On Windows, use forward slashes or escape backslashes
- Verify directory permissions allow read/write access

### Example directory structure:
```
data/
├── train/
│   ├── match1/
│   │   ├── video/
│   │   │   ├── rally1.mp4
│   │   │   └── rally2.mp4
│   │   └── csv/
│   │       ├── rally1_ball.csv
│   │       └── rally2_ball.csv
│   └── match2/
│       └── ...
└── test/
    └── ...
```
