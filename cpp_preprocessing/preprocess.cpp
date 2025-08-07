#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <future>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <regex>
#include <atomic>
#include <mutex>
#include <chrono>

namespace fs = std::filesystem;

// Global variables
const std::string IMG_FORMAT = "png";

// Thread-safe counter for progress tracking
std::atomic<int> processed_videos{0};
std::atomic<int> processed_matches{0};
std::mutex cout_mutex;

// Configuration for parallel processing
const size_t MAX_CONCURRENT_MATCHES = std::thread::hardware_concurrency();
const size_t MAX_CONCURRENT_VIDEOS_PER_MATCH = 2; // Limit to prevent memory issues

// Utility: List directories in a given path
std::vector<fs::path> list_dirs(const fs::path &dir)
{
    std::vector<fs::path> dirs;
    if (!fs::exists(dir))
        return dirs;

    for (const auto &entry : fs::directory_iterator(dir))
    {
        if (entry.is_directory())
        {
            dirs.push_back(entry.path());
        }
    }

    // Sort directories
    std::sort(dirs.begin(), dirs.end(), [](const fs::path &a, const fs::path &b)
              { return a.filename().string() < b.filename().string(); });

    return dirs;
}

// Utility: List files in a given path
std::vector<fs::path> list_files(const fs::path &dir, const std::string &extension = "")
{
    std::vector<fs::path> files;
    if (!fs::exists(dir))
        return files;

    for (const auto &entry : fs::directory_iterator(dir))
    {
        if (entry.is_regular_file())
        {
            if (extension.empty() || entry.path().extension() == extension)
            {
                files.push_back(entry.path());
            }
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

// Utility: Copy directory recursively
void copy_dir_recursive(const fs::path &src, const fs::path &dst)
{
    try
    {
        fs::create_directories(dst);
        for (const auto &entry : fs::recursive_directory_iterator(src))
        {
            const auto &rel = fs::relative(entry.path(), src);
            const auto &dest = dst / rel;

            if (entry.is_directory())
            {
                fs::create_directories(dest);
            }
            else
            {
                fs::copy_file(entry.path(), dest, fs::copy_options::overwrite_existing);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error copying directory " << src << " to " << dst << ": " << e.what() << std::endl;
    }
}

// Utility: Copy single file
void copy_file_safe(const fs::path &src, const fs::path &dst)
{
    try
    {
        fs::create_directories(dst.parent_path());
        fs::copy_file(src, dst, fs::copy_options::overwrite_existing);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error copying file " << src << " to " << dst << ": " << e.what() << std::endl;
    }
}

// Utility: Move file or directory
void move_file_safe(const fs::path &src, const fs::path &dst)
{
    try
    {
        fs::create_directories(dst.parent_path());
        fs::rename(src, dst);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error moving " << src << " to " << dst << ": " << e.what() << std::endl;
    }
}

// Parse video name from file path
std::string parse_video_name(const fs::path &video_file)
{
    return video_file.stem().string();
}

// Parse match ID from directory path
std::string parse_match_id(const fs::path &match_dir)
{
    std::string dir_name = match_dir.filename().string();
    std::regex match_regex("match([0-9]+)");
    std::smatch match;
    if (std::regex_search(dir_name, match, match_regex))
    {
        return match[1].str();
    }
    return "unknown";
}

// Count frames in a directory
int get_num_frames(const fs::path &frame_dir)
{
    if (!fs::exists(frame_dir))
        return 0;

    int count = 0;
    for (const auto &entry : fs::directory_iterator(frame_dir))
    {
        if (entry.is_regular_file() && entry.path().extension() == ("." + IMG_FORMAT))
        {
            count++;
        }
    }
    return count;
}

// Check if all videos in a match are already processed
bool are_all_videos_processed(const fs::path &match_dir)
{
    fs::path video_dir = match_dir / "video";
    fs::path frame_dir = match_dir / "frame";

    if (!fs::exists(video_dir))
    {
        return false;
    }

    auto video_files = list_files(video_dir, ".mp4");
    if (video_files.empty())
    {
        return true; // No videos to process
    }

    // If frame directory doesn't exist, definitely not processed
    if (!fs::exists(frame_dir))
    {
        return false;
    }

    int videos_with_csv = 0;
    int processed_videos = 0;

    // Check if each video has corresponding frames
    for (const auto &video_file : video_files)
    {
        std::string rally_id = parse_video_name(video_file);
        fs::path rally_dir = frame_dir / rally_id;

        // Check if CSV file exists (required for processing)
        fs::path csv_file = match_dir / "csv" / (rally_id + "_ball.csv");
        if (!fs::exists(csv_file))
        {
            continue; // Skip videos without CSV files
        }

        videos_with_csv++;

        // Check if frames exist for this video
        if (fs::exists(rally_dir) && get_num_frames(rally_dir) > 0)
        {
            processed_videos++;
        }
    }

    // Only return true if we have videos with CSV files and all of them are processed
    return videos_with_csv > 0 && processed_videos == videos_with_csv;
}

// Extract frames from video and save them (memory-efficient version)
void generate_data_frames(const fs::path &video_file)
{
    if (!fs::exists(video_file))
    {
        std::cerr << "Video file does not exist: " << video_file << std::endl;
        return;
    }

    // Parse paths
    fs::path match_dir = video_file.parent_path().parent_path();
    std::string rally_id = parse_video_name(video_file);
    fs::path rally_dir = match_dir / "frame" / rally_id;

    // Check if CSV file exists
    fs::path csv_file = match_dir / "csv" / (rally_id + "_ball.csv");
    if (!fs::exists(csv_file))
    {
        std::cerr << "CSV file does not exist: " << csv_file << std::endl;
        return;
    }

    // Check if already processed
    if (fs::exists(rally_dir))
    {
        int existing_frames = get_num_frames(rally_dir);
        if (existing_frames > 0)
        {
            // Already processed, skip
            return;
        }
    }

    // Create output directory
    fs::create_directories(rally_dir);

    // Open video
    cv::VideoCapture cap(video_file.string());
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open video: " << video_file << std::endl;
        return;
    }

    // Extract frames without storing all in memory
    cv::Mat frame;
    int frame_count = 0;
    std::vector<cv::Mat> sample_frames; // Only sample some frames for median
    int sample_interval = 10;           // Sample every 10th frame for median calculation

    while (cap.read(frame))
    {
        if (!frame.empty())
        {
            // Save frame
            std::string frame_filename = std::to_string(frame_count) + "." + IMG_FORMAT;
            fs::path frame_path = rally_dir / frame_filename;
            cv::imwrite(frame_path.string(), frame);

            // Sample frames for median calculation (memory efficient)
            if (frame_count % sample_interval == 0 && sample_frames.size() < 50)
            {
                sample_frames.push_back(frame.clone());
            }

            frame_count++;
        }
    }

    cap.release();

    // Calculate and save median frame from samples
    if (!sample_frames.empty())
    {
        // Use a more memory-efficient median calculation
        cv::Mat median_frame;
        if (sample_frames.size() == 1)
        {
            median_frame = sample_frames[0].clone();
        }
        else
        {
            // Simple median: use middle frame from samples
            std::sort(sample_frames.begin(), sample_frames.end(),
                      [](const cv::Mat &a, const cv::Mat &b)
                      {
                          return cv::sum(a)[0] < cv::sum(b)[0]; // Sort by brightness
                      });
            median_frame = sample_frames[sample_frames.size() / 2].clone();
        }

        // Save median as image
        fs::path median_path = rally_dir / "median.png";
        cv::imwrite(median_path.string(), median_frame);

        // Clear sample frames to free memory
        sample_frames.clear();
    }
}

// Calculate match median from all rally medians
void get_match_median(const fs::path &match_dir)
{
    fs::path frame_dir = match_dir / "frame";
    if (!fs::exists(frame_dir))
        return;

    auto rally_dirs = list_dirs(frame_dir);
    std::vector<cv::Mat> medians;

    for (const auto &rally_dir : rally_dirs)
    {
        fs::path median_file = rally_dir / "median.png";
        if (fs::exists(median_file))
        {
            cv::Mat median = cv::imread(median_file.string());
            if (!median.empty())
            {
                medians.push_back(median);
            }
        }
    }

    if (!medians.empty())
    {
        // Calculate median of medians (simplified)
        cv::Mat match_median = medians[medians.size() / 2].clone();

        // Save match median
        fs::path match_median_path = match_dir / "median.png";
        cv::imwrite(match_median_path.string(), match_median);
    }
}

// Process a single video (for parallel execution)
std::tuple<std::string, std::string, std::string, int> process_video(
    const fs::path &video_file,
    const fs::path &match_dir,
    const std::string &match_id,
    const std::string &split)
{

    generate_data_frames(video_file);

    std::string video_name = parse_video_name(video_file);
    fs::path rally_dir = match_dir / "frame" / video_name;
    int video_frame_count = get_num_frames(rally_dir);

    // Thread-safe progress update
    int current_count = ++processed_videos;

    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "[" << split << " / match" << match_id << " / " << video_name
                  << "]\tvideo frames: " << video_frame_count
                  << " (processed: " << current_count << ")" << std::endl;
    }

    return std::make_tuple(split, match_id, video_name, video_frame_count);
}

// Process a single match with controlled parallelism
std::tuple<std::string, std::string, int> process_match(
    const fs::path &match_dir,
    const std::string &split)
{
    std::string match_id = parse_match_id(match_dir);
    int match_frame_count = 0;

    auto video_files = list_files(match_dir / "video", ".mp4");

    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "[" << split << " / match" << match_id << "] Found " << video_files.size() << " video files" << std::endl;
    }

    // Check if all videos in this match are already processed
    if (are_all_videos_processed(match_dir))
    {
        // Count existing frames for reporting
        fs::path frame_dir = match_dir / "frame";
        if (fs::exists(frame_dir))
        {
            auto rally_dirs = list_dirs(frame_dir);
            for (const auto &rally_dir : rally_dirs)
            {
                match_frame_count += get_num_frames(rally_dir);
            }
        }

        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "[" << split << " / match" << match_id << "]:\talready processed, skipping (total frames: "
                      << match_frame_count << ")" << std::endl;
        }

        return std::make_tuple(split, match_id, match_frame_count);
    }

    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "[" << split << " / match" << match_id << "] Processing " << video_files.size() << " videos..." << std::endl;
    }

    // Process videos with limited parallelism to control memory usage
    std::vector<std::future<std::tuple<std::string, std::string, std::string, int>>> futures;
    size_t active_futures = 0;
    size_t video_index = 0;

    while (video_index < video_files.size() || active_futures > 0)
    {
        // Start new tasks up to the limit
        while (active_futures < MAX_CONCURRENT_VIDEOS_PER_MATCH && video_index < video_files.size())
        {
            futures.push_back(std::async(std::launch::async,
                                         process_video, video_files[video_index], match_dir, match_id, split));
            active_futures++;
            video_index++;
        }

        // Check for completed tasks
        for (auto it = futures.begin(); it != futures.end();)
        {
            if (it->wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
            {
                auto result = it->get();
                match_frame_count += std::get<3>(result);
                it = futures.erase(it);
                active_futures--;
            }
            else
            {
                ++it;
            }
        }

        // Small delay to prevent busy waiting
        if (active_futures >= MAX_CONCURRENT_VIDEOS_PER_MATCH)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    // Calculate match median
    get_match_median(match_dir);

    // Thread-safe progress update
    int current_match_count = ++processed_matches;

    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "[" << split << " / match" << match_id << "]:\ttotal frames: "
                  << match_frame_count << " (matches processed: " << current_match_count << ")" << std::endl;
    }

    return std::make_tuple(split, match_id, match_frame_count);
}

// Process all videos in a split with parallel execution at match level
void process_split(const fs::path &data_dir, const std::string &split)
{
    int split_frame_count = 0;
    auto match_dirs = list_dirs(data_dir / split);

    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "[" << split << "] Found " << match_dirs.size() << " match directories" << std::endl;
    }

    if (match_dirs.empty())
    {
        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "[" << split << "]:\ttotal frames: 0" << std::endl;
        }
        return;
    }

    // Process matches with controlled parallelism
    std::vector<std::future<std::tuple<std::string, std::string, int>>> futures;
    size_t active_futures = 0;
    size_t match_index = 0;

    while (match_index < match_dirs.size() || active_futures > 0)
    {
        // Start new match processing tasks up to the limit
        while (active_futures < MAX_CONCURRENT_MATCHES && match_index < match_dirs.size())
        {
            futures.push_back(std::async(std::launch::async,
                                         process_match, match_dirs[match_index], split));
            active_futures++;
            match_index++;
        }

        // Check for completed match processing
        for (auto it = futures.begin(); it != futures.end();)
        {
            if (it->wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
            {
                auto result = it->get();
                split_frame_count += std::get<2>(result);
                it = futures.erase(it);
                active_futures--;
            }
            else
            {
                ++it;
            }
        }

        // Small delay to prevent busy waiting
        if (active_futures >= MAX_CONCURRENT_MATCHES)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "[" << split << "]:\ttotal frames: " << split_frame_count << std::endl;
    }
}

int main(int argc, char *argv[])
{
    std::string data_dir_str;

    // Parse command line arguments
    if (argc > 1)
    {
        data_dir_str = argv[1];
    }
    else
    {
        data_dir_str = "../data"; // Default fallback
    }

    fs::path data_dir(data_dir_str);

    std::cout << "Starting C++ preprocessing..." << std::endl;
    std::cout << "Using data directory: " << data_dir.string() << std::endl;

    // Check if data directory exists
    if (!fs::exists(data_dir))
    {
        std::cerr << "Error: Data directory does not exist: " << data_dir.string() << std::endl;
        std::cerr << "Usage: " << argv[0] << " [data_directory_path]" << std::endl;
        return 1;
    }

    // 1. Replace csv to corrected csv in test set
    fs::path corrected_test_label_dir = data_dir.parent_path() / "corrected_test_label";
    if (fs::exists(corrected_test_label_dir))
    {
        std::cout << "Copying corrected test labels..." << std::endl;

        auto match_dirs = list_dirs(data_dir / "test");

        for (const auto &match_dir : match_dirs)
        {
            std::string match_name = match_dir.filename().string();
            fs::path corrected_csv = corrected_test_label_dir / match_name / "corrected_csv";
            fs::path dest_csv = data_dir / "test" / match_name / "corrected_csv";

            if (fs::exists(corrected_csv) && !fs::exists(dest_csv))
            {
                copy_dir_recursive(corrected_csv, dest_csv);
                copy_file_safe(corrected_test_label_dir / "drop_frame.json", data_dir / "drop_frame.json");
            }
        }
    }

    // 2. Generate frames from videos (parallelized at match and video level)
    std::cout << "Generating frames from videos..." << std::endl;
    std::cout << "Using " << MAX_CONCURRENT_MATCHES << " concurrent matches and "
              << MAX_CONCURRENT_VIDEOS_PER_MATCH << " concurrent videos per match" << std::endl;

    processed_videos = 0;
    processed_matches = 0;

    process_split(data_dir, "train");
    process_split(data_dir, "test");

    // 3. Form validation set
    if (!fs::exists(data_dir / "val"))
    {
        std::cout << "Creating validation set..." << std::endl;

        auto match_dirs = list_dirs(data_dir / "train");

        for (const auto &match_dir : match_dirs)
        {
            std::string match_name = match_dir.filename().string();

            // Pick last rally in each match as validation set
            auto video_files = list_files(match_dir / "video", ".mp4");
            if (!video_files.empty())
            {
                fs::path last_video = video_files.back();
                std::string rally_id = parse_video_name(last_video);

                // Create validation directories
                fs::path val_match_dir = data_dir / "val" / match_name;
                fs::create_directories(val_match_dir / "csv");
                fs::create_directories(val_match_dir / "video");
                fs::create_directories(val_match_dir / "frame");

                // Move files from train to val
                try
                {
                    move_file_safe(
                        match_dir / "csv" / (rally_id + "_ball.csv"),
                        val_match_dir / "csv" / (rally_id + "_ball.csv"));

                    move_file_safe(
                        match_dir / "video" / (rally_id + ".mp4"),
                        val_match_dir / "video" / (rally_id + ".mp4"));

                    move_file_safe(
                        match_dir / "frame" / rally_id,
                        val_match_dir / "frame" / rally_id);

                    // Copy median file
                    if (fs::exists(match_dir / "median.png"))
                    {
                        copy_file_safe(
                            match_dir / "median.png",
                            val_match_dir / "median.png");
                    }
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Error creating validation set for " << match_name
                              << ": " << e.what() << std::endl;
                }
            }
        }
    }

    std::cout << "Done." << std::endl;
    std::cout << "Total videos processed: " << processed_videos.load() << std::endl;
    std::cout << "Total matches processed: " << processed_matches.load() << std::endl;

    return 0;
}
