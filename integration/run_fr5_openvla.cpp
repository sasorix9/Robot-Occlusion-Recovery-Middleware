#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <system_error>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include <curl/curl.h>
#include <librealsense2/rs.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <robot.h>
#include <sys/file.h>

namespace {

using Json = nlohmann::json;
using Action = std::array<double, 7>;
using namespace std::chrono_literals;

constexpr char kServerUrl[] = "http://192.168.3.23:18080";
constexpr char kExpectedBackend[] = "openvla";
constexpr char kExpectedUnnormKey[] = "fr5_dataset";
constexpr char kRobotIp[] = "192.168.58.2";
constexpr char kGripperPort[] = "/dev/ttyACM0";
constexpr char kCameraSerial[] = "348122071193";
constexpr int kToolId = 2;
constexpr int kUserId = 0;
constexpr int kImageWidth = 640;
constexpr int kImageHeight = 480;
constexpr int kImageRateHz = 30;
constexpr int kJpegQuality = 95;
constexpr double kPi = 3.14159265358979323846;
constexpr double kServoPeriodSeconds = 0.008;
constexpr int kServoSubsteps = 13;
constexpr std::array<double, 6> kInitialJointDegrees{
    130.0, -110.0, 130.0, -115.0, -90.0, -10.0};
constexpr float kInitialMoveSpeedPercent = 10.0F;
constexpr double kInitialJointToleranceDegrees = 0.15;

std::atomic<bool> g_interrupted{false};

void SignalHandler(int) { g_interrupted.store(true, std::memory_order_relaxed); }

struct Options {
  std::string instruction;
  int max_steps = 0;
  double max_translation_m = 0.04;
  double max_rotation_rad = 0.10;
  double translation_scale = 1.0;
  bool enable_motion = false;
  bool self_test = false;
};

[[noreturn]] void UsageError(const std::string& message) {
  throw std::runtime_error(
      message +
      "\nusage: run_fr5_openvla --instruction <text> --max-steps <count> "
      "[--enable-motion] [--translation-scale <factor>] "
      "[--max-translation-m <m>] "
      "[--max-rotation-rad <rad>] | --self-test");
}

double ParsePositiveDouble(const std::string& text, const char* name) {
  std::size_t used = 0;
  double value = 0.0;
  try {
    value = std::stod(text, &used);
  } catch (const std::exception&) {
    UsageError(std::string("invalid ") + name + ": " + text);
  }
  if (used != text.size() || !std::isfinite(value) || value <= 0.0) {
    UsageError(std::string("invalid ") + name + ": " + text);
  }
  return value;
}

int ParsePositiveInt(const std::string& text, const char* name) {
  std::size_t used = 0;
  long value = 0;
  try {
    value = std::stol(text, &used);
  } catch (const std::exception&) {
    UsageError(std::string("invalid ") + name + ": " + text);
  }
  if (used != text.size() || value <= 0 ||
      value > std::numeric_limits<int>::max()) {
    UsageError(std::string("invalid ") + name + ": " + text);
  }
  return static_cast<int>(value);
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next_value = [&](const char* name) -> std::string {
      if (++index >= argc) {
        UsageError(std::string("missing value for ") + name);
      }
      return argv[index];
    };
    if (argument == "--instruction") {
      options.instruction = next_value("--instruction");
    } else if (argument == "--max-steps") {
      options.max_steps = ParsePositiveInt(next_value("--max-steps"),
                                           "--max-steps");
    } else if (argument == "--max-translation-m") {
      options.max_translation_m = ParsePositiveDouble(
          next_value("--max-translation-m"), "--max-translation-m");
    } else if (argument == "--translation-scale") {
      options.translation_scale = ParsePositiveDouble(
          next_value("--translation-scale"), "--translation-scale");
    } else if (argument == "--max-rotation-rad") {
      options.max_rotation_rad = ParsePositiveDouble(
          next_value("--max-rotation-rad"), "--max-rotation-rad");
    } else if (argument == "--enable-motion") {
      options.enable_motion = true;
    } else if (argument == "--self-test") {
      options.self_test = true;
    } else {
      UsageError("unknown argument: " + argument);
    }
  }
  if (options.self_test) {
    if (argc != 2) {
      UsageError("--self-test cannot be combined with other arguments");
    }
    return options;
  }
  if (options.instruction.empty()) {
    UsageError("--instruction is required");
  }
  if (options.max_steps == 0) {
    UsageError("--max-steps is required");
  }
  return options;
}

std::string Base64Encode(const std::vector<unsigned char>& input) {
  static constexpr char alphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string output;
  output.reserve(((input.size() + 2) / 3) * 4);
  for (std::size_t index = 0; index < input.size(); index += 3) {
    const std::uint32_t first = input[index];
    const std::uint32_t second =
        index + 1 < input.size() ? input[index + 1] : 0;
    const std::uint32_t third =
        index + 2 < input.size() ? input[index + 2] : 0;
    const std::uint32_t value = (first << 16U) | (second << 8U) | third;
    output.push_back(alphabet[(value >> 18U) & 0x3FU]);
    output.push_back(alphabet[(value >> 12U) & 0x3FU]);
    output.push_back(index + 1 < input.size()
                         ? alphabet[(value >> 6U) & 0x3FU]
                         : '=');
    output.push_back(index + 2 < input.size() ? alphabet[value & 0x3FU] : '=');
  }
  return output;
}

double VectorNorm(const Action& action, std::size_t begin,
                  std::size_t end) {
  double squared = 0.0;
  for (std::size_t index = begin; index < end; ++index) {
    squared += action[index] * action[index];
  }
  return std::sqrt(squared);
}

double MaximumJointErrorDegrees(const JointPos& actual) {
  double maximum = 0.0;
  for (std::size_t index = 0; index < kInitialJointDegrees.size(); ++index) {
    if (!std::isfinite(actual.jPos[index])) {
      return std::numeric_limits<double>::infinity();
    }
    maximum = std::max(
        maximum, std::abs(actual.jPos[index] - kInitialJointDegrees[index]));
  }
  return maximum;
}

void ValidateAction(const Action& action, double maximum_translation_m,
                    double maximum_rotation_rad) {
  if (!std::all_of(action.begin(), action.end(),
                   [](double value) { return std::isfinite(value); })) {
    throw std::runtime_error("inference returned a non-finite action");
  }
  const double translation = VectorNorm(action, 0, 3);
  const double rotation = VectorNorm(action, 3, 6);
  if (translation > maximum_translation_m) {
    throw std::runtime_error("translation action exceeds safety limit: " +
                             std::to_string(translation) + " m");
  }
  if (rotation > maximum_rotation_rad) {
    throw std::runtime_error("rotation action exceeds safety limit: " +
                             std::to_string(rotation) + " rad");
  }
  if (action[6] < 0.0 || action[6] > 1.0) {
    throw std::runtime_error("gripper action is outside [0, 1]");
  }
}

Action ScaleTranslation(Action action, double scale,
                        double maximum_translation_m,
                        double maximum_rotation_rad) {
  for (std::size_t axis = 0; axis < 3; ++axis) {
    action[axis] *= scale;
  }
  ValidateAction(action, maximum_translation_m, maximum_rotation_rad);
  return action;
}

Action ParseActionResponse(const std::string& body,
                           double maximum_translation_m,
                           double maximum_rotation_rad) {
  const Json response = Json::parse(body);
  if (!response.contains("action") || !response.at("action").is_array() ||
      response.at("action").size() != 7) {
    throw std::runtime_error("inference response must contain 7D action");
  }
  Action action{};
  for (std::size_t index = 0; index < action.size(); ++index) {
    if (!response.at("action").at(index).is_number()) {
      throw std::runtime_error("inference action contains a non-number");
    }
    action[index] = response.at("action").at(index).get<double>();
  }
  ValidateAction(action, maximum_translation_m, maximum_rotation_rad);
  return action;
}

void ValidateHealthResponse(const std::string& body) {
  const Json health = Json::parse(body);
  if (!health.value("ready", false)) {
    throw std::runtime_error("inference server is not ready");
  }
  if (health.value("backend", std::string()) != kExpectedBackend) {
    throw std::runtime_error("inference backend is not openvla");
  }
  const std::string key = health.value("unnorm_key", std::string());
  if (key != kExpectedUnnormKey) {
    throw std::runtime_error("expected unnorm_key " +
                             std::string(kExpectedUnnormKey) + ", got " + key);
  }
}

std::size_t CurlWrite(char* data, std::size_t size, std::size_t count,
                      void* output) {
  const std::size_t bytes = size * count;
  static_cast<std::string*>(output)->append(data, bytes);
  return bytes;
}

int CurlProgress(void*, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
  return g_interrupted.load(std::memory_order_relaxed) ? 1 : 0;
}

class HttpClient {
 public:
  HttpClient() {
    const CURLcode result = curl_global_init(CURL_GLOBAL_DEFAULT);
    if (result != CURLE_OK) {
      throw std::runtime_error("curl_global_init failed");
    }
  }

  HttpClient(const HttpClient&) = delete;
  HttpClient& operator=(const HttpClient&) = delete;
  ~HttpClient() { curl_global_cleanup(); }

  void CheckHealth() const {
    ValidateHealthResponse(Request(std::string(kServerUrl) + "/health", ""));
  }

  Action Predict(const std::vector<unsigned char>& jpeg,
                 const std::string& instruction, double maximum_translation_m,
                 double maximum_rotation_rad) const {
    const std::string request =
        Json{{"image", Base64Encode(jpeg)}, {"instruction", instruction}}
            .dump();
    return ParseActionResponse(
        Request(std::string(kServerUrl) + "/act", request),
        maximum_translation_m, maximum_rotation_rad);
  }

 private:
  static std::string Request(const std::string& url,
                             const std::string& request_body) {
    CURL* handle = curl_easy_init();
    if (handle == nullptr) {
      throw std::runtime_error("curl_easy_init failed");
    }
    std::string response;
    char error_buffer[CURL_ERROR_SIZE]{};
    curl_slist* headers = nullptr;
    if (!request_body.empty()) {
      headers = curl_slist_append(headers, "Content-Type: application/json");
    }
    curl_easy_setopt(handle, CURLOPT_URL, url.c_str());
    curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT_MS, 2000L);
    curl_easy_setopt(handle, CURLOPT_TIMEOUT_MS, 30000L);
    curl_easy_setopt(handle, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, CurlWrite);
    curl_easy_setopt(handle, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(handle, CURLOPT_ERRORBUFFER, error_buffer);
    curl_easy_setopt(handle, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(handle, CURLOPT_XFERINFOFUNCTION, CurlProgress);
    if (!request_body.empty()) {
      curl_easy_setopt(handle, CURLOPT_HTTPHEADER, headers);
      curl_easy_setopt(handle, CURLOPT_POST, 1L);
      curl_easy_setopt(handle, CURLOPT_POSTFIELDS, request_body.data());
      curl_easy_setopt(handle, CURLOPT_POSTFIELDSIZE_LARGE,
                       static_cast<curl_off_t>(request_body.size()));
    }
    const CURLcode result = curl_easy_perform(handle);
    long status = 0;
    curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &status);
    curl_slist_free_all(headers);
    curl_easy_cleanup(handle);
    if (result != CURLE_OK) {
      throw std::runtime_error(
          std::string("HTTP request failed: ") +
          (error_buffer[0] != '\0' ? error_buffer : curl_easy_strerror(result)));
    }
    if (status != 200) {
      if (response.size() > 300) {
        response.resize(300);
      }
      throw std::runtime_error("HTTP " + std::to_string(status) + ": " +
                               response);
    }
    return response;
  }
};

class Camera {
 public:
  Camera() {
    rs2::config configuration;
    configuration.enable_device(kCameraSerial);
    configuration.enable_stream(RS2_STREAM_COLOR, kImageWidth, kImageHeight,
                                RS2_FORMAT_RGB8, kImageRateHz);
    pipeline_.start(configuration);
    for (int count = 0; count < kImageRateHz; ++count) {
      pipeline_.wait_for_frames(5000);
    }
  }

  Camera(const Camera&) = delete;
  Camera& operator=(const Camera&) = delete;
  ~Camera() {
    try {
      pipeline_.stop();
    } catch (...) {
    }
  }

  std::vector<unsigned char> CaptureJpeg() {
    const rs2::frameset frames = pipeline_.wait_for_frames(5000);
    const rs2::video_frame color = frames.get_color_frame();
    if (!color) {
      throw std::runtime_error("RealSense returned no color frame");
    }
    cv::Mat rgb(color.get_height(), color.get_width(), CV_8UC3,
                const_cast<void*>(color.get_data()),
                static_cast<std::size_t>(color.get_stride_in_bytes()));
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    std::vector<unsigned char> jpeg;
    if (!cv::imencode(".jpg", bgr, jpeg,
                      {cv::IMWRITE_JPEG_QUALITY, kJpegQuality})) {
      throw std::runtime_error("JPEG encoding failed");
    }
    return jpeg;
  }

 private:
  rs2::pipeline pipeline_;
};

std::uint16_t ModbusCrc16(const std::uint8_t* data, std::size_t size) {
  std::uint16_t crc = 0xFFFF;
  for (std::size_t index = 0; index < size; ++index) {
    crc ^= data[index];
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc & 1U) != 0U ? static_cast<std::uint16_t>((crc >> 1U) ^ 0xA001U)
                             : static_cast<std::uint16_t>(crc >> 1U);
    }
  }
  return crc;
}

std::vector<std::uint8_t> GripperWriteRequest(std::uint16_t address,
                                               std::uint16_t value) {
  std::vector<std::uint8_t> request{
      1, 0x10, static_cast<std::uint8_t>(address >> 8U),
      static_cast<std::uint8_t>(address & 0xFFU), 0, 1, 2,
      static_cast<std::uint8_t>(value >> 8U),
      static_cast<std::uint8_t>(value & 0xFFU)};
  const std::uint16_t crc = ModbusCrc16(request.data(), request.size());
  request.push_back(static_cast<std::uint8_t>(crc & 0xFFU));
  request.push_back(static_cast<std::uint8_t>(crc >> 8U));
  return request;
}

void WriteAll(int descriptor, const std::uint8_t* data, std::size_t size) {
  std::size_t written = 0;
  while (written < size) {
    const ssize_t result = ::write(descriptor, data + written, size - written);
    if (result > 0) {
      written += static_cast<std::size_t>(result);
    } else if (result < 0 && errno == EINTR) {
      continue;
    } else {
      throw std::system_error(errno, std::generic_category(), "serial write failed");
    }
  }
}

class FileLock {
 public:
  explicit FileLock(int descriptor) : descriptor_(descriptor) {
    while (::flock(descriptor_, LOCK_EX) != 0) {
      if (errno != EINTR) {
        throw std::system_error(errno, std::generic_category(),
                                "failed to lock gripper port");
      }
    }
  }
  FileLock(const FileLock&) = delete;
  FileLock& operator=(const FileLock&) = delete;
  ~FileLock() {
    while (::flock(descriptor_, LOCK_UN) != 0 && errno == EINTR) {
    }
  }

 private:
  int descriptor_;
};

class Gripper {
 public:
  Gripper() {
    descriptor_ = ::open(kGripperPort, O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (descriptor_ < 0) {
      throw std::system_error(errno, std::generic_category(),
                              std::string("failed to open ") + kGripperPort);
    }
    try {
      termios settings{};
      if (tcgetattr(descriptor_, &settings) != 0) {
        throw std::system_error(errno, std::generic_category(),
                                "tcgetattr failed");
      }
      cfmakeraw(&settings);
      cfsetispeed(&settings, B115200);
      cfsetospeed(&settings, B115200);
      settings.c_cflag |= CLOCAL | CREAD;
      settings.c_cflag &= ~(CSTOPB | PARENB | CSIZE);
      settings.c_cflag |= CS8;
      settings.c_cc[VMIN] = 0;
      settings.c_cc[VTIME] = 0;
      if (tcsetattr(descriptor_, TCSANOW, &settings) != 0) {
        throw std::system_error(errno, std::generic_category(),
                                "tcsetattr failed");
      }
      WriteRegister(0x9C41, 20);
      WriteRegister(0x9C42, 20);
    } catch (...) {
      ::close(descriptor_);
      descriptor_ = -1;
      throw;
    }
  }

  Gripper(const Gripper&) = delete;
  Gripper& operator=(const Gripper&) = delete;
  ~Gripper() {
    if (descriptor_ >= 0) {
      ::close(descriptor_);
    }
  }

  void Apply(double open_fraction) {
    const int desired = open_fraction >= 0.5 ? 100 : 0;
    if (desired != last_amplitude_) {
      WriteRegister(0x9C40, static_cast<std::uint16_t>(desired));
      last_amplitude_ = desired;
    }
  }

 private:
  void WriteRegister(std::uint16_t address, std::uint16_t value) {
    FileLock lock(descriptor_);
    if (tcflush(descriptor_, TCIFLUSH) != 0) {
      throw std::system_error(errno, std::generic_category(), "tcflush failed");
    }
    const std::vector<std::uint8_t> request =
        GripperWriteRequest(address, value);
    WriteAll(descriptor_, request.data(), request.size());
    if (tcdrain(descriptor_) != 0) {
      throw std::system_error(errno, std::generic_category(), "tcdrain failed");
    }
    std::vector<std::uint8_t> response;
    std::size_t expected = 0;
    const auto deadline = std::chrono::steady_clock::now() + 150ms;
    while (std::chrono::steady_clock::now() < deadline) {
      const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
          deadline - std::chrono::steady_clock::now());
      pollfd item{descriptor_, POLLIN, 0};
      const int ready = ::poll(
          &item, 1, std::max(1, static_cast<int>(remaining.count())));
      if (ready < 0 && errno == EINTR) {
        continue;
      }
      if (ready <= 0) {
        break;
      }
      std::uint8_t buffer[32];
      const ssize_t count = ::read(descriptor_, buffer, sizeof(buffer));
      if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        continue;
      }
      if (count <= 0) {
        throw std::system_error(errno, std::generic_category(),
                                "serial read failed");
      }
      response.insert(response.end(), buffer, buffer + count);
      if (response.size() >= 2 && expected == 0) {
        expected = (response[1] & 0x80U) != 0U ? 5 : 8;
      }
      if (expected != 0 && response.size() >= expected) {
        break;
      }
    }
    if (response.size() != expected || (expected != 5 && expected != 8)) {
      throw std::runtime_error("gripper write response timeout or bad length");
    }
    const std::uint16_t received_crc =
        static_cast<std::uint16_t>(response[response.size() - 2]) |
        static_cast<std::uint16_t>(response.back() << 8U);
    if (received_crc != ModbusCrc16(response.data(), response.size() - 2)) {
      throw std::runtime_error("gripper write response CRC mismatch");
    }
    if ((response[1] & 0x80U) != 0U) {
      throw std::runtime_error("gripper Modbus exception: " +
                               std::to_string(response[2]));
    }
    if (!std::equal(response.begin(), response.begin() + 6,
                    request.begin())) {
      throw std::runtime_error("gripper write response does not match request");
    }
  }

  int descriptor_ = -1;
  int last_amplitude_ = -1;
};

DescPose ServoIncrement(const Action& action) {
  DescPose increment;
  increment.tran.x = action[0] * 1000.0 / kServoSubsteps;
  increment.tran.y = action[1] * 1000.0 / kServoSubsteps;
  increment.tran.z = action[2] * 1000.0 / kServoSubsteps;
  increment.rpy.rx = action[3] * 180.0 / kPi / kServoSubsteps;
  increment.rpy.ry = action[4] * 180.0 / kPi / kServoSubsteps;
  increment.rpy.rz = action[5] * 180.0 / kPi / kServoSubsteps;
  return increment;
}

class RobotMotion {
 public:
  RobotMotion() {
    try {
      const int result = robot_.RPC(kRobotIp);
      if (result != 0) {
        throw std::runtime_error("FAIRINO RPC failed: " +
                                 std::to_string(result));
      }
      connected_ = true;
      ValidateCoordinateFrames();
      ValidateState(true);
      ROBOT_STATE_PKG state{};
      CheckCall(robot_.GetRobotRealTimeState(&state),
                "GetRobotRealTimeState");
      if (state.robot_mode != 0) {
        CheckCall(robot_.Mode(0), "Mode(0)");
        std::this_thread::sleep_for(200ms);
      }
      CheckCall(robot_.GetRobotRealTimeState(&state),
                "GetRobotRealTimeState");
      if (state.rbtEnableState != 1) {
        CheckCall(robot_.RobotEnable(1), "RobotEnable(1)");
        std::this_thread::sleep_for(500ms);
      }
      ValidateState(false);
      MoveToInitialPosition();
      CheckCall(robot_.ServoMoveStart(), "ServoMoveStart");
      servo_started_ = true;
      worker_ = std::thread(&RobotMotion::Loop, this);
    } catch (...) {
      Cleanup();
      throw;
    }
  }

  RobotMotion(const RobotMotion&) = delete;
  RobotMotion& operator=(const RobotMotion&) = delete;
  ~RobotMotion() { Cleanup(); }

  void Execute(const Action& action) {
    std::unique_lock<std::mutex> lock(mutex_);
    RaiseWorkerErrorLocked();
    if (pending_.has_value() || busy_) {
      throw std::runtime_error("previous robot action is still active");
    }
    const std::uint64_t sequence = ++submitted_sequence_;
    pending_ = action;
    condition_.notify_all();
    condition_.wait(lock, [&] {
      return completed_sequence_ >= sequence || !worker_error_.empty() ||
             stopped_ || g_interrupted.load(std::memory_order_relaxed);
    });
    RaiseWorkerErrorLocked();
    if (completed_sequence_ < sequence) {
      throw std::runtime_error("robot action interrupted");
    }
  }

 private:
  static void CheckCall(int result, const char* operation) {
    if (result != 0) {
      throw std::runtime_error(std::string(operation) + " failed: " +
                               std::to_string(result));
    }
  }

  void MoveToInitialPosition() {
    if (g_interrupted.load(std::memory_order_relaxed)) {
      throw std::runtime_error("initial position move interrupted");
    }
    JointPos actual;
    CheckCall(robot_.GetActualJointPosDegree(0, &actual),
              "GetActualJointPosDegree");
    if (MaximumJointErrorDegrees(actual) <= kInitialJointToleranceDegrees) {
      std::cout << "initial joint position already reached\n";
      return;
    }

    JointPos target(kInitialJointDegrees[0], kInitialJointDegrees[1],
                    kInitialJointDegrees[2], kInitialJointDegrees[3],
                    kInitialJointDegrees[4], kInitialJointDegrees[5]);
    ExaxisPos external_axes(0.0, 0.0, 0.0, 0.0);
    DescPose offset(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    std::cout << "moving to initial joint position "
                 "[130, -110, 130, -115, -90, -10] deg\n";
    CheckCall(robot_.MoveJ(&target, kToolId, kUserId,
                           kInitialMoveSpeedPercent, 100.0F, 100.0F,
                           &external_axes, -1.0F, 0, &offset),
              "MoveJ(initial position)");
    if (g_interrupted.load(std::memory_order_relaxed)) {
      throw std::runtime_error("initial position move interrupted");
    }
    CheckCall(robot_.GetActualJointPosDegree(0, &actual),
              "GetActualJointPosDegree");
    const double error = MaximumJointErrorDegrees(actual);
    if (error > kInitialJointToleranceDegrees) {
      throw std::runtime_error(
          "robot did not reach initial joint position: maximum error=" +
          std::to_string(error) + " deg");
    }
    std::cout << "initial joint position reached: maximum error=" << error
              << " deg\n";
  }

  void ValidateCoordinateFrames() {
    int tool = -1;
    int user = -1;
    CheckCall(robot_.GetActualTCPNum(1, &tool), "GetActualTCPNum");
    CheckCall(robot_.GetActualWObjNum(1, &user), "GetActualWObjNum");
    if (tool != kToolId || user != kUserId) {
      throw std::runtime_error("expected Tool/User 2/0, got " +
                               std::to_string(tool) + "/" +
                               std::to_string(user));
    }
  }

  void ValidateState(bool allow_disabled) {
    int communication = -1;
    int main_code = -1;
    int sub_code = -1;
    std::uint8_t emergency_stop = 1;
    ROBOT_STATE_PKG state{};
    CheckCall(robot_.GetSDKComState(&communication), "GetSDKComState");
    CheckCall(robot_.GetRobotErrorCode(&main_code, &sub_code),
              "GetRobotErrorCode");
    CheckCall(robot_.GetRobotEmergencyStopState(&emergency_stop),
              "GetRobotEmergencyStopState");
    CheckCall(robot_.GetRobotRealTimeState(&state),
              "GetRobotRealTimeState");
    if (communication != 0 || main_code != 0 || sub_code != 0 ||
        emergency_stop != 0 || state.EmergencyStop != 0 ||
        state.safety_stop0_state != 0 || state.safety_stop1_state != 0 ||
        state.collisionState != 0 || state.robot_state == 3 ||
        state.robot_state == 4 || (!allow_disabled && state.robot_mode != 0) ||
        (!allow_disabled && state.rbtEnableState != 1)) {
      throw std::runtime_error(
          "robot is not in a safe motion-ready state; inspect teach pendant");
    }
    DescPose pose;
    CheckCall(robot_.GetActualTCPPose(1, &pose), "GetActualTCPPose");
    const std::array<double, 6> values{
        pose.tran.x, pose.tran.y, pose.tran.z,
        pose.rpy.rx, pose.rpy.ry, pose.rpy.rz};
    if (!std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); })) {
      throw std::runtime_error("GetActualTCPPose returned a non-finite pose");
    }
  }

  void Loop() {
    try {
      auto deadline = std::chrono::steady_clock::now();
      std::optional<Action> active;
      int remaining = 0;
      std::uint64_t active_sequence = 0;
      while (true) {
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (stopped_ || g_interrupted.load(std::memory_order_relaxed)) {
            break;
          }
          if (!active.has_value() && pending_.has_value()) {
            ValidateState(false);
            active = *pending_;
            pending_.reset();
            remaining = kServoSubsteps;
            active_sequence = submitted_sequence_;
            busy_ = true;
            deadline = std::chrono::steady_clock::now();
          }
        }

        Action command{};
        if (active.has_value()) {
          command = *active;
        }
        DescPose increment = ServoIncrement(command);
        float gains[6] = {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
        CheckCall(robot_.ServoCart(1, &increment, gains, 0.0F, 0.0F,
                                  static_cast<float>(kServoPeriodSeconds),
                                  0.0F, 0.0F),
                  "ServoCart");

        if (active.has_value() && --remaining == 0) {
          std::lock_guard<std::mutex> lock(mutex_);
          active.reset();
          busy_ = false;
          completed_sequence_ = active_sequence;
          condition_.notify_all();
        }
        deadline += std::chrono::microseconds(8000);
        const auto now = std::chrono::steady_clock::now();
        if (deadline < now) {
          deadline = now;
        }
        std::this_thread::sleep_until(deadline);
      }
    } catch (const std::exception& error) {
      std::lock_guard<std::mutex> lock(mutex_);
      worker_error_ = error.what();
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopped_ = true;
      condition_.notify_all();
    }
  }

  void RaiseWorkerErrorLocked() const {
    if (!worker_error_.empty()) {
      throw std::runtime_error("robot servo worker failed: " + worker_error_);
    }
  }

  void Cleanup() noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopped_ = true;
      condition_.notify_all();
    }
    if (worker_.joinable()) {
      worker_.join();
    }
    if (servo_started_) {
      robot_.ServoMoveEnd();
      servo_started_ = false;
    }
    if (connected_) {
      robot_.CloseRPC();
      connected_ = false;
    }
  }

  FRRobot robot_;
  bool connected_ = false;
  bool servo_started_ = false;
  std::thread worker_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::optional<Action> pending_;
  bool busy_ = false;
  bool stopped_ = false;
  std::uint64_t submitted_sequence_ = 0;
  std::uint64_t completed_sequence_ = 0;
  std::string worker_error_;
};

void PrintAction(int step, const Action& action) {
  std::cout << "step=" << step << " action=[";
  for (std::size_t index = 0; index < action.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << action[index];
  }
  std::cout << "]\n";
}

class TestRunner {
 public:
  void Check(bool condition, const std::string& message) {
    if (!condition) {
      ++failures_;
      std::cerr << "SELF-TEST FAIL: " << message << '\n';
    }
  }

  template <typename Callback>
  void ExpectThrow(Callback callback, const std::string& message) {
    try {
      callback();
      Check(false, message);
    } catch (const std::exception&) {
    }
  }

  int failures() const { return failures_; }

 private:
  int failures_ = 0;
};

int RunSelfTest() {
  TestRunner tests;
  tests.Check(Base64Encode({'M', 'a', 'n'}) == "TWFu", "base64 encoding");
  ValidateHealthResponse(
      R"({"ready":true,"backend":"openvla","unnorm_key":"fr5_dataset"})");
  tests.ExpectThrow(
      [] { ValidateHealthResponse(R"({"ready":true,"backend":"openvla","unnorm_key":"wrong"})"); },
      "wrong unnorm key must fail");

  const Action valid = ParseActionResponse(
      R"({"action":[0.01,-0.01,0.0,0.01,0.0,-0.01,1.0]})", 0.04,
      0.10);
  tests.Check(valid[0] == 0.01 && valid[6] == 1.0,
              "valid action parsing");
  const Action scaled = ScaleTranslation(valid, 1.2, 0.04, 0.10);
  tests.Check(std::abs(scaled[0] - 0.012) < 1e-12 &&
                  scaled[3] == valid[3] && scaled[6] == valid[6],
              "translation-only action scaling");
  tests.ExpectThrow(
      [&] { ScaleTranslation(valid, 3.0, 0.04, 0.10); },
      "scaled translation above the safety limit must fail");
  tests.ExpectThrow(
      [] {
        ParseActionResponse(R"({"action":[0.05,0,0,0,0,0,1]})", 0.04,
                            0.10);
      },
      "oversized translation must fail");
  tests.ExpectThrow(
      [] {
        ParseActionResponse(R"({"action":[0,0,0,0.11,0,0,1]})", 0.04,
                            0.10);
      },
      "oversized rotation must fail");
  tests.ExpectThrow(
      [] {
        ParseActionResponse(R"({"action":[0,0,0,0,0,0]})", 0.04, 0.10);
      },
      "wrong action length must fail");

  const DescPose increment = ServoIncrement(valid);
  tests.Check(std::abs(increment.tran.x * kServoSubsteps - 10.0) < 1e-12,
              "translation conversion and subdivision");
  tests.Check(std::abs(increment.rpy.rx * kServoSubsteps * kPi / 180.0 -
                       valid[3]) < 1e-12,
              "rotation conversion and subdivision");

  JointPos initial(kInitialJointDegrees[0], kInitialJointDegrees[1],
                   kInitialJointDegrees[2], kInitialJointDegrees[3],
                   kInitialJointDegrees[4], kInitialJointDegrees[5]);
  tests.Check(MaximumJointErrorDegrees(initial) == 0.0 &&
                  kInitialMoveSpeedPercent == 10.0F &&
                  kInitialJointToleranceDegrees == 0.15,
              "fixed initial joint position and movement limits");
  initial.jPos[2] += 0.16;
  tests.Check(MaximumJointErrorDegrees(initial) >
                  kInitialJointToleranceDegrees,
              "initial joint position tolerance");

  const std::vector<std::uint8_t> request =
      GripperWriteRequest(0x9C40, 100);
  tests.Check(request.size() == 11 && request[0] == 1 && request[1] == 0x10 &&
                  request[2] == 0x9C && request[3] == 0x40 &&
                  ModbusCrc16(request.data(), request.size()) == 0,
              "gripper write frame and CRC");

  if (tests.failures() != 0) {
    return 1;
  }
  std::cout << "SELF-TEST PASSED\n";
  return 0;
}

int Run(const Options& options) {
  HttpClient http;
  http.CheckHealth();
  std::cout << "server ready: " << kServerUrl
            << " backend=" << kExpectedBackend
            << " unnorm_key=" << kExpectedUnnormKey << '\n';
  Camera camera;
  std::cout << "camera ready: serial=" << kCameraSerial << ' '
            << kImageWidth << 'x' << kImageHeight << '@' << kImageRateHz
            << " Hz\n";
  std::cout << "translation scale: " << options.translation_scale << '\n';

  if (!options.enable_motion) {
    std::cout << "DRY RUN: robot and gripper will not be opened\n";
    for (int step = 1;
         step <= options.max_steps &&
         !g_interrupted.load(std::memory_order_relaxed);
         ++step) {
      const Action action = ScaleTranslation(
          http.Predict(camera.CaptureJpeg(), options.instruction,
                       options.max_translation_m, options.max_rotation_rad),
          options.translation_scale, options.max_translation_m,
          options.max_rotation_rad);
      PrintAction(step, action);
    }
    return 0;
  }

  Gripper gripper;
  gripper.Apply(1.0);
  RobotMotion motion;
  std::cout << "motion armed from fixed initial position: Tool/User=2/0; "
               "Ctrl+C or emergency stop exits\n";
  for (int step = 1;
       step <= options.max_steps &&
       !g_interrupted.load(std::memory_order_relaxed);
       ++step) {
    const Action action = ScaleTranslation(
        http.Predict(camera.CaptureJpeg(), options.instruction,
                     options.max_translation_m, options.max_rotation_rad),
        options.translation_scale, options.max_translation_m,
        options.max_rotation_rad);
    PrintAction(step, action);
    motion.Execute(action);
    gripper.Apply(action[6]);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);
  try {
    const Options options = ParseOptions(argc, argv);
    return options.self_test ? RunSelfTest() : Run(options);
  } catch (const std::exception& error) {
    std::cerr << "run_fr5_openvla: " << error.what() << '\n';
    return 1;
  }
}
