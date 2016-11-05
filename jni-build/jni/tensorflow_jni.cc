/* Copyright 2015 Google Inc. All Rights Reserved.
Modifications copyright (C) 2016 Michael Sladoje and Mike Schälchli.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_jni.h"

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>

#include <jni.h>
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>
#include <queue>
#include <sstream>
#include <string>

#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/stat_summarizer.h"
#include "jni_utils.h"

using namespace tensorflow;

// Global variables that holds the Tensorflow classifier.
static std::unique_ptr<tensorflow::Session> session;

static bool g_compute_graph_initialized = false;
//static mutex g_compute_graph_mutex(base::LINKER_INITIALIZED);

static int g_tensorflow_input_size;  // The image size for the mognet input.
static int g_image_mean;  // The image mean.
static std::unique_ptr<StatSummarizer> g_stats;

// For basic benchmarking.
static int g_num_runs = 0;
static int64 g_timing_total_us = 0;
static Stat<int64> g_frequency_start;
static Stat<int64> g_frequency_end;

#ifdef LOG_DETAILED_STATS
static const bool kLogDetailedStats = true;
#else
static const bool kLogDetailedStats = false;
#endif

// Improve benchmarking by limiting runs to predefined amount.
// 0 (default) denotes infinite runs.
#ifndef MAX_NUM_RUNS
#define MAX_NUM_RUNS 0
#endif

#ifdef SAVE_STEP_STATS
static const bool kSaveStepStats = true;
#else
static const bool kSaveStepStats = false;
#endif

inline static int64 CurrentThreadTimeUs() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}

JNIEXPORT jint JNICALL
TENSORFLOW_METHOD(initializeTensorflow)(
    JNIEnv* env, jobject thiz, jobject java_asset_manager,
    jstring model, jint mognet_input_size, jint image_mean) {
  g_num_runs = 0;
  g_timing_total_us = 0;
  g_frequency_start.Reset();
  g_frequency_end.Reset();

  //MutexLock input_lock(&g_compute_graph_mutex);
  if (g_compute_graph_initialized) {
    LOG(INFO) << "Compute graph already loaded. skipping.";
    return 0;
  }

  const int64 start_time = CurrentThreadTimeUs();

  const char* const model_cstr = env->GetStringUTFChars(model, NULL);

  g_tensorflow_input_size = mognet_input_size;
  g_image_mean = image_mean;

  LOG(INFO) << "Loading Tensorflow.";

  LOG(INFO) << "Making new SessionOptions.";
  tensorflow::SessionOptions options;
  tensorflow::ConfigProto& config = options.config;
  LOG(INFO) << "Got config, " << config.device_count_size() << " devices";

  session.reset(tensorflow::NewSession(options));
  LOG(INFO) << "Session created.";

  tensorflow::GraphDef tensorflow_graph;
  LOG(INFO) << "Graph created.";

  AAssetManager* const asset_manager =
      AAssetManager_fromJava(env, java_asset_manager);
  LOG(INFO) << "Acquired AssetManager.";

  LOG(INFO) << "Reading file to proto: " << model_cstr;
  ReadFileToProto(asset_manager, model_cstr, &tensorflow_graph);

  g_stats.reset(new StatSummarizer(tensorflow_graph));

  LOG(INFO) << "Creating session.";
  tensorflow::Status s = session->Create(tensorflow_graph);
  if (!s.ok()) {
    LOG(ERROR) << "Could not create Tensorflow Graph: " << s;
    return -1;
  }

  // Clear the proto to save memory space.
  tensorflow_graph.Clear();
  LOG(INFO) << "Tensorflow graph loaded from: " << model_cstr;

  g_compute_graph_initialized = true;

  const int64 end_time = CurrentThreadTimeUs();
  LOG(INFO) << "Initialization done in " << (end_time - start_time) / 1000
            << "ms";

  return 0;
}

namespace {
typedef struct {
  uint8 red;
  uint8 green;
  uint8 blue;
  uint8 alpha;
} RGBA;
}  // namespace

// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
static void GetTopN(
    const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>,
                           Eigen::Aligned>& prediction,
    const int num_results, const float threshold,
    std::vector<std::pair<float, int> >* top_results) {
  // Will contain top N results in ascending order.
  std::priority_queue<std::pair<float, int>,
      std::vector<std::pair<float, int> >,
      std::greater<std::pair<float, int> > > top_result_pq;

  const int count = prediction.size();
  for (int i = 0; i < count; ++i) {
    const float value = prediction(i);

    // Only add it if it beats the threshold and has a chance at being in
    // the top N.
    if (value < threshold) {
      continue;
    }

    top_result_pq.push(std::pair<float, int>(value, i));

    // If at capacity, kick the smallest value out.
    if (top_result_pq.size() > num_results) {
      top_result_pq.pop();
    }
  }

  // Copy to output vector and reverse into descending order.
  while (!top_result_pq.empty()) {
    top_results->push_back(top_result_pq.top());
    top_result_pq.pop();
  }
  std::reverse(top_results->begin(), top_results->end());
}

static int64 GetCpuSpeed() {
  string scaling_contents;
  ReadFileToString(nullptr,
                   "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
                   &scaling_contents);
  std::stringstream ss(scaling_contents);
  int64 result;
  ss >> result;
  return result;
}

static std::string ClassifyImage(const RGBA* const bitmap_src,
                                  const char* cInputLayer,
                                  const char* cOutputLayer,
                                  int outputSize,
                                 const int in_stride,
                                 const int width, const int height) {
  // Force the app to quit if we've reached our run quota, to make
  // benchmarks more reproducible.
  if (MAX_NUM_RUNS > 0 && g_num_runs >= MAX_NUM_RUNS) {
    LOG(INFO) << "Benchmark complete. "
              << (g_timing_total_us / g_num_runs / 1000) << "ms/run avg over "
              << g_num_runs << " runs.";
    LOG(INFO) << "";
    exit(0);
  }

  ++g_num_runs;

  // Create input tensor
  tensorflow::Tensor input_tensor(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape({
          1, g_tensorflow_input_size, g_tensorflow_input_size, 3}));

  auto input_tensor_mapped = input_tensor.tensor<float, 4>();

  LOG(INFO) << "Tensorflow: Copying Data.";
  for (int i = 0; i < g_tensorflow_input_size; ++i) {
    const RGBA* src = bitmap_src + i * g_tensorflow_input_size;
    for (int j = 0; j < g_tensorflow_input_size; ++j) {
       // Copy 3 values
      input_tensor_mapped(0, i, j, 0) =
          (static_cast<float>(src->red) - g_image_mean)/g_image_mean;
      input_tensor_mapped(0, i, j, 1) =
          (static_cast<float>(src->green) - g_image_mean)/g_image_mean;
      input_tensor_mapped(0, i, j, 2) =
          (static_cast<float>(src->blue) - g_image_mean)/g_image_mean;
      ++src;
    }
  }

  std::vector<std::pair<std::string, tensorflow::Tensor> > input_tensors(
      {{cInputLayer, input_tensor}});

  VLOG(0) << "Start computing.";
  std::vector<tensorflow::Tensor> output_tensors;
  std::vector<std::string> output_names({cOutputLayer});

  tensorflow::Status s;
  int64 start_time, end_time;

  start_time = CurrentThreadTimeUs();
  s = session->Run(input_tensors, output_names, {}, &output_tensors);
  end_time = CurrentThreadTimeUs();

  const int64 elapsed_time_inf = end_time - start_time;
  g_timing_total_us += elapsed_time_inf;
  VLOG(0) << "End computing. Ran in " << elapsed_time_inf / 1000 << "ms ("
          << (g_timing_total_us / g_num_runs / 1000) << "ms avg over "
          << g_num_runs << " runs)";

  if (!s.ok()) {
    LOG(ERROR) << "Error during inference: " << s;
    return "";
  }

  VLOG(0) << "Reading from layer " << output_names[0];
  tensorflow::Tensor* output = &output_tensors[0];

  return output->SummarizeValue(outputSize);
}

JNIEXPORT jstring JNICALL
TENSORFLOW_METHOD(classifyImageRgb)(
    JNIEnv* env, jobject thiz, jstring inputLayer, jstring outputLayer, jint outputSize, jintArray image, jint width, jint height) {
      const char* const cInputLayer = env->GetStringUTFChars(inputLayer, NULL);
      const char* const cOutputLayer = env->GetStringUTFChars(outputLayer, NULL);

  // Copy image into currFrame.
  jboolean iCopied = JNI_FALSE;
  jint* pixels = env->GetIntArrayElements(image, &iCopied);

  std::string result = ClassifyImage(
      reinterpret_cast<const RGBA*>(pixels), cInputLayer, cOutputLayer, outputSize, width * 4, width, height);

  env->ReleaseIntArrayElements(image, pixels, JNI_ABORT);

  return env->NewStringUTF(result.c_str());
}

JNIEXPORT jstring JNICALL
TENSORFLOW_METHOD(classifyImageBmp)(
    JNIEnv* env, jobject thiz, jstring inputLayer, jstring outputLayer, jint outputSize, jobject bitmap) {
      const char* const cInputLayer = env->GetStringUTFChars(inputLayer, NULL);
      const char* const cOutputLayer = env->GetStringUTFChars(outputLayer, NULL);

  // Obtains the bitmap information.
  AndroidBitmapInfo info;
  CHECK_EQ(AndroidBitmap_getInfo(env, bitmap, &info),
           ANDROID_BITMAP_RESULT_SUCCESS);
  void* pixels;
  CHECK_EQ(AndroidBitmap_lockPixels(env, bitmap, &pixels),
           ANDROID_BITMAP_RESULT_SUCCESS);
  LOG(INFO) << "Image dimensions: " << info.width << "x" << info.height
            << " stride: " << info.stride;
  // TODO(jiayq): deal with other formats if necessary.
  if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
    return env->NewStringUTF(
        "Error: Android system is not using RGBA_8888 in default.");
  }

  std::string result = ClassifyImage(
      static_cast<const RGBA*>(pixels), cInputLayer, cOutputLayer, outputSize, info.stride, info.width, info.height);

  // Finally, unlock the pixels
  CHECK_EQ(AndroidBitmap_unlockPixels(env, bitmap),
           ANDROID_BITMAP_RESULT_SUCCESS);

  return env->NewStringUTF(result.c_str());
}
