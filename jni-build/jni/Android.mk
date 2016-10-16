LOCAL_PATH:=$(call my-dir)

#Module jnilibsvm

include $(CLEAR_VARS)

LOCAL_MODULE	:= jnilibsvm
LOCAL_CFLAGS    := -DDEV_NDK=1
LOCAL_SRC_FILES := \
	common.cpp jnilibsvm.cpp \
	libsvm/svm-train.cpp \
	libsvm/svm-predict.cpp \
	libsvm/svm.cpp

LOCAL_LDLIBS	+= -llog -ldl

include $(BUILD_SHARED_LIBRARY)

#Module tensorflow_jni

include $(CLEAR_VARS)

TENSORFLOW_CFLAGS	  := \
  -fstack-protector-strong \
  -fpic \
  -ffunction-sections \
  -funwind-tables \
  -no-canonical-prefixes \
  -fno-canonical-system-headers \
  -DHAVE_PTHREAD \
  -Wall \
  -Wwrite-strings \
  -Woverloaded-virtual \
  -Wno-sign-compare \
  '-Wno-error=unused-function' \
  '-std=c++11' \
  -fno-exceptions \
  -DEIGEN_AVOID_STL_ARRAY \
  '-std=c++11' \
  '-DMIN_LOG_LEVEL=0' \
  -DTF_LEAN_BINARY \
  -O2 \
  -Os \
  -frtti \
  -MD \
	-DPROTOBUF_DEPRECATED_ATTR="" \

TENSORFLOW_SRC_FILES := tensorflow_jni.cc \
	jni_utils.cc \

LOCAL_MODULE    := tensorflow
LOCAL_ARM_MODE  := arm
LOCAL_SRC_FILES := $(TENSORFLOW_SRC_FILES)
LOCAL_CFLAGS    := $(TENSORFLOW_CFLAGS)

LOCAL_LDLIBS    := \
	-Wl,-whole-archive \
	$(LOCAL_PATH)/libs/$(TARGET_ARCH_ABI)/libandroid_tensorflow_lib.lo \
	$(LOCAL_PATH)/libs/$(TARGET_ARCH_ABI)/libandroid_tensorflow_kernels.lo \
	$(LOCAL_PATH)/libs/$(TARGET_ARCH_ABI)/libandroid_tensorflow_lib_lite.lo \
	$(LOCAL_PATH)/libs/$(TARGET_ARCH_ABI)/libprotos_all_cc.a \
	$(LOCAL_PATH)/libs/$(TARGET_ARCH_ABI)/libprotobuf.a \
	$(LOCAL_PATH)/libs/$(TARGET_ARCH_ABI)/libprotobuf_lite.a \
	-Wl,-no-whole-archive \
	$(NDK_ROOT)/sources/cxx-stl/gnu-libstdc++/4.9/libs/$(TARGET_ARCH_ABI)/libgnustl_static.a \
	$(NDK_ROOT)/sources/cxx-stl/gnu-libstdc++/4.9/libs/$(TARGET_ARCH_ABI)/libsupc++.a \
	-landroid \
	-ljnigraphics \
	-llog \
	-lm \
	-z defs \
	-s \
	-Wl,--exclude-libs,ALL \
	-lz \
	-static-libgcc \
	-no-canonical-prefixes \
	-Wl,-S \

LOCAL_C_INCLUDES += $(LOCAL_PATH)/include \
	$(LOCAL_PATH)/genfiles \
	$(LOCAL_PATH)/include/google/protobuf \
	$(LOCAL_PATH)/include/external/bazel_tools \
	$(LOCAL_PATH)/include/external/eigen_archive \
	$(LOCAL_PATH)/include/google/protobuf/src \
	$(LOCAL_PATH)/include/external/bazel_tools/tools/cpp/gcc3 \

NDK_MODULE_PATH := $(call my-dir)

include $(BUILD_SHARED_LIBRARY)
