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

TENSORFLOW_CFLAGS	  := -frtti \
  -fstack-protector-strong \
  -fpic \
  -ffunction-sections \
  -funwind-tables \
  -no-canonical-prefixes \
  -fno-canonical-system-headers \
  '-march=armv7-a' \
  '-mfpu=vfpv3-d16' \
  '-mfloat-abi=softfp' \
  '-std=c++11' '-mfpu=neon' -O2 \
  '-DMIN_LOG_LEVEL=0' \

TENSORFLOW_SRC_FILES := tensorflow_jni.cc \
	jni_utils.cc \

LOCAL_MODULE    := tensorflow
LOCAL_ARM_MODE  := arm
LOCAL_SRC_FILES := $(TENSORFLOW_SRC_FILES)
LOCAL_CFLAGS    := $(TENSORFLOW_CFLAGS)

LOCAL_LDLIBS    := \
	-Wl,-whole-archive \
	$(LOCAL_PATH)/libs/$(TARGET_ARCH_ABI)/libandroid_tensorflow_kernels.lo \
	$(LOCAL_PATH)/libs/$(TARGET_ARCH_ABI)/libandroid_tensorflow_lib.lo \
	$(LOCAL_PATH)/libs/$(TARGET_ARCH_ABI)/libandroid_tensorflow_lib_lite.lo \
	$(LOCAL_PATH)/libs/$(TARGET_ARCH_ABI)/libre2.a \
	$(LOCAL_PATH)/libs/$(TARGET_ARCH_ABI)/libprotos_all_cc.a \
	$(LOCAL_PATH)/libs/$(TARGET_ARCH_ABI)/libprotobuf.a \
	$(LOCAL_PATH)/libs/$(TARGET_ARCH_ABI)/libprotobuf_lite.a \
	-Wl,-no-whole-archive \
	$(NDK_ROOT)/sources/cxx-stl/gnu-libstdc++/4.9/libs/$(TARGET_ARCH_ABI)/libgnustl_static.a \
	$(NDK_ROOT)/sources/cxx-stl/gnu-libstdc++/4.9/libs/$(TARGET_ARCH_ABI)/libsupc++.a \
	-llog -landroid -lm -ljnigraphics -pthread -no-canonical-prefixes '-march=armv7-a' -Wl,--fix-cortex-a8 -Wl,-S \

LOCAL_C_INCLUDES += $(LOCAL_PATH)/include \
	$(LOCAL_PATH)/genfiles \
	$(LOCAL_PATH)/include/third_party/eigen3 \
	$(LOCAL_PATH)/include/external/eigen_archive \
	$(LOCAL_PATH)/include/external/eigen_archive/eigen-eigen-4c94692de3e5 \
	$(LOCAL_PATH)/include/google/protobuf/src

NDK_MODULE_PATH := $(call my-dir)

include $(BUILD_SHARED_LIBRARY)
