#Android Face Recognition with Deep Learning
##Acknowledgements
This app was developed by Michael Sladoje and Mike Schälchli during a bachelor thesis at the Zurich University of Applied Sciences.

Acknowledgements go to the thesis supervisors Dr. Martin Loeser, Dr. Oliver Dürr, Diego Browarnik and all the contributors of our code sources.

Code has been derived from the following sources:
- OpenCV - https://github.com/opencv/opencv
- LIBSVM - https://github.com/cjlin1/libsvm
- AndroidLibSvm - https://github.com/yctung/AndroidLibSvm
- TensorFlow - https://github.com/tensorflow/tensorflow
- TensorFlow Android Demo - https://github.com/miyosuda/TensorFlowAndroidDemo
- Caffe - https://github.com/BVLC/caffe
- caffe-android-demo - https://github.com/sh1r0/caffe-android-demo
- caffe-android-lib - https://github.com/sh1r0/caffe-android-lib

##App architecture
![alt tag](https://github.com/Qualeams/Android-Face-Recognition-with-Deep-Learning/blob/master/AppArchitecture.png)

##Usage
###Compilation
####APK - Package
The app can be downloaded directly from the Google Play Store - [Face Recognition](https://play.google.com/store/apps/details?id=ch.zhaw.facerecognition).
####Android SDK - Java
The source can be compiled using Android Studio (common gradle scripts).

####Android NDK - C++
The libs for LIBSVM and TensorFlow can be compiled outside of Android Studio with the make command (the Makefile is located under /jni-build).
There are 3 different usages:
- make clean (executes ndk-build clean)
- make (executes ndk-build)
- make install (copies the libs to the folder /app/src/main/jniLibs/armeabi-v7a)

###User manual
The user manual can be found [here](https://github.com/Qualeams/Android-Face-Recognition-with-Deep-Learning/blob/master/USER%20MANUAL.md)
