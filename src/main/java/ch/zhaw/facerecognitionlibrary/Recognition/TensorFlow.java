/* Copyright 2016 Michael Sladoje and Mike Schälchli. All Rights Reserved.

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

package ch.zhaw.facerecognitionlibrary.Recognition;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.util.ArrayList;
import java.util.List;

import ch.zhaw.facerecognitionlibrary.Helpers.FileHelper;
import ch.zhaw.facerecognitionlibrary.Helpers.PreferencesHelper;

/***************************************************************************************
 *    Title: TensorFlowAndroidDemo
 *    Author: miyosuda
 *    Date: 23.04.2016
 *    Code version: -
 *    Availability: https://github.com
 *
 ***************************************************************************************/

public class TensorFlow implements Recognition {
    private static final String STRING_SPLIT_CHARACTER = " ";

    private String inputLayer;
    private String outputLayer;

    private int inputSize;
    private int outputSize;

    Recognition rec;

    public TensorFlow(Context context, int method) {
        String dataPath = FileHelper.TENSORFLOW_PATH;
        PreferencesHelper preferencesHelper = new PreferencesHelper(context);
        inputSize = preferencesHelper.getTensorFlowInputSize();
        int imageMean = preferencesHelper.getTensorFlowImageMean();
        outputSize = preferencesHelper.getTensorFlowOutputSize();
        inputLayer = preferencesHelper.getTensorFlowInputLayer();
        outputLayer = preferencesHelper.getTensorFlowOutputLayer();
        String modelFile = preferencesHelper.getTensorFlowModelFile();
        Boolean classificationMethod = preferencesHelper.getClassificationMethodTFCaffe();

        initializeTensorflow(context.getAssets(), dataPath + modelFile, inputSize, imageMean);

        if(classificationMethod){
            rec = new SupportVectorMachine(context, method);
        }
        else {
            rec = new KNearestNeighbor(context, method);
        }
    }

    public TensorFlow(Context context, int inputSize, int imageMean, int outputSize, String inputLayer, String outputLayer, String modelFile){
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.inputLayer = inputLayer;
        this.outputLayer = outputLayer;

        initializeTensorflow(context.getAssets(), modelFile, inputSize, imageMean);
    }

    // link jni library
    static {
        System.loadLibrary("tensorflow");
    }

    // connect the native functions
    private native int initializeTensorflow(AssetManager assetManager,
                                             String model,
                                             int inputSize,
                                             int imageMean);
    private native String classifyImageBmp(String inputLayer, String outputLayer, int outputSize, Bitmap bitmap);
    private native String classifyImageRgb(String inputLayer, String outputLayer, int outputSize, int[] output, int width, int height);

    @Override
    public boolean train() {
        return rec.train();
    }

    @Override
    public String recognize(Mat img, String expectedLabel) {
        return rec.recognize(getFeatureVector(img), expectedLabel);
    }

    @Override
    public void saveToFile() {

    }

    @Override
    public void loadFromFile() {

    }

    @Override
    public void saveTestData() {
        rec.saveTestData();
    }

    @Override
    public void addImage(Mat img, String label, boolean featuresAlreadyExtracted) {
        if (featuresAlreadyExtracted){
            rec.addImage(img, label, true);
        } else {
            rec.addImage(getFeatureVector(img), label, true);
        }
    }

    public Mat getFeatureVector(Mat img){
        Imgproc.resize(img, img, new Size(inputSize, inputSize));

        Bitmap bmp = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(img, bmp);

        String[] sVector = classifyImageBmp(inputLayer, outputLayer, outputSize, bmp).split(STRING_SPLIT_CHARACTER);

        System.out.println(sVector.length);

        List<Float> fVector = new ArrayList<>();
        for(String s : sVector){
            fVector.add(Float.parseFloat(s));
        }

        return Converters.vector_float_to_Mat(fVector);
    }
}
