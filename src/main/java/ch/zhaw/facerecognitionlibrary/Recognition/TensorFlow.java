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

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

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
    private String inputLayer;
    private String outputLayer;

    private int inputSize;
    private int outputSize;

    private int channels;

    private Recognition rec;

    private TensorFlowInferenceInterface inferenceInterface;

    private boolean logStats = false;

    public TensorFlow(Context context, int method) {
        String dataPath = FileHelper.TENSORFLOW_PATH;
        PreferencesHelper preferencesHelper = new PreferencesHelper(context);
        inputSize = preferencesHelper.getTensorFlowInputSize();
        outputSize = preferencesHelper.getTensorFlowOutputSize();
        inputLayer = preferencesHelper.getTensorFlowInputLayer();
        outputLayer = preferencesHelper.getTensorFlowOutputLayer();
        channels = preferencesHelper.getTensorFlowChannels();
        String modelFile = preferencesHelper.getTensorFlowModelFile();
        Boolean classificationMethod = preferencesHelper.getClassificationMethodTFCaffe();

        inferenceInterface = new TensorFlowInferenceInterface(context.getAssets(), dataPath + modelFile);

        if(classificationMethod){
            rec = new SupportVectorMachine(context, method);
        }
        else {
            rec = new KNearestNeighbor(context, method);
        }
    }

    public TensorFlow(Context context, int inputSize, int outputSize, String inputLayer, String outputLayer, String modelFile){
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.inputLayer = inputLayer;
        this.outputLayer = outputLayer;

        inferenceInterface = new TensorFlowInferenceInterface(context.getAssets(), modelFile);
    }

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

        inferenceInterface.feed(inputLayer, getPixels(img), 1, inputSize, inputSize, channels);
        inferenceInterface.run(new String[]{outputLayer}, logStats);
        float[] outputs = new float[outputSize];
        inferenceInterface.fetch(outputLayer, outputs);

        List<Float> fVector = new ArrayList<>();
        for(float o : outputs){
            fVector.add(o);
        }

        return Converters.vector_float_to_Mat(fVector);
    }

    private float[] getPixels(Mat img){
        img.convertTo(img, CvType.CV_32FC4);
        float[] pixels = new float[img.rows() * img.cols() * channels];
        for (int col=0; col<img.cols(); col++){
            for (int row=0; row<img.rows(); row++){
                pixels[col*row] = (float)img.get(row, col)[0];
                pixels[col*row + 1] = (float)img.get(row, col)[1];
                pixels[col*row + 2] = (float)img.get(row, col)[2];
            }
        }
        return pixels;
    }
}
