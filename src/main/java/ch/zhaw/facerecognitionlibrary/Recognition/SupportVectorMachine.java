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
import android.content.SharedPreferences;
import android.preference.PreferenceManager;

import org.opencv.core.Mat;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import ch.zhaw.facerecognitionlibrary.Helpers.FileHelper;
import ch.zhaw.facerecognitionlibrary.Helpers.OneToOneMap;

/***************************************************************************************
 *    Title: AndroidLibSvm
 *    Author: yctung
 *    Date: 16.09.2015
 *    Code version: -
 *    Availability: https://github.com/
 *
 ***************************************************************************************/

public class SupportVectorMachine implements Recognition {
    private Context context;
    private FileHelper fh;
    private File trainingFile;
    private File predictionFile;
    private File testFile;
    private List<String> trainingList;
    private List<String> testList;
    private OneToOneMap<String, Integer> labelMap;
    private OneToOneMap<String, Integer> labelMapTest;
    private int method;

    public SupportVectorMachine(Context context, int method) {
        this.context = context;
        fh = new FileHelper();
        trainingFile = fh.createSvmTrainingFile();
        predictionFile = fh.createSvmPredictionFile();
        testFile = fh.createSvmTestFile();
        trainingList = new ArrayList<>();
        testList = new ArrayList<>();
        labelMap = new OneToOneMap<String, Integer>();
        labelMapTest = new OneToOneMap<String, Integer>();
        this.method = method;
        if(method == RECOGNITION){
            loadFromFile();
        }
    }

    // link jni library
    static {
        System.loadLibrary("jnilibsvm");
    }

    // connect the native functions
    private native void jniSvmTrain(String cmd);
    private native void jniSvmPredict(String cmd);

    @Override
    public boolean train() {

        fh.saveStringList(trainingList, trainingFile);

        // linear kernel -t 0
        SharedPreferences sharedPref = PreferenceManager.getDefaultSharedPreferences((context.getApplicationContext()));
        String svmTrainOptions = sharedPref.getString("key_svmTrainOptions", "-t 0 ");
        String training = trainingFile.getAbsolutePath();
        String model = trainingFile.getAbsolutePath() + "_model";
        jniSvmTrain(svmTrainOptions + " " + training + " " + model);

        saveToFile();
        return true;
    }

    @Override
    public String recognize(Mat img, String expectedLabel) {
        try {
            FileWriter fw = new FileWriter(predictionFile, false);
            String line = imageToSvmString(img, expectedLabel);
            testList.add(line);
            fw.append(line);
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        String prediction = predictionFile.getAbsolutePath();
        String model = trainingFile.getAbsolutePath() + "_model";
        String output = predictionFile.getAbsolutePath() + "_output";
        jniSvmPredict(prediction + " " + model + " " + output);

        try {
            BufferedReader buf = new BufferedReader(new FileReader(output));
            int iLabel = Integer.valueOf(buf.readLine());
            buf.close();
            return labelMap.getKey(iLabel);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    @Override
    public void saveToFile() {
        if(method == TRAINING){
            fh.saveLabelMapToFile(fh.SVM_PATH, labelMap, "train");
        } else {
            fh.saveLabelMapToFile(fh.SVM_PATH, labelMapTest, "test");
        }
    }

    @Override
    public void saveTestData(){
        fh.saveStringList(testList, testFile);
    }

    @Override
    public void loadFromFile() {
        labelMap = fh.getLabelMapFromFile(fh.SVM_PATH);
    }

    @Override
    public void addImage(Mat img, String label, boolean featuresAlreadyExtracted) {
        // Ignore featuresAlreadyExtracted because either SVM get the features from TensorFlow or Caffe, or it takes the image reshaping method (image itself)
        if(method == TRAINING){
            trainingList.add(imageToSvmString(img, label));
        } else {
            testList.add(imageToSvmString(img, label));
        }
    }

    public Mat getFeatureVector(Mat img){
        return img.reshape(1,1);
    }

    private String imageToSvmString(Mat img, String label){
        int iLabel = 0;
        if(method == TRAINING){
            if (labelMap.containsKey(label)) {
                iLabel = labelMap.getValue(label);
            } else {
                iLabel = labelMap.size() + 1;
                labelMap.put(label, iLabel);
            }
        } else {
            if (labelMapTest.containsKey(label)){
                iLabel = labelMapTest.getValue(label);
            } else {
                iLabel = labelMapTest.size() + 1;
                labelMapTest.put(label, iLabel);
            }
        }
        String result = String.valueOf(iLabel);
        img = getFeatureVector(img);
        for (int i=0; i<img.cols(); i++){
            result = result + " " + i + ":" + img.get(0,i)[0];
        }
        result += "\n";
        return result;
    }
}
