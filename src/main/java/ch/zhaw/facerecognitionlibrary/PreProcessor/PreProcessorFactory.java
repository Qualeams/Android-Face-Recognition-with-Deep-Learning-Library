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

package ch.zhaw.facerecognitionlibrary.PreProcessor;

import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.Resources;
import android.preference.PreferenceManager;

import org.opencv.core.Mat;
import org.opencv.core.Rect;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import ch.zhaw.facerecognitionlibrary.Helpers.Eyes;
import ch.zhaw.facerecognitionlibrary.Helpers.FaceDetection;
import ch.zhaw.facerecognitionlibrary.PreProcessor.BrightnessCorrection.GammaCorrection;
import ch.zhaw.facerecognitionlibrary.PreProcessor.Contours.DifferenceOfGaussian;
import ch.zhaw.facerecognitionlibrary.PreProcessor.Contours.LocalBinaryPattern;
import ch.zhaw.facerecognitionlibrary.PreProcessor.Contours.Masking;
import ch.zhaw.facerecognitionlibrary.PreProcessor.ContrastAdjustment.HistogrammEqualization;
import ch.zhaw.facerecognitionlibrary.PreProcessor.StandardPostprocessing.Resize;
import ch.zhaw.facerecognitionlibrary.PreProcessor.StandardPreprocessing.Crop;
import ch.zhaw.facerecognitionlibrary.PreProcessor.StandardPreprocessing.EyeAlignment;
import ch.zhaw.facerecognitionlibrary.PreProcessor.StandardPreprocessing.GrayScale;
import ch.zhaw.facerecognitionlibrary.R;

public class PreProcessorFactory {
    private Context context;
    private int N;
    private PreProcessor preProcessor;
    private List<Mat> images;
    public CommandFactory commandFactory;
    private FaceDetection faceDetection;

    public PreProcessorFactory(Context context, int N) {
        this.context = context;
        this.N = N;
        this.faceDetection = new FaceDetection(context);
        Resources res = context.getResources();
        SharedPreferences sharedPref = PreferenceManager.getDefaultSharedPreferences((context.getApplicationContext()));
        commandFactory = new CommandFactory();
        commandFactory.addCommand(res.getString(R.string.grayscale), new GrayScale());
        commandFactory.addCommand(res.getString(R.string.eyeAlignment), new EyeAlignment());
        commandFactory.addCommand(res.getString(R.string.crop), new Crop());
        commandFactory.addCommand(res.getString(R.string.gammaCorrection), new GammaCorrection(Float.valueOf(sharedPref.getString("key_gamma", res.getString(R.string.gamma)))));
        String[] sigmasString = sharedPref.getString("key_sigmas", res.getString(R.string.sigmas)).split(",");
        if(sigmasString.length != 2){
            sigmasString = res.getString(R.string.sigmas).split(",");
        }
        double[] sigmas = new double[3];
        for(int i=0; i<2; i++){
            sigmas[i] = Double.parseDouble(sigmasString[i]);
        }
        commandFactory.addCommand(res.getString(R.string.doG), new DifferenceOfGaussian(sigmas));
        commandFactory.addCommand(res.getString(R.string.masking), new Masking());
        commandFactory.addCommand(res.getString(R.string.histogrammEqualization), new HistogrammEqualization());
        commandFactory.addCommand(res.getString(R.string.resize), new Resize());
        commandFactory.addCommand(res.getString(R.string.lbp), new LocalBinaryPattern());
    }

    public PreProcessorFactory(Context context){
        this.context = context;
        this.faceDetection = new FaceDetection(context);
        commandFactory = new CommandFactory();
        Resources res = context.getResources();
        commandFactory.addCommand(res.getString(R.string.crop), new Crop());
    }

    public Mat getCroppedImage(Mat img){
        preProcessor = new PreProcessor(context, faceDetection);
        images = new ArrayList<Mat>();
        images.add(img);
        preProcessor.setImages(images);
        preProcessor.setFaces();
        List<Mat> result = null;
        try {
            Resources res = context.getResources();
            preProcessor = commandFactory.executeCommand(res.getString(R.string.crop), preProcessor);
            preProcessor.setEyes();
            Eyes[] eyes = preProcessor.getEyes();
            if (eyes == null || eyes[0] == null){
                return null;
            }
            result = preProcessor.getImages();
        } catch (NullPointerException e){
            e.printStackTrace();
        }
        if((result != null) && (result.size() == 1)){
            return result.get(0);
        } else {
            return null;
        }
    }

    public List<Mat> getProcessedImage(Mat img) throws NullPointerException {
        preProcessor = new PreProcessor(context, N, faceDetection);
        images = new ArrayList<Mat>();
        images.add(img);
        preProcessor.setImages(images);
        SharedPreferences sharedPref = PreferenceManager.getDefaultSharedPreferences(context);
        Set<String> standard_pre_set = sharedPref.getStringSet("key_standard_pre", null);
        Set<String> brightness_set = sharedPref.getStringSet("key_brightness", null);
        Set<String> contours_set = sharedPref.getStringSet("key_contours", null);
        Set<String> contrast_set = sharedPref.getStringSet("key_contrast", null);
        Set<String> standard_post_set = sharedPref.getStringSet("key_standard_post", null);
        ArrayList<String> preprocessings = new ArrayList<String>();
        ArrayList<String> standard_pre;
        ArrayList<String> brightness;
        ArrayList<String> contours;
        ArrayList<String> contrast;
        ArrayList<String> standard_post;
        if(standard_pre_set != null) {
            standard_pre = new ArrayList<String>(standard_pre_set);
            Collections.sort(standard_pre);
            preprocessings.addAll(standard_pre);
        }
        if(brightness_set != null) {
            brightness = new ArrayList<String>(brightness_set);
            Collections.sort(brightness);
            preprocessings.addAll(brightness);
        }
        if(contours_set != null){
            contours = new ArrayList<String>(contours_set);
            Collections.sort(contours);
            preprocessings.addAll(contours);
        }
        if(contrast_set != null){
            contrast = new ArrayList<String>(contrast_set);
            Collections.sort(contrast);
            preprocessings.addAll(contrast);
        }
        if(standard_post_set != null) {
            standard_post = new ArrayList<String>(standard_post_set);
            Collections.sort(standard_post);
            preprocessings.addAll(standard_post);
        }
        try {
            preProcessor.setFaces();

            for (String name : preprocessings){
                preProcessor = commandFactory.executeCommand(name, preProcessor);
            }

        } catch (NullPointerException e){
            e.printStackTrace();
            return null;
        }
        return preProcessor.getImages();
    }

    public Rect[] getFacesForRecognition() {
        if(preProcessor != null){
            return preProcessor.getFaces();
        } else {
            return null;
        }
    }

    public int getAngleForRecognition(){
        return preProcessor.getAngle();
    }
}
