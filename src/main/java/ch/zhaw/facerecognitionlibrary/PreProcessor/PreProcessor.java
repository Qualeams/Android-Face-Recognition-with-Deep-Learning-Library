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

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;

import java.util.List;

import ch.zhaw.facerecognitionlibrary.Helpers.Eyes;
import ch.zhaw.facerecognitionlibrary.Helpers.FaceDetection;

public class PreProcessor {
    private Context context;
    private int angle;
    private int N;
    private Mat img;
    private List<Mat> images;
    private Rect[] faces;
    private Eyes[] eyes;
    private FaceDetection faceDetection;

    public PreProcessor(Context context, int N, FaceDetection faceDetection){
        this.context = context;
        this.N = N;
        this.faceDetection = faceDetection;
    }

    public PreProcessor(Context context, FaceDetection faceDetection){
        this.context = context;
        this.faceDetection = faceDetection;
    }

    public void setFaces() {
        List<Mat> images = getImages();
        faces = faceDetection.getFaces(images.get(0));
        angle = faceDetection.getAngle();
        // Change also the rotation of the image
        images.remove(0);
        images.add(faceDetection.getImg());
        setImages(images);
    }

    public void setEyes() {
        List<Mat> images = getImages();
        eyes = new Eyes[images.size()];
        for (int i=0; i<images.size(); i++){
            Mat img = images.get(i);
            normalize0255(img);
            eyes[i] = faceDetection.getEyes(img);
        }
    }

    public int getN() {
        return N;
    }

    public Eyes[] getEyes() {
        return eyes;
    }

    public Rect[] getFaces() {
        return faces;
    }

    public int getAngle() { return angle; }

    public Mat getImg() {
        return img;
    }

    public void setImages(List<Mat> images) {
        this.images = images;
    }

    public List<Mat> getImages() {
        return images;
    }

    public void setImg(Mat img) {
        this.img = img;
    }

    public void normalize01(Mat norm){
        Core.normalize(norm, norm, 0.0, 1.0, Core.NORM_MINMAX, CvType.CV_64FC1);
        Core.MinMaxLocResult minmax = Core.minMaxLoc(norm);
        Scalar min = new Scalar(minmax.minVal);
        Core.subtract(norm, min, norm);
        minmax = Core.minMaxLoc(norm);
        Scalar max = new Scalar(minmax.maxVal);
        Core.divide(norm, max, norm);
    }

    public void normalize0255(Mat norm){
        Core.normalize(norm, norm, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC1);
    }
}
