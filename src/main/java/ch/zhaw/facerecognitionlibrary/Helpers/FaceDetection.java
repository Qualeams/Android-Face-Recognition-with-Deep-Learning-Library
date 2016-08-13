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

package ch.zhaw.facerecognitionlibrary.Helpers;

import android.content.Context;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import ch.zhaw.facerecognitionlibrary.R;

public class FaceDetection {
    private Mat img;
    private int angle;
    private static final String TAG = "Face Detection";
    private CascadeClassifier faceDetector;
    private CascadeClassifier leftEyeDetector;
    private CascadeClassifier rightEyeDetector;

    public FaceDetection(Context context) {
        // load cascade file from application resources
        File cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);

        InputStream is = context.getResources().openRawResource(R.raw.haarcascade_frontalface_default);
        File mCascadeFile = new File(cascadeDir, "haarcascade_frontalface_default.xml");
        String path = getClassifierPath(mCascadeFile,is);
        faceDetector = new CascadeClassifier(path);
        if (faceDetector.empty()) {
            Log.e(TAG, "Failed to load face classifier");
            faceDetector = null;
        }

        is = context.getResources().openRawResource(R.raw.haarcascade_lefteye_2splits);
        mCascadeFile = new File(cascadeDir, "haarcascade_lefteye_2splits.xml");
        path = getClassifierPath(mCascadeFile, is);
        leftEyeDetector = new CascadeClassifier(path);
        if (leftEyeDetector.empty()) {
            Log.e(TAG, "Failed to load leftEye classifier");
            leftEyeDetector = null;
        }

        is = context.getResources().openRawResource(R.raw.haarcascade_righteye_2splits);
        mCascadeFile = new File(cascadeDir, "haarcascade_righteye_2splits.xml");
        path = getClassifierPath(mCascadeFile, is);
        rightEyeDetector = new CascadeClassifier(path);
        if (rightEyeDetector.empty()) {
            Log.e(TAG, "Failed to load rightEye classifier");
            rightEyeDetector = null;
        }

        cascadeDir.delete();
    }

    public Rect[] getFaces(Mat img) {
        MatOfRect faces = new MatOfRect();
        List<Rect> facesList = null;
        float mRelativeFaceSize = 0.2f;
        int mAbsoluteFaceSize = 0;
        if(faceDetector !=null){
            // If no face detected --> rotate the picture 90° and try again
            angle = 0;
            for(int i=1; i<=4; i++){
                int height = img.rows();
                if (Math.round(height * mRelativeFaceSize) > 0) {
                    mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
                }
                faceDetector.detectMultiScale(img, faces, 1.1 ,2, 2, new Size(mAbsoluteFaceSize,mAbsoluteFaceSize), new Size());
                // Rotate by 90°
                if(faces.empty()){
                    angle = 90 * i;
                    MatOperation.rotate_90n(img, 90);
                } else {
                    facesList = faces.toList();
                    // Check that each found face rectangle fits in the image, if not, remove it
                    for (Rect face : facesList){
                        if(!(0 <= face.x && 0 <= face.width && face.x + face.width <= img.cols() && 0 <= face.y && 0 <= face.height && face.y + face.height <= img.rows())){
                            facesList.remove(face);
                        }
                    }
                    if(!(facesList.size()>0)){
                        return null;
                    }
                    // Faces found with the current image rotation
                    this.img = img;
                    break;
                }
            }

        } else {
            Log.e(TAG, "Detection method is not selected!");
        }
        if(facesList != null){
            return (Rect[])facesList.toArray();
        } else {
            return null;
        }
    }

    public Eyes getEyes(Mat img){
        double halfWidth = img.cols() / 2;
        double height = img.rows();
        double[] values = new double[4];
        values[0] = 0;
        values[1] = 0;
        values[2] = halfWidth;
        values[3] = height;
        Rect rightHalf = new Rect(values);
        values[0] = halfWidth;
        Rect leftHalf = new Rect(values);
        MatOfRect rightEyes = new MatOfRect();
        MatOfRect leftEyes = new MatOfRect();

        Mat rightHalfImg = img.submat(rightHalf);
        rightEyeDetector.detectMultiScale(rightHalfImg, rightEyes);
        Mat leftHalfImg = img.submat(leftHalf);
        leftEyeDetector.detectMultiScale(leftHalfImg, leftEyes);

        if (rightEyes.empty() || leftEyes.empty() || rightEyes.toArray().length > 1 || leftEyes.toArray().length > 1){
            return null;
        }

        Rect rightEye = rightEyes.toArray()[0];
        Rect leftEye = leftEyes.toArray()[0];

        MatOfFloat rightPoint = new MatOfFloat(rightEye.x + rightEye.width / 2, rightEye.y + rightEye.height / 2);
        MatOfFloat leftPoint = new MatOfFloat(img.cols() / 2 + leftEye.x + leftEye.width / 2, leftEye.y + leftEye.height / 2);

        MatOfFloat diff = new MatOfFloat();
        Core.subtract(leftPoint, rightPoint, diff);
        double angle = Core.fastAtan2(diff.toArray()[1], diff.toArray()[0]);
        double dist = Core.norm(leftPoint, rightPoint, Core.NORM_L2);
        Eyes eyes = new Eyes(dist, rightPoint, leftPoint, angle);
        return eyes;
    }

    private String getClassifierPath(File mCascadeFile, InputStream is){
        try {
            FileOutputStream os = new FileOutputStream(mCascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();
        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
        }
        return mCascadeFile.getAbsolutePath();
    }

    public Mat getImg() {
        return img;
    }

    public int getAngle() {
        return angle;
    }
}
