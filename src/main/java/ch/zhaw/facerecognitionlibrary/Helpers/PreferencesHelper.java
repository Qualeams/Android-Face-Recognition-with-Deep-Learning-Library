package ch.zhaw.facerecognitionlibrary.Helpers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import ch.zhaw.facerecognitionlibrary.FaceRecognitionLibrary;
import ch.zhaw.facerecognitionlibrary.R;

/**
 * Created by sladomic on 05.11.16.
 */

public class PreferencesHelper {
    public enum Usage {RECOGNITION, DETECTION};

    public static String getClassificationMethod(){
        return FaceRecognitionLibrary.sharedPreferences.getString("key_classification_method", FaceRecognitionLibrary.resources.getString(R.string.eigenfaces));
    }

    public static boolean getClassificationMethodTFCaffe(){
        return FaceRecognitionLibrary.sharedPreferences.getBoolean("key_classificationMethodTFCaffe", true);
    }

    public static float getGamma(){
        return Float.valueOf(FaceRecognitionLibrary.sharedPreferences.getString("key_gamma", FaceRecognitionLibrary.resources.getString(R.string.gamma)));
    }

    public static  double[] getSigmas(){
        String[] sigmasString = FaceRecognitionLibrary.sharedPreferences.getString("key_sigmas", FaceRecognitionLibrary.resources.getString(R.string.sigmas)).split(",");
        if(sigmasString.length != 2){
            sigmasString = FaceRecognitionLibrary.resources.getString(R.string.sigmas).split(",");
        }
        double[] sigmas = new double[3];
        for(int i=0; i<2; i++){
            sigmas[i] = Double.parseDouble(sigmasString[i]);
        }
        return sigmas;
    }

    public static boolean getEyeDetectionEnabled(){
        return FaceRecognitionLibrary.sharedPreferences.getBoolean("key_eye_detection", true);
    }

    public static List<String> getStandardPreprocessing(Usage usage){
        if (usage == Usage.RECOGNITION){
            return getPreferenceList("key_standard_pre");
        } else if (usage == Usage.DETECTION){
            return getPreferenceList("key_detection_standard_pre");
        } else {
            return new ArrayList<>();
        }
    }

    public static List<String> getBrightnessPreprocessing(Usage usage){
        if (usage == Usage.RECOGNITION){
            return getPreferenceList("key_brightness");
        } else if (usage == Usage.DETECTION){
            return getPreferenceList("key_detection_brightness");
        } else {
            return new ArrayList<>();
        }
    }

    public static List<String> getContoursPreprocessing(Usage usage){
        if (usage == Usage.RECOGNITION){
            return getPreferenceList("key_contours");
        } else if (usage == Usage.DETECTION){
            return getPreferenceList("key_detection_contours");
        } else {
            return new ArrayList<>();
        }
    }

    public static List<String> getContrastPreprocessing(Usage usage){
        if (usage == Usage.RECOGNITION){
            return getPreferenceList("key_contrast");
        } else if (usage == Usage.DETECTION){
            return getPreferenceList("key_detection_contrast");
        } else {
            return new ArrayList<>();
        }
    }

    public static List<String> getStandardPostrocessing(Usage usage){
        if (usage == Usage.RECOGNITION){
            return getPreferenceList("key_standard_post");
        } else if (usage == Usage.DETECTION){
            return getPreferenceList("key_detection_standard_post");
        } else {
            return new ArrayList<>();
        }
    }

    private static List<String> getPreferenceList(String key){
        Set<String> set = FaceRecognitionLibrary.sharedPreferences.getStringSet(key, null);
        ArrayList<String> list;
        if(set != null) {
            list = new ArrayList<String>(set);
            Collections.sort(list);
            return list;
        } else {
            return new ArrayList<>();
        }
    }

    public static String getCaffeModelFile(){
        return FaceRecognitionLibrary.sharedPreferences.getString("key_modelFileCaffe", FaceRecognitionLibrary.resources.getString(R.string.modelFileCaffe));
    }

    public static String getCaffeWeightsFile(){
        return FaceRecognitionLibrary.sharedPreferences.getString("key_weightsFileCaffe", FaceRecognitionLibrary.resources.getString(R.string.weightsFileCaffe));
    }

    public static String getCaffeOutputLayer(){
        return FaceRecognitionLibrary.sharedPreferences.getString("key_outputLayerCaffe", FaceRecognitionLibrary.resources.getString(R.string.weightsFileCaffe));
    }

    public static float[] getCaffeMeanValues(){
        String[] meanValuesString = FaceRecognitionLibrary.sharedPreferences.getString("key_meanValuesCaffe", FaceRecognitionLibrary.resources.getString(R.string.meanValuesCaffe)).split(",");
        if(meanValuesString.length != 3){
            meanValuesString = FaceRecognitionLibrary.resources.getString(R.string.meanValuesCaffe).split(",");
        }
        float[] meanValues = new float[3];
        for(int i=0; i<3; i++){
            meanValues[i] = Float.parseFloat(meanValuesString[i]);
        }
        return meanValues;
    }

    public static String getSvmTrainOptions(){
        return FaceRecognitionLibrary.sharedPreferences.getString("key_svmTrainOptions", "-t 0 ");
    }

    public static int getK(){
        return Integer.valueOf(FaceRecognitionLibrary.sharedPreferences.getString("key_K", "20"));
    }

    public static int getN(){
        return Integer.valueOf(FaceRecognitionLibrary.sharedPreferences.getString("key_N", "25"));
    }

    public static int getTensorFlowInputSize(){
        return Integer.valueOf(FaceRecognitionLibrary.sharedPreferences.getString("key_inputSize", "224"));
    }

    public static int getTensorFlowImageMean(){
        return Integer.valueOf(FaceRecognitionLibrary.sharedPreferences.getString("key_imageMean", "128"));
    }

    public static int getTensorFlowOutputSize(){
        return Integer.valueOf(FaceRecognitionLibrary.sharedPreferences.getString("key_outputSize", "1024"));
    }

    public static String getTensorFlowInputLayer(){
        return FaceRecognitionLibrary.sharedPreferences.getString("key_inputLayer", "input");
    }

    public static String getTensorFlowOutputLayer(){
        return FaceRecognitionLibrary.sharedPreferences.getString("key_outputLayer", "avgpool0");
    }

    public static String getTensorFlowModelFile(){
        return FaceRecognitionLibrary.sharedPreferences.getString("key_modelFileTensorFlow", "tensorflow_inception_graph.pb");
    }

    public static float getPCAThreshold(){
        return Float.valueOf(FaceRecognitionLibrary.sharedPreferences.getString("key_pca_threshold", "0.98f"));
    }

    public static String getFaceCascadeFile(){
        return FaceRecognitionLibrary.sharedPreferences.getString("key_face_cascade_file", FaceRecognitionLibrary.resources.getString(R.string.haarcascade_alt2));
    }

    public static String getLefteyeCascadeFile(){
        return FaceRecognitionLibrary.sharedPreferences.getString("key_lefteye_cascade_file", FaceRecognitionLibrary.resources.getString(R.string.haarcascade_lefteye));
    }

    public static String getRighteyeCascadeFile(){
        return FaceRecognitionLibrary.sharedPreferences.getString("key_righteye_cascade_file", FaceRecognitionLibrary.resources.getString(R.string.haarcascade_righteye));
    }

    public static double getDetectionScaleFactor(){
        return Double.parseDouble(FaceRecognitionLibrary.sharedPreferences.getString("key_scaleFactor", "1.1"));
    }

    public static int getDetectionMinNeighbors(){
        return Integer.parseInt(FaceRecognitionLibrary.sharedPreferences.getString("key_minNeighbors", "3"));
    }

    public static int getDetectionFlags(){
        return Integer.parseInt(FaceRecognitionLibrary.sharedPreferences.getString("key_flags", "2"));
    }
}
