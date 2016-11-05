package ch.zhaw.facerecognitionlibrary;

import android.app.Application;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.AssetManager;
import android.content.res.Resources;
import android.preference.PreferenceManager;

import ch.zhaw.facerecognitionlibrary.Helpers.PreferencesHelper;

/**
 * Created by sladomic on 05.11.16.
 */

public class FaceRecognitionLibrary extends Application {
    public static Resources resources;
    public static Context context;
    public static AssetManager assets;
    public static SharedPreferences sharedPreferences;

    @Override
    public void onCreate() {
        super.onCreate();

        resources = getResources();
        context = this;
        assets = getAssets();
        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this);
    }
}
