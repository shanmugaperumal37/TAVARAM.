package com.example.tavaram;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    private static final int PICK_IMAGE = 1;
    private ImageView selectedImage;
    private TextView diseaseResult, diseaseNote;
    private Interpreter tflite;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        selectedImage = findViewById(R.id.selectedImage);
        diseaseResult = findViewById(R.id.diseaseResult);
        diseaseNote = findViewById(R.id.diseaseNote);
        Button selectImageButton = findViewById(R.id.selectImageButton);

        // Initialize TensorFlow Lite interpreter
        try {
            tflite = new Interpreter(loadModelFile());
        } catch (Exception e) {
            e.printStackTrace();
        }

        selectImageButton.setOnClickListener(v -> openGallery());
    }

    private void openGallery() {
        Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(galleryIntent, PICK_IMAGE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == PICK_IMAGE && resultCode == RESULT_OK && data != null) {
            // Get the selected image from gallery
            Bitmap bitmap = null;
            try {
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), data.getData());
                selectedImage.setImageBitmap(bitmap);
                runModel(bitmap); // Run model inference after selecting the image
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private void runModel(Bitmap bitmap) {
        try {
            // Convert bitmap to ByteBuffer for TensorFlow Lite
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3);
            byteBuffer.order(ByteOrder.nativeOrder());
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
            int[] intValues = new int[224 * 224];
            resizedBitmap.getPixels(intValues, 0, 224, 0, 0, 224, 224);
            for (int pixelValue : intValues) {
                byteBuffer.putFloat(((pixelValue >> 16) & 0xFF) * (1.f / 255.f));
                byteBuffer.putFloat(((pixelValue >> 8) & 0xFF) * (1.f / 255.f));
                byteBuffer.putFloat((pixelValue & 0xFF) * (1.f / 255.f));
            }

            // Create TensorBuffer for input
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            inputFeature0.loadBuffer(byteBuffer);

            // Run inference
            float[][] output = new float[1][3];  // Example output for 3 classes
            tflite.run(inputFeature0.getBuffer(), output);

            // Get the result and display the corresponding disease
            displayDiseaseNotes(output[0]);

        } catch (Exception e) {
            Log.e("TFLite", "Error during inference", e);
        }
    }

    private void displayDiseaseNotes(float[] result) {
        String disease = "Corn Common Rust";  // Default disease
        String diseaseInfoEN = "Caused by the fungus Puccinia sorghi...";
        String diseaseInfoTA = "Puccinia sorghi...";

        // Assuming the model output indicates probabilities for 3 diseases
        if (result[0] > 0.5) {
            disease = "Corn Common Rust";
            diseaseInfoEN = "Caused by the fungus Puccinia sorghi...";
            diseaseInfoTA = "Puccinia sorghi...";
        } else if (result[1] > 0.5) {
            disease = "Tomato Mosaic Virus";
            diseaseInfoEN = "Caused by the Tomato mosaic virus...";
            diseaseInfoTA = "Tomato Mosaic Virus...";
        } else if (result[2] > 0.5) {
            disease = "Apple Black Rot";
            diseaseInfoEN = "Caused by the fungus Botryosphaeria obtusa...";
            diseaseInfoTA = "Apple Black Rot...";
        }

        diseaseResult.setText("Detection Result: " + disease);
        diseaseNote.setText("English:\n" + diseaseInfoEN + "\n\nTamil:\n" + diseaseInfoTA);
    }

    // Load TensorFlow Lite model from assets folder
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("model_unquant.tflite");
        FileInputStream inputStream = fileDescriptor.createInputStream();
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}