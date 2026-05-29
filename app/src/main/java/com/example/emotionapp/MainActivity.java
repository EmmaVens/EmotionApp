package com.example.emotionapp;
import java.util.List;
import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Bundle;
import android.util.Size;
import android.widget.FrameLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.OptIn;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private PreviewView previewView;
    private TextView statusText;
    private TextView emotionText;
    private TextView topResultsText;
    private FrameLayout cameraFrame;
    private FaceOverlayView overlayView;

    private Interpreter interpreter;
    private FaceDetector faceDetector;
    private ExecutorService cameraExecutor;

    private boolean isProcessing = false;
    private long lastAnalysisTime = 0;

    private int modelWidth = 48;
    private int modelHeight = 48;
    private int modelChannels = 1;
    private DataType inputDataType = DataType.FLOAT32;

    private static final int CAMERA_PERMISSION_CODE = 10;

    private final String[] emotionLabels = {
            "Злость",
            "Отвращение",
            "Страх",
            "Радость",
            "Грусть",
            "Удивление",
            "Нейтрально"
    };

    private static class EmotionResult {
        String label;
        float score;

        EmotionResult(String label, float score) {
            this.label = label;
            this.score = score;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.previewView);
        statusText = findViewById(R.id.statusText);
        emotionText = findViewById(R.id.emotionText);
        topResultsText = findViewById(R.id.topResultsText);
        cameraFrame = findViewById(R.id.cameraFrame);

        overlayView = new FaceOverlayView(this);
        cameraFrame.addView(
                overlayView,
                new FrameLayout.LayoutParams(
                        FrameLayout.LayoutParams.MATCH_PARENT,
                        FrameLayout.LayoutParams.MATCH_PARENT
                )
        );

        cameraExecutor = Executors.newSingleThreadExecutor();

        try {
            loadTfliteModel();
            setupFaceDetector();
        } catch (Exception e) {
            Toast.makeText(this, "Ошибка загрузки модели: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(
                    this,
                    new String[]{Manifest.permission.CAMERA},
                    CAMERA_PERMISSION_CODE
            );
        }
    }

    private void loadTfliteModel() throws Exception {
        interpreter = new Interpreter(loadModelFile("emotion_model.tflite"));

        int[] inputShape = interpreter.getInputTensor(0).shape();

        if (inputShape.length == 4) {
            modelHeight = inputShape[1];
            modelWidth = inputShape[2];
            modelChannels = inputShape[3];
        }

        inputDataType = interpreter.getInputTensor(0).dataType();
    }

    private MappedByteBuffer loadModelFile(String fileName) throws Exception {
        FileInputStream inputStream = new FileInputStream(getAssets().openFd(fileName).getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();

        long startOffset = getAssets().openFd(fileName).getStartOffset();
        long declaredLength = getAssets().openFd(fileName).getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void setupFaceDetector() {
        FaceDetectorOptions options =
                new FaceDetectorOptions.Builder()
                        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                        .setMinFaceSize(0.15f)
                        .enableTracking()
                        .build();

        faceDetector = FaceDetection.getClient(options);
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                Preview preview = new Preview.Builder().build();

                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                ImageAnalysis imageAnalysis =
                        new ImageAnalysis.Builder()
                                .setTargetResolution(new Size(480, 640))
                                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                                .build();

                imageAnalysis.setAnalyzer(cameraExecutor, this::analyzeFrame);

                CameraSelector cameraSelector =
                        new CameraSelector.Builder()
                                .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                                .build();

                cameraProvider.unbindAll();

                cameraProvider.bindToLifecycle(
                        this,
                        cameraSelector,
                        preview,
                        imageAnalysis
                );

            } catch (Exception e) {
                Toast.makeText(this, "Ошибка запуска камеры", Toast.LENGTH_SHORT).show();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    @OptIn(markerClass = ExperimentalGetImage.class)
    private void analyzeFrame(@NonNull ImageProxy imageProxy) {
        long now = System.currentTimeMillis();

        if (isProcessing || now - lastAnalysisTime < 600) {
            imageProxy.close();
            return;
        }

        isProcessing = true;
        lastAnalysisTime = now;

        Image mediaImage = imageProxy.getImage();

        if (mediaImage == null) {
            isProcessing = false;
            imageProxy.close();
            return;
        }

        Bitmap bitmap = imageProxyToBitmap(imageProxy);
        Bitmap rotatedBitmap = rotateBitmap(bitmap, imageProxy.getImageInfo().getRotationDegrees());

        InputImage inputImage = InputImage.fromBitmap(rotatedBitmap, 0);

        faceDetector.process(inputImage)
                .addOnSuccessListener(faces -> {
                    if (faces.isEmpty()) {
                        runOnUiThread(() -> {
                            statusText.setText("Лицо не найдено");
                            emotionText.setText("Эмоция: —");
                            topResultsText.setText("Посмотрите прямо во фронтальную камеру");
                            overlayView.clear();
                        });
                        return;
                    }

                    Face face = getLargestFace(faces);

                    if (face == null) return;

                    Rect faceRect = face.getBoundingBox();

                    Bitmap faceBitmap = cropFace(rotatedBitmap, faceRect);

                    if (faceBitmap == null) {
                        runOnUiThread(() -> statusText.setText("Лицо слишком близко к краю кадра"));
                        return;
                    }

                    ArrayList<EmotionResult> results = runEmotionModel(faceBitmap);

                    runOnUiThread(() -> {
                        if (results.isEmpty()) {
                            emotionText.setText("Эмоция: —");
                            topResultsText.setText("Модель не вернула результат");
                        } else {
                            EmotionResult best = results.get(0);

                            emotionText.setText(
                                    "Эмоция: " + best.label + " " +
                                            String.format(Locale.getDefault(), "%.1f%%", best.score * 100)
                            );

                            topResultsText.setText(formatTopResults(results));
                            statusText.setText("Лицо найдено, анализ выполнен");
                        }

                        overlayView.setFaceRect(
                                faceRect,
                                rotatedBitmap.getWidth(),
                                rotatedBitmap.getHeight()
                        );
                    });
                })
                .addOnFailureListener(e -> runOnUiThread(() -> {
                    statusText.setText("Ошибка анализа лица");
                    emotionText.setText("Эмоция: —");
                }))
                .addOnCompleteListener(task -> {
                    isProcessing = false;
                    imageProxy.close();
                });
    }

    private Face getLargestFace(List<Face> faces) {
        Face largest = null;
        int largestArea = 0;

        for (Face face : faces) {
            Rect rect = face.getBoundingBox();
            int area = rect.width() * rect.height();

            if (area > largestArea) {
                largestArea = area;
                largest = face;
            }
        }

        return largest;
    }

    private Bitmap cropFace(Bitmap source, Rect rect) {
        int paddingX = rect.width() / 5;
        int paddingY = rect.height() / 5;

        int left = Math.max(0, rect.left - paddingX);
        int top = Math.max(0, rect.top - paddingY);
        int right = Math.min(source.getWidth(), rect.right + paddingX);
        int bottom = Math.min(source.getHeight(), rect.bottom + paddingY);

        int width = right - left;
        int height = bottom - top;

        if (width <= 10 || height <= 10) {
            return null;
        }

        return Bitmap.createBitmap(source, left, top, width, height);
    }

    private ArrayList<EmotionResult> runEmotionModel(Bitmap faceBitmap) {
        ArrayList<EmotionResult> results = new ArrayList<>();

        if (interpreter == null) return results;

        Bitmap resized = Bitmap.createScaledBitmap(faceBitmap, modelWidth, modelHeight, true);

        ByteBuffer inputBuffer = createInputBuffer(resized);

        int[] outputShape = interpreter.getOutputTensor(0).shape();
        int classesCount = outputShape[outputShape.length - 1];

        float[][] output = new float[1][classesCount];

        interpreter.run(inputBuffer, output);

        for (int i = 0; i < classesCount; i++) {
            String label;

            if (i < emotionLabels.length) {
                label = emotionLabels[i];
            } else {
                label = "Класс " + i;
            }

            results.add(new EmotionResult(label, output[0][i]));
        }

        Collections.sort(results, (a, b) -> Float.compare(b.score, a.score));

        return results;
    }

    private ByteBuffer createInputBuffer(Bitmap bitmap) {
        int bytesPerValue = inputDataType == DataType.FLOAT32 ? 4 : 1;

        ByteBuffer buffer = ByteBuffer.allocateDirect(
                1 * modelWidth * modelHeight * modelChannels * bytesPerValue
        );

        buffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[modelWidth * modelHeight];
        bitmap.getPixels(pixels, 0, modelWidth, 0, 0, modelWidth, modelHeight);

        for (int y = 0; y < modelHeight; y++) {
            for (int x = 0; x < modelWidth; x++) {
                int pixel = pixels[y * modelWidth + x];

                int r = Color.red(pixel);
                int g = Color.green(pixel);
                int b = Color.blue(pixel);

                if (modelChannels == 1) {
                    int gray = (r + g + b) / 3;

                    if (inputDataType == DataType.FLOAT32) {
                        buffer.putFloat(gray / 255.0f);
                    } else {
                        buffer.put((byte) gray);
                    }
                } else {
                    if (inputDataType == DataType.FLOAT32) {
                        buffer.putFloat(r / 255.0f);
                        buffer.putFloat(g / 255.0f);
                        buffer.putFloat(b / 255.0f);
                    } else {
                        buffer.put((byte) r);
                        buffer.put((byte) g);
                        buffer.put((byte) b);
                    }
                }
            }
        }

        buffer.rewind();
        return buffer;
    }

    private String formatTopResults(ArrayList<EmotionResult> results) {
        StringBuilder builder = new StringBuilder();

        int limit = Math.min(3, results.size());

        for (int i = 0; i < limit; i++) {
            EmotionResult r = results.get(i);

            builder.append(i + 1)
                    .append(". ")
                    .append(r.label)
                    .append(" — ")
                    .append(String.format(Locale.getDefault(), "%.1f%%", r.score * 100));

            if (i < limit - 1) {
                builder.append("\n");
            }
        }

        return builder.toString();
    }

    private Bitmap imageProxyToBitmap(ImageProxy imageProxy) {
        Image image = imageProxy.getImage();

        if (image == null) {
            return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888);
        }

        Image.Plane[] planes = image.getPlanes();

        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];

        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(
                nv21,
                ImageFormat.NV21,
                image.getWidth(),
                image.getHeight(),
                null
        );

        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();

        yuvImage.compressToJpeg(
                new Rect(0, 0, image.getWidth(), image.getHeight()),
                90,
                outputStream
        );

        byte[] imageBytes = outputStream.toByteArray();

        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    private Bitmap rotateBitmap(Bitmap bitmap, int rotationDegrees) {
        if (rotationDegrees == 0) return bitmap;

        Matrix matrix = new Matrix();
        matrix.postRotate(rotationDegrees);

        return Bitmap.createBitmap(
                bitmap,
                0,
                0,
                bitmap.getWidth(),
                bitmap.getHeight(),
                matrix,
                true
        );
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        if (cameraExecutor != null) {
            cameraExecutor.shutdown();
        }

        if (interpreter != null) {
            interpreter.close();
        }

        if (faceDetector != null) {
            faceDetector.close();
        }
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode,
            @NonNull String[] permissions,
            @NonNull int[] grantResults
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.length > 0
                    && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera();
            } else {
                Toast.makeText(this, "Разрешение на камеру не выдано", Toast.LENGTH_LONG).show();
            }
        }
    }

    public static class FaceOverlayView extends android.view.View {

        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);

        private Rect faceRect;
        private int imageWidth = 1;
        private int imageHeight = 1;

        public FaceOverlayView(android.content.Context context) {
            super(context);

            paint.setColor(Color.rgb(34, 197, 94));
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(7);
        }

        public void setFaceRect(Rect rect, int imageWidth, int imageHeight) {
            this.faceRect = new Rect(rect);
            this.imageWidth = imageWidth;
            this.imageHeight = imageHeight;
            invalidate();
        }

        public void clear() {
            faceRect = null;
            invalidate();
        }

        @Override
        protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);

            if (faceRect == null) return;

            float viewWidth = getWidth();
            float viewHeight = getHeight();

            float scaleX = viewWidth / imageWidth;
            float scaleY = viewHeight / imageHeight;

            float left = faceRect.left * scaleX;
            float top = faceRect.top * scaleY;
            float right = faceRect.right * scaleX;
            float bottom = faceRect.bottom * scaleY;

            canvas.drawRoundRect(left, top, right, bottom, 28, 28, paint);
        }
    }
}