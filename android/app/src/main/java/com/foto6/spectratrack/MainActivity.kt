package com.foto6.spectratrack

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.io.File
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var overlay: HudOverlayView
    private lateinit var errorText: TextView
    private lateinit var importModelButton: Button
    private val executor = Executors.newSingleThreadExecutor()
    private val tracker = LiteTracker()
    private var detector: OnnxDetector? = null
    private var smoothedFps = 0f
    private var lastFrameNs = 0L
    private var cameraStarted = false

    private val permissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCameraIfReady() else showError("Camera permission is required.", showImport = detector == null)
    }

    private val modelPicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) importModel(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.previewView)
        overlay = findViewById(R.id.hudOverlay)
        errorText = findViewById(R.id.errorText)
        importModelButton = findViewById(R.id.importModelButton)
        importModelButton.setOnClickListener {
            modelPicker.launch(arrayOf("application/octet-stream", "application/x-onnx", "*/*"))
        }

        loadDetectorOrPrompt()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCameraIfReady()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun modelFile(): File {
        val dir = File(filesDir, "models").apply { mkdirs() }
        return File(dir, "detector.onnx")
    }

    private fun loadDetectorOrPrompt() {
        detector?.close()
        detector = null
        try {
            detector = OnnxDetector.fromAssetIfPresent(this)
            if (detector == null && modelFile().exists()) detector = OnnxDetector.fromFile(this, modelFile())
            if (detector == null) {
                showError(
                    "No detector model installed. Tap IMPORT MODEL and choose a fixed-size YOLO11/YOLOv8 COCO ONNX file (recommended: 640x640).",
                    showImport = true,
                )
            } else {
                clearError()
            }
        } catch (t: Throwable) {
            showError("Model load failed: ${t.message}", showImport = true)
        }
    }

    private fun importModel(uri: Uri) {
        try {
            val target = modelFile()
            contentResolver.openInputStream(uri).use { input ->
                requireNotNull(input) { "Cannot open selected file" }
                target.outputStream().use { output -> input.copyTo(output) }
            }
            loadDetectorOrPrompt()
            if (detector != null) startCameraIfReady()
        } catch (t: Throwable) {
            showError("Model import failed: ${t.message}", showImport = true)
        }
    }

    private fun startCameraIfReady() {
        if (detector == null || cameraStarted) return
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return
        startCamera()
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
                val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                    .build()
                analysis.setAnalyzer(executor) { image -> analyzeFrame(image) }
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
                cameraStarted = true
                clearError()
            } catch (t: Throwable) {
                showError("Camera start failed: ${t.message}", showImport = detector == null)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(image: ImageProxy) {
        try {
            val activeDetector = detector ?: return
            val now = SystemClock.elapsedRealtimeNanos()
            if (lastFrameNs != 0L) {
                val fps = 1_000_000_000f / (now - lastFrameNs).coerceAtLeast(1L)
                smoothedFps = if (smoothedFps == 0f) fps else smoothedFps * 0.88f + fps * 0.12f
            }
            lastFrameNs = now

            var bitmap = image.toBitmap()
            val rotation = image.imageInfo.rotationDegrees
            if (rotation != 0) {
                val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
                val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
                if (rotated !== bitmap) bitmap.recycle()
                bitmap = rotated
            }

            val start = SystemClock.elapsedRealtimeNanos()
            val detections = activeDetector.detect(bitmap)
            val tracked = tracker.update(detections)
            val inferMs = (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000f
            val selected = overlay.selectedId
            val thumb = selected?.let { id ->
                tracked.firstOrNull { it.id == id }?.let { t -> cropThumbnail(bitmap, t) }
            }
            val w = bitmap.width
            val h = bitmap.height
            bitmap.recycle()

            runOnUiThread {
                overlay.update(tracked, w, h, smoothedFps, inferMs, activeDetector.providerLabel, thumb)
            }
        } catch (t: Throwable) {
            runOnUiThread { showError("Inference error: ${t.message}", showImport = true) }
        } finally {
            image.close()
        }
    }

    private fun cropThumbnail(bitmap: Bitmap, t: TrackedObject): Bitmap? {
        if (bitmap.width < 2 || bitmap.height < 2) return null
        val mx = t.width * 0.20f
        val my = t.height * 0.20f
        val x1 = (t.left - mx).toInt().coerceIn(0, bitmap.width - 2)
        val y1 = (t.top - my).toInt().coerceIn(0, bitmap.height - 2)
        val x2 = (t.right + mx).toInt().coerceIn(x1 + 1, bitmap.width)
        val y2 = (t.bottom + my).toInt().coerceIn(y1 + 1, bitmap.height)
        val crop = Bitmap.createBitmap(bitmap, x1, y1, x2 - x1, y2 - y1)
        val out = Bitmap.createScaledBitmap(crop, 360, 240, true)
        if (out !== crop) crop.recycle()
        return out
    }

    private fun showError(message: String, showImport: Boolean) {
        errorText.text = message
        errorText.visibility = View.VISIBLE
        importModelButton.visibility = if (showImport) View.VISIBLE else View.GONE
    }

    private fun clearError() {
        errorText.visibility = View.GONE
        importModelButton.visibility = View.GONE
    }

    override fun onDestroy() {
        detector?.close()
        executor.shutdown()
        super.onDestroy()
    }
}
