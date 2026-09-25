package com.foto6.spectratrack

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

class OnnxDetector private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val labels: List<String>,
    val providerLabel: String,
    private val inputSize: Int,
    private val confidence: Float,
    private val iouThreshold: Float,
) : AutoCloseable {
    companion object {
        fun fromBytes(
            context: Context,
            modelBytes: ByteArray,
            labelsAsset: String = "coco80.txt",
            inputSize: Int = 640,
            confidence: Float = 0.35f,
            iouThreshold: Float = 0.45f,
        ): OnnxDetector {
            val env = OrtEnvironment.getEnvironment()
            val labels = context.assets.open(labelsAsset).bufferedReader().useLines {
                it.filter(String::isNotBlank).toList()
            }
            val options = OrtSession.SessionOptions()
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            var provider = "CPU"
            try {
                options.addNnapi()
                provider = "NNAPI"
            } catch (_: Throwable) {
            }
            val session = env.createSession(modelBytes, options)
            return OnnxDetector(env, session, labels, provider, inputSize, confidence, iouThreshold)
        }

        fun fromFile(
            context: Context,
            file: File,
            inputSize: Int = 640,
            confidence: Float = 0.35f,
            iouThreshold: Float = 0.45f,
        ): OnnxDetector = fromBytes(context, file.readBytes(), inputSize = inputSize, confidence = confidence, iouThreshold = iouThreshold)

        fun fromAssetIfPresent(
            context: Context,
            assetName: String = "yolo11n.onnx",
            inputSize: Int = 640,
            confidence: Float = 0.35f,
            iouThreshold: Float = 0.45f,
        ): OnnxDetector? {
            return try {
                val bytes = context.assets.open(assetName).use { it.readBytes() }
                fromBytes(context, bytes, inputSize = inputSize, confidence = confidence, iouThreshold = iouThreshold)
            } catch (_: java.io.FileNotFoundException) {
                null
            }
        }
    }

    fun detect(source: Bitmap): List<Detection> {
        val prepared = letterbox(source)
        val input = bitmapToTensor(prepared.bitmap)
        val inputName = session.inputNames.first()
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong()))
        tensor.use {
            session.run(mapOf(inputName to tensor)).use { result ->
                val out = result[0] as OnnxTensor
                val shape = out.info.shape
                val buffer = out.floatBuffer
                val data = FloatArray(buffer.remaining())
                buffer.get(data)
                return decode(data, shape, source.width, source.height, prepared.scale, prepared.padX, prepared.padY)
            }
        }
    }

    private data class Prepared(val bitmap: Bitmap, val scale: Float, val padX: Float, val padY: Float)

    private fun letterbox(src: Bitmap): Prepared {
        val scale = min(inputSize.toFloat() / src.width, inputSize.toFloat() / src.height)
        val nw = max(1, (src.width * scale).toInt())
        val nh = max(1, (src.height * scale).toInt())
        val resized = Bitmap.createScaledBitmap(src, nw, nh, true)
        val canvasBitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(canvasBitmap)
        canvas.drawColor(Color.rgb(114, 114, 114))
        val padX = (inputSize - nw) / 2f
        val padY = (inputSize - nh) / 2f
        canvas.drawBitmap(resized, padX, padY, Paint(Paint.FILTER_BITMAP_FLAG))
        if (resized !== src) resized.recycle()
        return Prepared(canvasBitmap, scale, padX, padY)
    }

    private fun bitmapToTensor(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        bitmap.recycle()
        val plane = inputSize * inputSize
        val out = FloatArray(plane * 3)
        pixels.forEachIndexed { i, c ->
            out[i] = Color.red(c) / 255f
            out[plane + i] = Color.green(c) / 255f
            out[plane * 2 + i] = Color.blue(c) / 255f
        }
        return out
    }

    private data class Candidate(
        val left: Float, val top: Float, val right: Float, val bottom: Float,
        val score: Float, val classId: Int,
    )

    private fun decode(
        data: FloatArray,
        shape: LongArray,
        sourceW: Int,
        sourceH: Int,
        scale: Float,
        padX: Float,
        padY: Float,
    ): List<Detection> {
        require(shape.size == 3) { "Unsupported output shape: ${shape.contentToString()}" }
        val a = shape[1].toInt()
        val b = shape[2].toInt()
        val transposed = a < b && a <= 512
        val rows = if (transposed) b else a
        val features = if (transposed) a else b
        require(features >= 6) { "Output has too few features: $features" }

        fun value(row: Int, feature: Int): Float =
            if (transposed) data[feature * rows + row] else data[row * features + feature]

        var normalized = true
        val checks = min(rows, 32)
        for (r in 0 until checks) {
            if (value(r, 0) > 2.5f || value(r, 2) > 2.5f) {
                normalized = false
                break
            }
        }

        val candidates = ArrayList<Candidate>()
        for (r in 0 until rows) {
            var bestClass = 0
            var bestScore = Float.NEGATIVE_INFINITY
            for (f in 4 until features) {
                val s = value(r, f)
                if (s > bestScore) {
                    bestScore = s
                    bestClass = f - 4
                }
            }
            if (bestScore < confidence) continue
            var cx = value(r, 0)
            var cy = value(r, 1)
            var bw = value(r, 2)
            var bh = value(r, 3)
            if (normalized) {
                cx *= inputSize
                cy *= inputSize
                bw *= inputSize
                bh *= inputSize
            }
            val x1 = ((cx - bw / 2f) - padX) / scale
            val y1 = ((cy - bh / 2f) - padY) / scale
            val x2 = ((cx + bw / 2f) - padX) / scale
            val y2 = ((cy + bh / 2f) - padY) / scale
            candidates += Candidate(
                x1.coerceIn(0f, sourceW - 1f), y1.coerceIn(0f, sourceH - 1f),
                x2.coerceIn(0f, sourceW - 1f), y2.coerceIn(0f, sourceH - 1f),
                bestScore, bestClass,
            )
        }

        val sorted = candidates.sortedByDescending { it.score }
        val kept = mutableListOf<Candidate>()
        sorted.forEach { c ->
            val suppressed = kept.any { k -> c.classId == k.classId && iou(c, k) >= iouThreshold }
            if (!suppressed) kept += c
        }
        return kept.map { c ->
            val label = labels.getOrElse(c.classId) { "class_${c.classId}" }
            Detection(c.left, c.top, c.right, c.bottom, c.score, c.classId, label)
        }
    }

    private fun iou(a: Candidate, b: Candidate): Float {
        val x1 = max(a.left, b.left)
        val y1 = max(a.top, b.top)
        val x2 = min(a.right, b.right)
        val y2 = min(a.bottom, b.bottom)
        val inter = max(0f, x2 - x1) * max(0f, y2 - y1)
        val aa = max(0f, a.right - a.left) * max(0f, a.bottom - a.top)
        val ba = max(0f, b.right - b.left) * max(0f, b.bottom - b.top)
        return inter / max(aa + ba - inter, 1e-6f)
    }

    override fun close() {
        session.close()
    }
}
