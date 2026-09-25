package com.foto6.spectratrack

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

class HudOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {
    private val line = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 2f
    }
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 30f
        typeface = android.graphics.Typeface.MONOSPACE
    }
    private val dimText = Paint(text).apply { color = Color.rgb(200, 200, 200); textSize = 23f }
    private val panel = Paint().apply { color = Color.argb(180, 0, 0, 0); style = Paint.Style.FILL }

    @Volatile var selectedId: Int? = null
        private set
    private var tracks: List<TrackedObject> = emptyList()
    private var fps = 0f
    private var inferMs = 0f
    private var provider = "CPU"
    private var sourceW = 1
    private var sourceH = 1
    private var thumbnail: Bitmap? = null

    fun update(
        objects: List<TrackedObject>, sourceWidth: Int, sourceHeight: Int,
        fps: Float, inferenceMs: Float, provider: String, targetThumbnail: Bitmap?,
    ) {
        this.tracks = objects
        this.sourceW = max(1, sourceWidth)
        this.sourceH = max(1, sourceHeight)
        this.fps = fps
        this.inferMs = inferenceMs
        this.provider = provider
        val oldThumb = this.thumbnail
        if (oldThumb != null && oldThumb !== targetThumbnail && !oldThumb.isRecycled) oldThumb.recycle()
        this.thumbnail = targetThumbnail
        if (selectedId != null && objects.none { it.id == selectedId }) selectedId = null
        invalidate()
    }

    private fun contentRect(): RectF {
        val srcRatio = sourceW.toFloat() / sourceH
        val viewRatio = width.toFloat() / max(1, height)
        return if (viewRatio > srcRatio) {
            val drawW = height * srcRatio
            val left = (width - drawW) / 2f
            RectF(left, 0f, left + drawW, height.toFloat())
        } else {
            val drawH = width / srcRatio
            val top = (height - drawH) / 2f
            RectF(0f, top, width.toFloat(), top + drawH)
        }
    }

    private fun mapX(x: Float, r: RectF) = r.left + x / sourceW * r.width()
    private fun mapY(y: Float, r: RectF) = r.top + y / sourceH * r.height()

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val r = contentRect()
        tracks.forEach { tr ->
            val box = RectF(mapX(tr.left, r), mapY(tr.top, r), mapX(tr.right, r), mapY(tr.bottom, r))
            drawCornerBox(canvas, box, tr.id == selectedId)
            canvas.drawText("T%03d %s %.2f".format(tr.id, tr.label.uppercase(), tr.score), box.left, max(30f, box.top - 8f), dimText)
            val pts = tr.history.toList()
            for (i in 1 until pts.size) {
                canvas.drawLine(mapX(pts[i - 1].first, r), mapY(pts[i - 1].second, r), mapX(pts[i].first, r), mapY(pts[i].second, r), line)
            }
        }
        drawReticle(canvas, r.centerX(), r.centerY())
        canvas.drawRect(0f, 0f, min(width.toFloat(), 620f), 100f, panel)
        canvas.drawText("SPECTRATRACK // LOCAL", 18f, 36f, text)
        canvas.drawText("FPS %.1f | %.1f ms | %s | TRACKS %d".format(fps, inferMs, provider, tracks.size), 18f, 72f, dimText)
        drawTargetPanel(canvas, r)
    }

    private fun drawTargetPanel(canvas: Canvas, content: RectF) {
        val target = tracks.firstOrNull { it.id == selectedId } ?: return
        val landscape = width > height
        val pw = if (landscape) min(width * 0.31f, 500f) else min(width * 0.46f, 460f)
        val ph = if (landscape) min(height * 0.58f, 560f) else min(height * 0.30f, 420f)
        val left = width - pw - 16f
        val top = 16f
        canvas.drawRoundRect(RectF(left, top, width - 16f, top + ph), 14f, 14f, panel)
        canvas.drawText("TARGET T%03d".format(target.id), left + 16f, top + 34f, text)
        thumbnail?.let { bmp ->
            val dest = RectF(left + 14f, top + 52f, width - 30f, top + ph * 0.62f)
            canvas.drawBitmap(bmp, null, dest, null)
        }
        val speed = hypot(target.vx, target.vy)
        val y0 = top + ph * 0.69f
        canvas.drawText(target.label.uppercase(), left + 16f, y0, dimText)
        canvas.drawText("CONF %.3f".format(target.score), left + 16f, y0 + 30f, dimText)
        canvas.drawText("MOTION %.1f px/f".format(speed), left + 16f, y0 + 60f, dimText)
        canvas.drawText("AGE ${target.age}  LOST ${target.missed}", left + 16f, y0 + 90f, dimText)
    }

    private fun drawReticle(canvas: Canvas, cx: Float, cy: Float) {
        val rad = min(width, height) * 0.045f
        canvas.drawCircle(cx, cy, rad, line)
        canvas.drawLine(cx - rad - 26f, cy, cx - rad + 4f, cy, line)
        canvas.drawLine(cx + rad - 4f, cy, cx + rad + 26f, cy, line)
        canvas.drawLine(cx, cy - rad - 26f, cx, cy - rad + 4f, line)
        canvas.drawLine(cx, cy + rad - 4f, cx, cy + rad + 26f, line)
    }

    private fun drawCornerBox(canvas: Canvas, b: RectF, selected: Boolean) {
        line.strokeWidth = if (selected) 4f else 2f
        val len = max(18f, min(b.width(), b.height()) * 0.22f)
        canvas.drawLine(b.left, b.top, b.left + len, b.top, line)
        canvas.drawLine(b.left, b.top, b.left, b.top + len, line)
        canvas.drawLine(b.right, b.top, b.right - len, b.top, line)
        canvas.drawLine(b.right, b.top, b.right, b.top + len, line)
        canvas.drawLine(b.left, b.bottom, b.left + len, b.bottom, line)
        canvas.drawLine(b.left, b.bottom, b.left, b.bottom - len, line)
        canvas.drawLine(b.right, b.bottom, b.right - len, b.bottom, line)
        canvas.drawLine(b.right, b.bottom, b.right, b.bottom - len, line)
        line.strokeWidth = 2f
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (event.action != MotionEvent.ACTION_DOWN) return true
        val r = contentRect()
        val sx = ((event.x - r.left) / r.width() * sourceW).coerceIn(0f, sourceW.toFloat())
        val sy = ((event.y - r.top) / r.height() * sourceH).coerceIn(0f, sourceH .toFloat())
        val hit = tracks.lastOrNull { sx in it.left..it.right && sy in it.top..it.bottom }
        selectedId = if (hit?.id == selectedId) null else hit?.id
        invalidate()
        return true
    }
}